from __future__ import annotations

import json
from dataclasses import replace

import pytest

from blueprinting.application import (
    CanonicalIRStage,
    DerivationDebugBundleCodec,
    DerivationStage,
    DerivationTrace,
    DerivationTransition,
    PassContractView,
    boundary_view,
    graph_view,
)
from blueprinting.schema.errors import SerializationError
from blueprinting.synthesizer.stages.concrete_plan.ir import ConcretePlanIR
from blueprinting.synthesizer.stages.distributed.ir import DistributedTaskIR
from blueprinting.synthesizer.stages.machine.ir import MachineIR
from blueprinting.synthesizer.stages.model.ir import ModelIR
from blueprinting.synthesizer.stages.portable_plan.ir import PortablePlanIR
from blueprinting.workbench.ir_expressions import (
    canonical_ir_expression,
    lowering_correspondence_rows,
    lowering_expression,
)
from blueprinting.workbench.presentation import semantic_boundary_rows


def _contract(name: str, source: object, target: object) -> PassContractView:
    return PassContractView(
        name=name,
        input_schema=type(source).__name__,
        output_schema=type(target).__name__,
        required_bindings=(),
        required_analyses=(),
        produced_analyses=(),
        mutation_model="immutable",
        verification="both",
        deterministic=True,
    )


def _full_trace(
    model_ir: ModelIR,
    distributed_ir: DistributedTaskIR,
    portable_ir: PortablePlanIR,
    concrete_ir: ConcretePlanIR,
    machine_ir: MachineIR,
) -> DerivationTrace:
    rows = (
        (CanonicalIRStage.MODEL, "model", model_ir),
        (CanonicalIRStage.DISTRIBUTED, "distributed", distributed_ir),
        (CanonicalIRStage.PORTABLE, "portable", portable_ir),
        (CanonicalIRStage.CONCRETE, "concrete", concrete_ir),
        (CanonicalIRStage.MACHINE, "machine", machine_ir),
    )
    stages = tuple(DerivationStage(stage, "training", label, label, 1, ir) for stage, label, ir in rows)
    transitions = []
    for source, target in zip(stages, stages[1:]):
        contract = _contract(target.pass_name, source.ir, target.ir)
        transitions.append(
            DerivationTransition(
                source.digest,
                target.digest,
                contract,
                1,
                boundary_view(source.ir, target.ir, contract.name),
            )
        )
    return DerivationTrace("request", "session", stages, tuple(transitions))


def test_all_five_canonical_layers_have_structural_graph_adapters(
    model_ir: ModelIR,
    distributed_ir: DistributedTaskIR,
    portable_ir: PortablePlanIR,
    concrete_ir: ConcretePlanIR,
    machine_ir: MachineIR,
) -> None:
    graphs = tuple(graph_view(item) for item in (model_ir, distributed_ir, portable_ir, concrete_ir, machine_ir))

    assert tuple(item.stage for item in graphs) == tuple(CanonicalIRStage)
    assert all(item.nodes for item in graphs)
    assert all(item.edges for item in graphs)
    assert {item.ref.kind for item in graphs[3].nodes} >= {"device", "queue", "buffer", "command"}
    assert {item.ref.kind for item in graphs[4].nodes} >= {"section", "instruction", "entry_point"}


def test_all_five_canonical_layers_have_short_and_detailed_typed_expressions(
    model_ir: ModelIR,
    distributed_ir: DistributedTaskIR,
    portable_ir: PortablePlanIR,
    concrete_ir: ConcretePlanIR,
    machine_ir: MachineIR,
) -> None:
    expressions = tuple(
        canonical_ir_expression(item) for item in (model_ir, distributed_ir, portable_ir, concrete_ir, machine_ir)
    )

    assert [item.short.split(maxsplit=1)[0] for item in expressions] == [
        "model",
        "distributed",
        "portable_plan",
        "concrete_plan",
        "machine_program",
    ]
    for ir, expression in zip((model_ir, distributed_ir, portable_ir, concrete_ir, machine_ir), expressions):
        assert str(ir.header.schema_version) in expression.short
        assert ir.digest in expression.detailed
        assert "lineage=" in expression.detailed or isinstance(ir, MachineIR) and not ir.instructions


def test_lowering_expression_exposes_contract_rules_and_evidence(
    model_ir: ModelIR,
    distributed_ir: DistributedTaskIR,
) -> None:
    contract = _contract("transformer-distribute", model_ir, distributed_ir)
    transition = DerivationTransition(
        model_ir.digest,
        distributed_ir.digest,
        contract,
        1,
        boundary_view(model_ir, distributed_ir, contract.name),
    )
    source_graph = graph_view(model_ir)
    target_graph = graph_view(distributed_ir)
    rows = semantic_boundary_rows(transition.boundary, source_graph, target_graph)

    expression = lowering_expression(transition, rows)

    assert "pass @transformer-distribute" in expression.short
    assert "transaction immutable; verify=both; deterministic=true; seed=false" in expression.short
    assert "commit_gate verify(input) -> run immutable transformation -> verify(output)" in expression.detailed
    assert "rules {" in expression.detailed
    assert "canonical_relations=" in expression.detailed


def test_correspondence_marks_undeclared_transforms_as_lineage_evidence(
    model_ir: ModelIR,
    distributed_ir: DistributedTaskIR,
) -> None:
    contract = _contract("transformer-distribute", model_ir, distributed_ir)
    transition = DerivationTransition(
        model_ir.digest,
        distributed_ir.digest,
        contract,
        1,
        boundary_view(model_ir, distributed_ir, contract.name),
    )
    rows = semantic_boundary_rows(transition.boundary, graph_view(model_ir), graph_view(distributed_ir))

    correspondence = lowering_correspondence_rows(transition, rows)

    assert correspondence
    assert all(not row["rule_declared"] for row in correspondence)
    assert all("<undeclared> ⇒ <lineage evidence>" in row["transform_expression"] for row in correspondence)
    assert all("correspondence is evidence only" in row["transform_expression"] for row in correspondence)


def test_boundary_mapping_uses_typed_lineage_and_reports_explicit_mismatch(
    model_ir: ModelIR,
    distributed_ir: DistributedTaskIR,
) -> None:
    boundary = boundary_view(model_ir, distributed_ir, "distribute")

    assert boundary.summary.target_entities == len(distributed_ir.tasks) + len(distributed_ir.values)
    assert boundary.summary.mapped_source_entities == len(model_ir.operations) + len(model_ir.values)
    assert boundary.summary.dangling_sources == 0
    assert boundary.summary.one_to_many > 0

    changed_value = replace(distributed_ir.values[0], source_value=model_ir.values[-1].id)
    changed = replace(distributed_ir, values=(changed_value,) + distributed_ir.values[1:])
    mismatch = boundary_view(model_ir, changed, "distribute")

    assert any(item.code == "lineage.explicit_source_mismatch" for item in mismatch.diagnostics)


def test_debug_bundle_round_trip_revalidates_a_full_five_layer_chain(
    model_ir: ModelIR,
    distributed_ir: DistributedTaskIR,
    portable_ir: PortablePlanIR,
    concrete_ir: ConcretePlanIR,
    machine_ir: MachineIR,
) -> None:
    trace = _full_trace(model_ir, distributed_ir, portable_ir, concrete_ir, machine_ir)

    encoded = DerivationDebugBundleCodec.dumps(trace)
    restored = DerivationDebugBundleCodec.loads(encoded)

    assert tuple(item.stage for item in restored.stages) == tuple(CanonicalIRStage)
    assert tuple(item.digest for item in restored.stages) == tuple(item.digest for item in trace.stages)
    assert len(restored.transitions) == 4
    assert DerivationDebugBundleCodec.dumps(restored) == encoded


def test_debug_bundle_rejects_tampered_snapshot(
    model_ir: ModelIR,
    distributed_ir: DistributedTaskIR,
    portable_ir: PortablePlanIR,
    concrete_ir: ConcretePlanIR,
    machine_ir: MachineIR,
) -> None:
    payload = json.loads(
        DerivationDebugBundleCodec.dumps(_full_trace(model_ir, distributed_ir, portable_ir, concrete_ir, machine_ir))
    )
    snapshot = json.loads(payload["stages"][0]["snapshot_json"])
    snapshot["fields"]["content_digest"] = "0" * 40
    payload["stages"][0]["snapshot_json"] = json.dumps(snapshot)

    with pytest.raises(SerializationError, match="digest mismatch"):
        DerivationDebugBundleCodec.loads(json.dumps(payload))


def test_debug_bundle_rejects_a_missing_adjacent_transition(
    model_ir: ModelIR,
    distributed_ir: DistributedTaskIR,
    portable_ir: PortablePlanIR,
    concrete_ir: ConcretePlanIR,
    machine_ir: MachineIR,
) -> None:
    payload = json.loads(
        DerivationDebugBundleCodec.dumps(_full_trace(model_ir, distributed_ir, portable_ir, concrete_ir, machine_ir))
    )
    payload["transitions"].pop()

    with pytest.raises(ValueError, match="cover every adjacent stage"):
        DerivationDebugBundleCodec.loads(json.dumps(payload))

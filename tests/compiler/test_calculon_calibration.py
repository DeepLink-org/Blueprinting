from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprinting.compiler.analysis.cost_model import CalibrationMode, HardwareProfile, estimate_iteration
from blueprinting.compiler.analysis.transformer_workload import EngineKind, PrimitiveInvocation, TrainingPhase
from blueprinting.compiler.experiments import discover_seqsel_tab5_cases, run_calculon_experiment
from blueprinting.compiler.lowering import DistributeTransformerTrainingPass, PlanTransformerTrainingPass
from blueprinting.compiler.models import (
    TransformerExecutionSpec,
    TransformerModelSpec,
    build_transformer_model_ir,
    compilation_session_for,
)
from blueprinting.compiler.passes import PassManager, PassPipeline

ROOT = Path(__file__).resolve().parents[2]


def _json(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _compile(model_name: str, mode: str):
    model_data = _json(ROOT / "data" / "models" / f"{model_name}.json")
    execution_data = _json(ROOT / "data" / "validation" / "seqsel" / "tab5" / f"{model_name}_{mode}.json")
    model = TransformerModelSpec.from_mapping(model_name, model_data)
    execution = TransformerExecutionSpec.from_mapping(execution_data)
    source = build_transformer_model_ir(model)
    result = PassManager().run(
        PassPipeline.of(DistributeTransformerTrainingPass(), PlanTransformerTrainingPass()),
        source,
        session=compilation_session_for(model, execution),
    )
    return model, execution, source, result


def test_transformer_lowering_produces_auditable_ir_checkpoints():
    _, _, source, result = _compile("gpt3-175B", "seqsel")

    assert source.require_valid() is None
    assert tuple(record.pass_name for record in result.records) == (
        "transformer-distribute-v1",
        "transformer-plan-work-v1",
    )
    assert tuple(checkpoint.ir.header.schema_name for checkpoint in result.checkpoints) == (
        "blueprinting.distributed-task",
        "blueprinting.portable-plan",
    )
    assert result.ir.require_valid() is None
    assert all(isinstance(task.attributes["invocation"], PrimitiveInvocation) for task in result.ir.tasks)


def test_selective_recompute_is_structural_and_linear_gradients_are_derived():
    _, _, _, result = _compile("gpt3-175B", "seqsel")
    invocations = tuple(task.attributes["invocation"] for task in result.ir.tasks)
    recomputed_layers = {
        invocation.source_layer for invocation in invocations if invocation.phase is TrainingPhase.RECOMPUTE
    }

    assert recomputed_layers == {
        "attention.query_key",
        "attention.softmax",
        "attention.probability_dropout",
    }
    query = {
        invocation.phase: invocation.work.operations
        for invocation in invocations
        if invocation.source_layer == "attention.query" and invocation.engine is EngineKind.MATRIX
    }
    assert query[TrainingPhase.FORWARD] == query[TrainingPhase.ACTIVATION_GRADIENT]
    assert query[TrainingPhase.FORWARD] == query[TrainingPhase.WEIGHT_GRADIENT]


def test_hardware_evidence_is_shared_and_does_not_change_workload():
    _, execution, _, result = _compile("gpt3-175B", "full")
    hardware = HardwareProfile.from_mapping(
        "a100_80g",
        _json(ROOT / "data" / "systems" / "a100_80g.json"),
        datatype=execution.datatype,
    )
    digests_before = tuple(task.workload for task in result.ir.tasks)

    peak = estimate_iteration(result.ir, hardware, CalibrationMode.PEAK_ONLY)
    calibrated = estimate_iteration(result.ir, hardware, CalibrationMode.SYSTEM_EVIDENCE)

    assert peak.total < calibrated.total
    assert tuple(task.workload for task in result.ir.tasks) == digests_before
    assert result.ir.attributes.get("duration") is None


@pytest.mark.parametrize("case_index", range(8))
def test_seqsel_cases_match_calculon_without_case_fitting(case_index: int):
    case = discover_seqsel_tab5_cases(ROOT / "data")[case_index]
    report = run_calculon_experiment((case,))
    result = report.cases[0]

    assert max(abs(metric.relative_error_percent) for metric in result.workload.values()) < 1e-9
    assert abs(result.calibrated_error_percent) < 1e-9
    assert abs(result.calibrated.memory.total - result.calculon_stats["proc_mem_tier1_cap_req"]) < 1
    assert report.calibration_policy["fit_against_case_outputs"] is False
    assert "Calculon duration" in report.calibration_policy["forbidden_inputs"]


def test_explicit_recompute_does_not_copy_calculon_prefix_counter():
    case = discover_seqsel_tab5_cases(ROOT / "data")[0]
    result = run_calculon_experiment((case,)).cases[0]
    audit = result.to_dict()["recompute_counter_audit"]

    assert audit["compiled_explicit_operations"] < audit["calculon_block_re_flops"]
    assert result.calibrated.recompute == pytest.approx(result.calculon_stats["recompute_time"], rel=1e-12)

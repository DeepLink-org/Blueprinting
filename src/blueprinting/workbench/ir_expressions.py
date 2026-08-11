"""Typed short and detailed text projections for canonical IR and lowering.

These expressions are rebuildable presentation views. They expose canonical
semantics and pass mechanics without becoming another serialization format.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from blueprinting.application import DerivationTransition
from blueprinting.synthesizer.expr import (
    Add,
    CeilDivide,
    Divide,
    Maximum,
    Minimum,
    Multiply,
    Subtract,
    Symbol,
)
from blueprinting.synthesizer.stages.common import TensorType
from blueprinting.synthesizer.stages.concrete_plan.ir import ConcretePlanIR
from blueprinting.synthesizer.stages.distributed.ir import Collective, DistributedTaskIR
from blueprinting.synthesizer.stages.machine.ir import MachineIR
from blueprinting.synthesizer.stages.model.ir import ModelIR
from blueprinting.synthesizer.stages.portable_plan.ir import PortablePlanIR


@dataclass(frozen=True)
class CanonicalIRExpression:
    short: str
    detailed: str


@dataclass(frozen=True)
class LoweringExpression:
    short: str
    detailed: str


def _refs(values: tuple[Any, ...]) -> str:
    return "[" + ", ".join(str(item) for item in values) + "]"


def _scalar(value: Any) -> str:
    if isinstance(value, Symbol):
        return f"${value.axis.value}.{value.name}"
    match value:
        case Add(terms=values):
            return "(" + " + ".join(_scalar(item) for item in values) + ")"
        case Subtract(left=left, right=right):
            return f"({_scalar(left)} - {_scalar(right)})"
        case Multiply(factors=values):
            return "(" + " * ".join(_scalar(item) for item in values) + ")"
        case Divide(numerator=left, denominator=right):
            return f"({_scalar(left)} / {_scalar(right)})"
        case CeilDivide(numerator=left, denominator=right):
            return f"ceil_div({_scalar(left)}, {_scalar(right)})"
        case Maximum(values=values):
            return f"max({', '.join(_scalar(item) for item in values)})"
        case Minimum(values=values):
            return f"min({', '.join(_scalar(item) for item in values)})"
        case _:
            return str(value)


def _tensor(value: TensorType) -> str:
    shape = "x".join(_scalar(item) for item in value.shape)
    layout = "" if value.layout is None else f", layout={value.layout}"
    return f"tensor<{shape}x{value.dtype}{layout}>"


def _lineage(value: Any) -> str:
    sources = _refs(value.sources)
    return f"{value.kind.value} via @{value.transform} from {sources}"


def _header(ir: Any) -> str:
    parents = ", ".join(ir.header.parent_digests) or "none"
    return (
        f"schema {ir.header.schema_name}@{ir.header.schema_version}\n"
        f"digest {ir.digest}\n"
        f"producer {ir.header.producer_version}\n"
        f"parents [{parents}]"
    )


def _model_expression(ir: ModelIR) -> CanonicalIRExpression:
    values = {item.id: item for item in ir.values}
    short = [f"model @{ir.name} : {ir.header.schema_name}@{ir.header.schema_version} {{"]
    for identifier in ir.inputs:
        value = values[identifier]
        short.append(f"  input %{value.name or value.id} : {_tensor(value.type)}")
    for operation in ir.operations:
        inputs = ", ".join(f"%{values[item].name or item}" for item in operation.inputs)
        outputs = ", ".join(f"%{values[item].name or item}" for item in operation.outputs)
        short.append(f"  {outputs} = {operation.operation}({inputs})")
    for identifier in ir.outputs:
        value = values[identifier]
        short.append(f"  return %{value.name or value.id} : {_tensor(value.type)}")
    short.append("}")

    detailed = [_header(ir), "", "values {"]
    for value in ir.values:
        detailed.append(
            f"  {value.id} name={value.name!r} role={value.role.value} type={_tensor(value.type)} "
            f"lineage=({_lineage(value.lineage)})"
        )
    detailed.append("}")
    detailed.append("operations {")
    for operation in ir.operations:
        detailed.append(
            f"  {operation.id} op={operation.operation} inputs={_refs(operation.inputs)} "
            f"outputs={_refs(operation.outputs)} control_deps={_refs(operation.control_dependencies)} "
            f"effects={len(operation.effects)} lineage=({_lineage(operation.lineage)})"
        )
    detailed.append("}")
    detailed.append(f"interface inputs={_refs(ir.inputs)} outputs={_refs(ir.outputs)}")
    return CanonicalIRExpression("\n".join(short), "\n".join(detailed))


def _distributed_expression(ir: DistributedTaskIR) -> CanonicalIRExpression:
    axes = ", ".join(f"{item.name}={item.size}" for item in ir.mesh.axes)
    phase_kinds: Counter[tuple[str, str]] = Counter()
    for task in ir.tasks:
        invocation = getattr(task.semantic, "invocation", None)
        phase = getattr(getattr(invocation, "phase", None), "value", "unscoped")
        phase_kinds[(phase, task.body_tag)] += 1
    short = [
        f"distributed @{ir.name} : {ir.header.schema_name}@{ir.header.schema_version} {{",
        f"  mesh @{ir.mesh.name}<{axes}> size={ir.mesh.size}",
    ]
    for (phase, kind), count in sorted(phase_kinds.items()):
        short.append(f"  phase @{phase} {{ {kind} x {count} }}")
    short.append(f"  interface values={len(ir.values)} inputs={_refs(ir.inputs)} outputs={_refs(ir.outputs)}")
    short.append("}")

    detailed = [_header(ir), "", f"mesh @{ir.mesh.name}<{axes}> size={ir.mesh.size}", "values {"]
    for value in ir.values:
        detailed.append(
            f"  {value.id} role={value.role.value} type={_tensor(value.type)} owners={_refs(value.owners)} "
            f"sharding={value.sharding} lineage=({_lineage(value.lineage)})"
        )
    detailed.append("}")
    detailed.append("tasks {")
    for task in ir.tasks:
        match task.body:
            case Collective(spec=spec):
                body = f"Collective({spec})"
            case _:
                body = type(task.body).__name__
        detailed.append(
            f"  {task.id} body={body} op={task.operation} ranks={_refs(task.ranks)} "
            f"inputs={_refs(task.inputs)} outputs={_refs(task.outputs)} deps={_refs(task.dependencies)} "
            f"lineage=({_lineage(task.lineage)})"
        )
    detailed.append("}")
    return CanonicalIRExpression("\n".join(short), "\n".join(detailed))


def _portable_expression(ir: PortablePlanIR) -> CanonicalIRExpression:
    phase_kinds: Counter[tuple[str, str]] = Counter()
    for task in ir.tasks:
        phase = getattr(getattr(task.semantic, "phase", None), "value", "unscoped")
        phase_kinds[(phase, task.kind.value)] += 1
    objectives = ", ".join(f"{item.kind.value}:{item.direction.value}" for item in ir.objectives)
    short = [
        f"portable_plan @{ir.name} : {ir.header.schema_name}@{ir.header.schema_version} {{",
        f"  strategy {ir.strategy_fingerprint}",
        f"  planner {ir.planner_revision}",
    ]
    for (phase, kind), count in sorted(phase_kinds.items()):
        short.append(f"  phase @{phase} {{ {kind} x {count} }}")
    short.append(f"  buffers {len(ir.buffers)}; objectives [{objectives}]")
    short.append("  target_binding none")
    short.append("}")

    detailed = [_header(ir), "", f"strategy {ir.strategy_fingerprint}", f"planner {ir.planner_revision}", "buffers {"]
    for buffer in ir.buffers:
        detailed.append(
            f"  {buffer.id} role={buffer.role.value} storage={buffer.storage_class.value} "
            f"size_bytes={_scalar(buffer.size_bytes)} alignment={buffer.alignment_bytes} producer={buffer.producer} "
            f"consumers={_refs(buffer.consumers)} lineage=({_lineage(buffer.lineage)})"
        )
    detailed.append("}")
    detailed.append("tasks {")
    for task in ir.tasks:
        workload = task.workload
        resources = ", ".join(
            f"{item.kind.value}:{_scalar(item.quantity)}/{item.scope.value}" for item in task.resources
        )
        implementations = ", ".join(
            f"{item.capability}[{', '.join(item.alternatives)}]" for item in task.implementations
        )
        detailed.append(
            f"  {task.id} kind={task.kind.value} op={task.operation} ranks={_refs(task.logical_ranks)} "
            f"deps={_refs(task.dependencies)} io={_refs(task.inputs)}->{_refs(task.outputs)} "
            f"workload=(ops={_scalar(workload.operations)}, read={_scalar(workload.read_bytes)}, "
            f"write={_scalar(workload.write_bytes)}, message={_scalar(workload.message_bytes)}, "
            f"temp={_scalar(workload.temporary_bytes)}, persistent={_scalar(workload.persistent_bytes)}) "
            f"resources=[{resources}] implementations=[{implementations}] concurrency={task.concurrency_group} "
            f"lineage=({_lineage(task.lineage)})"
        )
    detailed.append("}")
    detailed.append(f"objectives [{objectives}]")
    return CanonicalIRExpression("\n".join(short), "\n".join(detailed))


def _concrete_expression(ir: ConcretePlanIR) -> CanonicalIRExpression:
    command_kinds = Counter(item.kind.value for item in ir.commands)
    short = [
        f"concrete_plan @{ir.name} : {ir.header.schema_name}@{ir.header.schema_version} {{",
        f"  target {ir.target_fingerprint}; deployment {ir.deployment_fingerprint}",
        f"  devices {len(ir.devices)}; queues {len(ir.queues)}; regions {len(ir.memory_regions)}",
        "  commands " + ", ".join(f"{kind} x {count}" for kind, count in sorted(command_kinds.items())),
        "  predicted_time not-canonical",
        "}",
    ]
    detailed = [
        _header(ir),
        "",
        f"target={ir.target_fingerprint} deployment={ir.deployment_fingerprint} abi={ir.abi_revision}",
        "devices {",
    ]
    detailed.extend(f"  {item.id} rank={item.logical_rank} target={item.target_device}" for item in ir.devices)
    detailed.append("}")
    detailed.append("queues {")
    detailed.extend(
        f"  {item.id} device={item.device} kind={item.kind.value} engine={item.engine} ordered={item.ordered}"
        for item in ir.queues
    )
    detailed.append("}")
    detailed.append("commands {")
    for item in ir.commands:
        detailed.append(
            f"  {item.id} kind={item.kind.value} queue={item.queue} implementation={item.implementation} "
            f"deps={_refs(item.dependencies)} buffers={item.buffers} wait={_refs(item.wait_tokens)} "
            f"signal={_refs(item.signal_tokens)} lineage=({_lineage(item.lineage)})"
        )
    detailed.append("}")
    return CanonicalIRExpression("\n".join(short), "\n".join(detailed))


def _machine_expression(ir: MachineIR) -> CanonicalIRExpression:
    short = [
        f"machine_program @{ir.name} : {ir.header.schema_name}@{ir.header.schema_version} {{",
        f"  target_plugin {ir.target_plugin}; abi {ir.target_abi}; format {ir.program_format}",
    ]
    short.extend(
        f"  section @{section.name} kind={section.kind.value} instructions={len(section.instructions)} "
        f"data_bytes={len(section.data)}"
        for section in ir.sections
    )
    short.append("  entry_points [" + ", ".join(item.name for item in ir.entry_points) + "]")
    short.append("}")
    detailed = [
        _header(ir),
        "",
        f"plugin={ir.target_plugin} abi={ir.target_abi} emitter={ir.emitter_revision} format={ir.program_format}",
    ]
    for section in ir.sections:
        detailed.append(
            f"section @{section.name} kind={section.kind.value} alignment={section.alignment_bytes} "
            f"data_bytes={len(section.data)} {{"
        )
        for item in section.instructions:
            detailed.append(
                f"  {item.id} opcode={item.opcode} deps={_refs(item.dependencies)} operands={dict(item.operands)} "
                f"source_command={item.source_command} lineage=({_lineage(item.lineage)})"
            )
        detailed.append("}")
    detailed.append(
        "entry_points { " + ", ".join(f"@{item.name} -> {item.instruction}" for item in ir.entry_points) + " }"
    )
    return CanonicalIRExpression("\n".join(short), "\n".join(detailed))


def canonical_ir_expression(ir: Any) -> CanonicalIRExpression:
    if isinstance(ir, ModelIR):
        return _model_expression(ir)
    if isinstance(ir, DistributedTaskIR):
        return _distributed_expression(ir)
    if isinstance(ir, PortablePlanIR):
        return _portable_expression(ir)
    if isinstance(ir, ConcretePlanIR):
        return _concrete_expression(ir)
    if isinstance(ir, MachineIR):
        return _machine_expression(ir)
    raise TypeError(f"unsupported canonical IR type: {type(ir).__name__}")


def lowering_expression(transition: DerivationTransition, rows: list[dict[str, Any]]) -> LoweringExpression:
    contract = transition.contract
    rule_by_transform = {item.transform: item for item in contract.rules}
    transforms = [item.transform for item in contract.rules]
    for row in rows:
        if row["transform"] not in transforms:
            transforms.append(row["transform"])
    short = [
        f"pass @{contract.name} {{",
        f"  input  {contract.input_schema}",
        f"  output {contract.output_schema}",
        f"  requires bindings [{', '.join(contract.required_bindings) or 'none'}]",
        f"  requires analyses [{', '.join(contract.required_analyses) or 'none'}]",
        f"  produces analyses [{', '.join(contract.produced_analyses) or 'none'}]",
        f"  preserves analyses [{', '.join(contract.preserved_analyses) or 'none'}]",
        f"  transaction {contract.mutation_model}; verify={contract.verification}; "
        f"deterministic={str(contract.deterministic).lower()}; seed={str(contract.uses_session_seed).lower()}",
        f"  transition {transition.verification_status}; relations={transition.verified_relations}; "
        f"claims={transition.verified_claims}; canonical={str(transition.canonical_conformance is not None).lower()}",
        f"  normal_form @{contract.normal_form or 'none'}",
        "  rules {",
    ]
    for transform in transforms:
        rule = rule_by_transform.get(transform)
        if rule is None:
            short.append(f"    @{transform}: <undeclared semantic rule; lineage evidence only>")
        else:
            short.append(
                f"    @{transform}: {rule.source_entity} -> {rule.target_entity}; {rule.rewrite}; "
                f"invariant=@{rule.semantic_invariant or 'none'}"
            )
    short.extend(("  }", "}"))

    detailed = [
        f"pass @{contract.name}",
        f"input_schema  {contract.input_schema}",
        f"output_schema {contract.output_schema}",
        f"source_digest {transition.source_digest}",
        f"target_digest {transition.target_digest}",
        f"contract_digest {contract.contract_digest or 'unavailable'}",
        f"normal_form {contract.normal_form or 'none'}",
        f"required_bindings [{', '.join(contract.required_bindings) or 'none'}]",
        f"required_analyses [{', '.join(contract.required_analyses) or 'none'}]",
        f"produced_analyses [{', '.join(contract.produced_analyses) or 'none'}]",
        f"preserved_analyses [{', '.join(contract.preserved_analyses) or 'none'}]",
        f"mutation_model {contract.mutation_model}",
        f"verification {contract.verification}",
        f"deterministic {str(contract.deterministic).lower()}",
        f"uses_session_seed {str(contract.uses_session_seed).lower()}",
        f"transition_status {transition.verification_status}",
        f"canonical_conformance {transition.canonical_conformance or 'none'}",
        f"verified_relations {transition.verified_relations}",
        f"verified_claims {transition.verified_claims}",
        "commit_gate verify(input) -> run immutable transformation -> verify(output) -> verify(lineage) -> "
        "re-evaluate(normal_form) -> verify(relation invariants) -> canonical digest -> "
        "checkpoint observers -> analysis commit",
        "analysis_invalidation all non-preserved analyses",
        "",
        "rules {",
    ]
    for transform in transforms:
        transform_rows = [item for item in rows if item["transform"] == transform]
        rule = rule_by_transform.get(transform)
        detailed.append(f"  @{transform} {{")
        if rule is not None:
            detailed.extend(
                (
                    f"    signature  {rule.source_entity} -> {rule.target_entity}",
                    f"    rewrite    {rule.rewrite}",
                    f"    invariant  @{rule.semantic_invariant or 'none'}",
                    f"    preserves [{', '.join(rule.preserves) or 'none'}]",
                    f"    introduces [{', '.join(rule.introduces) or 'none'}]",
                    f"    forbids    [{', '.join(rule.forbids) or 'none'}]",
                )
            )
        else:
            detailed.append("    contract   undeclared; the rows below are lineage evidence, not a semantic pass rule")
        detailed.append(
            f"    evidence groups={len(transform_rows)} canonical_relations="
            f"{sum(item['relation_count'] for item in transform_rows)}"
        )
        for item in transform_rows:
            detailed.append(
                f"    {item['source']} [{item['source_kind']}] -> {item['target']} "
                f"[{item['target_kind']}] targets={item['target_count']} lineage={item['lineage_kind']}"
            )
        detailed.append("  }")
    detailed.append("}")
    return LoweringExpression("\n".join(short), "\n".join(detailed))


def lowering_correspondence_rows(
    transition: DerivationTransition,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach pass-owned rule identity and short semantics to lineage evidence rows."""

    contract = transition.contract
    rule_by_transform = {item.transform: item for item in contract.rules}
    result = []
    for row in rows:
        rule = rule_by_transform.get(row["transform"])
        qualified_rule = f"{contract.name}.{row['transform']}"
        if rule is None:
            signature = "<undeclared> ⇒ <lineage evidence>"
            rewrite = "No PassRule declaration; correspondence is evidence only"
        else:
            signature = f"{rule.source_entity} ⇒ {rule.target_entity}"
            rewrite = rule.rewrite
        expression = "\n".join(
            (
                f"{qualified_rule}(relation={row['mapping_kind']}, cardinality={row['mapping']})",
                signature,
                rewrite,
            )
        )
        result.append(
            {
                **row,
                "qualified_rule": qualified_rule,
                "pass_type": "Pass",
                "pass_tone": "slate",
                "pass_expression_name": qualified_rule,
                "pass_expression_parameters": (
                    {"name": "relation", "value": row["mapping_kind"], "category": "mapping"},
                    {"name": "cardinality", "value": row["mapping"], "category": "mapping"},
                ),
                "rule_signature": signature,
                "rule_rewrite": rewrite,
                "transform_expression": expression,
                "rule_declared": rule is not None,
            }
        )
    return result


__all__ = [
    "CanonicalIRExpression",
    "LoweringExpression",
    "canonical_ir_expression",
    "lowering_correspondence_rows",
    "lowering_expression",
]

"""Presentation-neutral derivation traces and canonical IR graph projections.

The objects in this module are derived, rebuildable views.  They never become a
sixth canonical IR and never write presentation or estimate fields back into a
canonical snapshot.  Cross-stage correspondence is read exclusively from typed
entity lineage.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Protocol, cast

from typing_extensions import assert_never

from blueprinting.schema.codec import canonical_dumps
from blueprinting.schema.frozen import FrozenDict
from blueprinting.synthesizer.ids import Lineage, StableId
from blueprinting.synthesizer.passes import PassPipeline, PipelineResult, TransitionReport
from blueprinting.synthesizer.stages.common import CanonicalIRMixin
from blueprinting.synthesizer.stages.concrete_plan.ir import ConcretePlanIR
from blueprinting.synthesizer.stages.distributed.ir import (
    AllGather,
    AllReduce,
    AllToAll,
    Broadcast,
    Collective,
    CollectiveKind,
    CollectiveSpecVariant,
    DistributedTaskIR,
    ReduceScatter,
    ReductionKind,
    collective_kind,
)
from blueprinting.synthesizer.stages.machine.ir import MachineIR
from blueprinting.synthesizer.stages.model.ir import ModelIR
from blueprinting.synthesizer.stages.portable_plan.ir import PortablePlanIR


class CanonicalIRStage(Enum):
    MODEL = "model"
    DISTRIBUTED = "distributed"
    PORTABLE = "portable"
    CONCRETE = "concrete"
    MACHINE = "machine"


CANONICAL_STAGE_ORDER = (
    CanonicalIRStage.MODEL,
    CanonicalIRStage.DISTRIBUTED,
    CanonicalIRStage.PORTABLE,
    CanonicalIRStage.CONCRETE,
    CanonicalIRStage.MACHINE,
)

_STAGE_LABELS = {
    CanonicalIRStage.MODEL: "模型语义",
    CanonicalIRStage.DISTRIBUTED: "分布式任务",
    CanonicalIRStage.PORTABLE: "可移植计划",
    CanonicalIRStage.CONCRETE: "具体执行计划",
    CanonicalIRStage.MACHINE: "目标机器程序",
}


def _collective_projection(
    spec: CollectiveSpecVariant,
) -> tuple[CollectiveKind, ReductionKind | None, int | None]:
    kind = collective_kind(spec)
    match spec:
        case AllReduce(reduction=reduction) | ReduceScatter(reduction=reduction):
            return kind, reduction, None
        case Broadcast(root=root):
            return kind, None, root
        case AllGather() | AllToAll():
            return kind, None, None
    assert_never(spec)


@dataclass(frozen=True, order=True)
class EntityRef:
    stage: CanonicalIRStage
    snapshot_digest: str
    kind: str
    entity_id: str

    @property
    def key(self) -> str:
        return f"{self.snapshot_digest}:{self.kind}:{self.entity_id}"


@dataclass(frozen=True)
class IRGraphNode:
    ref: EntityRef
    label: str
    group: str
    properties: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class IRGraphEdge:
    source: EntityRef
    target: EntityRef
    kind: str
    label: str = ""
    count: int = 1


@dataclass(frozen=True)
class IRGraphView:
    stage: CanonicalIRStage
    snapshot_digest: str
    nodes: tuple[IRGraphNode, ...]
    edges: tuple[IRGraphEdge, ...]

    def node(self, entity_id: str) -> IRGraphNode | None:
        return next((item for item in self.nodes if item.ref.entity_id == entity_id), None)


@dataclass(frozen=True)
class TraceDiagnostic:
    code: str
    message: str
    level: str = "warning"
    entity_id: str = ""


@dataclass(frozen=True)
class LineageRelation:
    target: EntityRef
    sources: tuple[EntityRef, ...]
    unresolved_sources: tuple[str, ...]
    lineage_kind: str
    transform: str
    explicit_source_mismatch: bool = False
    verified_claims: tuple[str, ...] = ()


@dataclass(frozen=True)
class MappingSummary:
    source_entities: int
    mapped_source_entities: int
    target_entities: int
    mapped_target_entities: int
    generated_targets: int
    dangling_sources: int
    one_to_one: int
    one_to_many: int
    many_to_one: int
    many_to_many: int


@dataclass(frozen=True)
class IRBoundaryView:
    source_stage: CanonicalIRStage
    target_stage: CanonicalIRStage
    source_digest: str
    target_digest: str
    pass_name: str
    relations: tuple[LineageRelation, ...]
    summary: MappingSummary
    diagnostics: tuple[TraceDiagnostic, ...] = ()


@dataclass(frozen=True)
class PassRuleView:
    transform: str
    source_entity: str
    target_entity: str
    rewrite: str
    preserves: tuple[str, ...]
    introduces: tuple[str, ...]
    forbids: tuple[str, ...]
    semantic_invariant: str | None = None


@dataclass(frozen=True)
class PassContractView:
    name: str
    input_schema: str
    output_schema: str
    required_bindings: tuple[str, ...]
    required_analyses: tuple[str, ...]
    produced_analyses: tuple[str, ...]
    mutation_model: str
    verification: str
    deterministic: bool
    revision: str = "1"
    preserved_analyses: tuple[str, ...] = ()
    uses_session_seed: bool = False
    contract_digest: str = ""
    normal_form: str | None = None
    rules: tuple[PassRuleView, ...] = ()


@dataclass(frozen=True)
class DerivationStage:
    stage: CanonicalIRStage
    branch: str
    label: str
    pass_name: str
    duration_ns: int
    ir: CanonicalIRMixin
    diagnostics: tuple[TraceDiagnostic, ...] = ()

    @property
    def digest(self) -> str:
        return self.ir.digest

    @property
    def schema(self) -> str:
        return f"{self.ir.header.schema_name}@{self.ir.header.schema_version}"

    @property
    def snapshot_json(self) -> str:
        return self.ir.to_json()


@dataclass(frozen=True)
class DerivationTransition:
    source_digest: str
    target_digest: str
    contract: PassContractView
    duration_ns: int
    boundary: IRBoundaryView
    verification_status: str = "structural_only"
    verified_relations: int = 0
    verified_claims: int = 0
    canonical_conformance: str | None = None


@dataclass(frozen=True)
class EntityOverlay:
    entity_id: str
    metrics: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class DerivedOverlay:
    name: str
    stage_digest: str
    provider: str
    revision: str
    entities: tuple[EntityOverlay, ...]


@dataclass(frozen=True)
class DerivationTrace:
    request_digest: str
    session_fingerprint: str
    stages: tuple[DerivationStage, ...]
    transitions: tuple[DerivationTransition, ...]
    overlays: tuple[DerivedOverlay, ...] = ()
    schema: str = "blueprinting.derivation-trace.v0"

    def __post_init__(self) -> None:
        object.__setattr__(self, "stages", tuple(self.stages))
        object.__setattr__(self, "transitions", tuple(self.transitions))
        object.__setattr__(self, "overlays", tuple(self.overlays))

    def stage_for(self, stage: CanonicalIRStage, branch: str = "training") -> DerivationStage | None:
        return next((item for item in self.stages if item.stage is stage and item.branch == branch), None)

    @property
    def branches(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(item.branch for item in self.stages))

    def graph(self, stage: CanonicalIRStage, branch: str = "training") -> IRGraphView | None:
        item = self.stage_for(stage, branch)
        return graph_view(item.ir) if item is not None else None


@dataclass(frozen=True)
class _LineageEntity:
    ref: EntityRef
    lineage: Lineage
    explicit_sources: tuple[str, ...] = ()


class IRVisualizationAdapter(Protocol):
    ir_type: type[CanonicalIRMixin]
    stage: CanonicalIRStage

    def graph(self, ir: CanonicalIRMixin) -> IRGraphView: ...

    def lineage_entities(self, ir: CanonicalIRMixin) -> tuple[_LineageEntity, ...]: ...


def _text(value: Any) -> str:
    if isinstance(value, Enum):
        return str(value.value)
    if isinstance(value, StableId):
        return str(value)
    if isinstance(value, (FrozenDict, dict, tuple, list, frozenset)):
        try:
            return canonical_dumps(value)
        except (TypeError, ValueError):
            return repr(value)
    return str(value)


def _properties(**values: Any) -> tuple[tuple[str, str], ...]:
    return tuple((key, _text(value)) for key, value in values.items() if value is not None and value != "")


def _dimension_text(value: Any) -> str:
    name = getattr(value, "name", None)
    if isinstance(name, str) and name:
        return name
    operation = getattr(getattr(value, "op", None), "value", None)
    arguments = getattr(value, "args", ())
    if isinstance(operation, str) and arguments:
        return f"{operation}({', '.join(_dimension_text(item) for item in arguments)})"
    return str(value)


def _shape_text(value: Any) -> str:
    return "[" + ", ".join(_dimension_text(item) for item in value.shape) + "]"


def _ref(stage: CanonicalIRStage, digest: str, kind: str, identifier: Any) -> EntityRef:
    return EntityRef(stage, digest, kind, str(identifier))


class _ModelAdapter:
    ir_type = ModelIR
    stage = CanonicalIRStage.MODEL

    def graph(self, ir: CanonicalIRMixin) -> IRGraphView:
        assert isinstance(ir, ModelIR)
        digest = ir.digest
        nodes = []
        edges: list[IRGraphEdge] = []
        values = {item.id: item for item in ir.values}
        value_refs = {}
        operation_refs = {}
        for value in ir.values:
            ref = _ref(self.stage, digest, "value", value.id)
            value_refs[value.id] = ref
            nodes.append(
                IRGraphNode(
                    ref,
                    value.name or str(value.id),
                    f"value/{value.role.value}",
                    _properties(
                        role=value.role,
                        shape=_shape_text(value.type),
                        dtype=value.type.dtype,
                        layout=value.type.layout,
                        lineage=value.lineage.kind,
                    ),
                )
            )
        for operation in ir.operations:
            ref = _ref(self.stage, digest, "operation", operation.id)
            operation_refs[operation.id] = ref
            nodes.append(
                IRGraphNode(
                    ref,
                    str(operation.operation),
                    f"operation/{operation.operation.dialect}",
                    _properties(
                        operation=operation.operation,
                        shape=_shape_text(values[operation.inputs[0]].type) if operation.inputs else None,
                        dtype=values[operation.inputs[0]].type.dtype if operation.inputs else None,
                        inputs=len(operation.inputs),
                        outputs=len(operation.outputs),
                        lineage=operation.lineage.kind,
                    ),
                )
            )
            edges.extend(
                IRGraphEdge(value_refs[item], ref, "dataflow", "input")
                for item in operation.inputs
                if item in value_refs
            )
            edges.extend(
                IRGraphEdge(ref, value_refs[item], "dataflow", "output")
                for item in operation.outputs
                if item in value_refs
            )
        for operation in ir.operations:
            for dependency in operation.control_dependencies:
                if dependency in operation_refs:
                    edges.append(IRGraphEdge(operation_refs[dependency], operation_refs[operation.id], "control"))
        return IRGraphView(self.stage, digest, tuple(nodes), tuple(edges))

    def lineage_entities(self, ir: CanonicalIRMixin) -> tuple[_LineageEntity, ...]:
        assert isinstance(ir, ModelIR)
        digest = ir.digest
        return tuple(
            [_LineageEntity(_ref(self.stage, digest, "operation", item.id), item.lineage) for item in ir.operations]
            + [_LineageEntity(_ref(self.stage, digest, "value", item.id), item.lineage) for item in ir.values]
        )


class _DistributedAdapter:
    ir_type = DistributedTaskIR
    stage = CanonicalIRStage.DISTRIBUTED

    def graph(self, ir: CanonicalIRMixin) -> IRGraphView:
        assert isinstance(ir, DistributedTaskIR)
        digest = ir.digest
        nodes = []
        edges: list[IRGraphEdge] = []
        value_refs = {}
        task_refs = {}
        for value in ir.values:
            ref = _ref(self.stage, digest, "value", value.id)
            value_refs[value.id] = ref
            nodes.append(
                IRGraphNode(
                    ref,
                    str(value.id),
                    f"value/{value.role.value}",
                    _properties(
                        role=value.role,
                        shape=_shape_text(value.type),
                        dtype=value.type.dtype,
                        owners=value.owners,
                        owner_count=len(value.owners),
                        sharding="["
                        + ",".join("+".join(axes) if axes else "-" for axes in value.sharding.dimension_axes)
                        + "]",
                        dimension_axes=value.sharding.dimension_axes,
                        replicated_axes=value.sharding.replicated_axes,
                    ),
                )
            )
        for task in ir.tasks:
            invocation = getattr(task.semantic, "invocation", None)
            phase = getattr(getattr(invocation, "phase", None), "value", "")
            source_layer = getattr(invocation, "source_layer", "")
            group = f"{phase or task.body_tag}/{source_layer or 'unscoped'}"
            collective: CollectiveSpecVariant | None = None
            kind: CollectiveKind | None = None
            reduction: ReductionKind | None = None
            root: int | None = None
            match task.body:
                case Collective(spec=spec):
                    collective = spec
                    kind, reduction, root = _collective_projection(collective)
                case _:
                    pass
            ref = _ref(self.stage, digest, "task", task.id)
            task_refs[task.id] = ref
            nodes.append(
                IRGraphNode(
                    ref,
                    str(task.operation),
                    group,
                    _properties(
                        kind=task.body_tag,
                        ranks=task.ranks,
                        rank_count=len(task.ranks),
                        collective_kind=kind,
                        participants=len(collective.participants) if collective is not None else None,
                        message_bytes=collective.message_bytes if collective is not None else None,
                        reduction=reduction,
                        root=root,
                        phase=phase,
                        source_layer=source_layer,
                    ),
                )
            )
            edges.extend(
                IRGraphEdge(value_refs[item], ref, "dataflow", "input") for item in task.inputs if item in value_refs
            )
            edges.extend(
                IRGraphEdge(ref, value_refs[item], "dataflow", "output") for item in task.outputs if item in value_refs
            )
        for task in ir.tasks:
            for dependency in task.dependencies:
                if dependency in task_refs:
                    edges.append(IRGraphEdge(task_refs[dependency], task_refs[task.id], "dependency"))
        return IRGraphView(self.stage, digest, tuple(nodes), tuple(edges))

    def lineage_entities(self, ir: CanonicalIRMixin) -> tuple[_LineageEntity, ...]:
        assert isinstance(ir, DistributedTaskIR)
        digest = ir.digest
        task_items = [_LineageEntity(_ref(self.stage, digest, "task", item.id), item.lineage) for item in ir.tasks]
        value_items = [
            _LineageEntity(
                _ref(self.stage, digest, "value", item.id),
                item.lineage,
                (str(item.source_value),) if item.source_value is not None else (),
            )
            for item in ir.values
        ]
        return tuple(task_items + value_items)


class _PortableAdapter:
    ir_type = PortablePlanIR
    stage = CanonicalIRStage.PORTABLE

    def graph(self, ir: CanonicalIRMixin) -> IRGraphView:
        assert isinstance(ir, PortablePlanIR)
        digest = ir.digest
        nodes = []
        edges: list[IRGraphEdge] = []
        buffer_refs = {}
        task_refs = {}
        for buffer in ir.buffers:
            ref = _ref(self.stage, digest, "buffer", buffer.id)
            buffer_refs[buffer.id] = ref
            nodes.append(
                IRGraphNode(
                    ref,
                    str(buffer.id),
                    f"buffer/{buffer.role.value}",
                    _properties(
                        role=buffer.role,
                        storage=buffer.storage_class,
                        size_bytes=buffer.size_bytes,
                        alignment_bytes=buffer.alignment_bytes,
                    ),
                )
            )
        for task in ir.tasks:
            phase = getattr(getattr(task.semantic, "phase", None), "value", task.kind.value)
            source_layer = getattr(task.semantic, "source_layer", "unscoped")
            ref = _ref(self.stage, digest, "task", task.id)
            task_refs[task.id] = ref
            nodes.append(
                IRGraphNode(
                    ref,
                    str(task.operation),
                    f"{phase}/{source_layer}",
                    _properties(
                        kind=task.kind,
                        ranks=task.logical_ranks,
                        rank_count=len(task.logical_ranks),
                        operations=task.workload.operations,
                        read_bytes=task.workload.read_bytes,
                        write_bytes=task.workload.write_bytes,
                        message_bytes=task.workload.message_bytes,
                        concurrency=task.concurrency_group,
                        phase=phase,
                        source_layer=source_layer,
                    ),
                )
            )
            edges.extend(
                IRGraphEdge(buffer_refs[item], ref, "buffer", "read") for item in task.inputs if item in buffer_refs
            )
            edges.extend(
                IRGraphEdge(ref, buffer_refs[item], "buffer", "write") for item in task.outputs if item in buffer_refs
            )
        for task in ir.tasks:
            for dependency in task.dependencies:
                if dependency in task_refs:
                    edges.append(IRGraphEdge(task_refs[dependency], task_refs[task.id], "dependency"))
        return IRGraphView(self.stage, digest, tuple(nodes), tuple(edges))

    def lineage_entities(self, ir: CanonicalIRMixin) -> tuple[_LineageEntity, ...]:
        assert isinstance(ir, PortablePlanIR)
        digest = ir.digest
        return tuple(
            [_LineageEntity(_ref(self.stage, digest, "task", item.id), item.lineage) for item in ir.tasks]
            + [_LineageEntity(_ref(self.stage, digest, "buffer", item.id), item.lineage) for item in ir.buffers]
        )


class _ConcreteAdapter:
    ir_type = ConcretePlanIR
    stage = CanonicalIRStage.CONCRETE

    def graph(self, ir: CanonicalIRMixin) -> IRGraphView:
        assert isinstance(ir, ConcretePlanIR)
        digest = ir.digest
        nodes = []
        edges = []
        refs: dict[Any, EntityRef] = {}
        for device in ir.devices:
            ref = _ref(self.stage, digest, "device", device.id)
            refs[device.id] = ref
            nodes.append(
                IRGraphNode(
                    ref, device.target_device, f"device/{device.target_device}", _properties(rank=device.logical_rank)
                )
            )
        for queue in ir.queues:
            ref = _ref(self.stage, digest, "queue", queue.id)
            refs[queue.id] = ref
            nodes.append(
                IRGraphNode(
                    ref, queue.engine, f"device/{queue.device}", _properties(kind=queue.kind, ordered=queue.ordered)
                )
            )
            if queue.device in refs:
                edges.append(IRGraphEdge(refs[queue.device], ref, "placement", "queue"))
        for region in ir.memory_regions:
            ref = _ref(self.stage, digest, "memory_region", region.id)
            refs[region.id] = ref
            nodes.append(
                IRGraphNode(
                    ref,
                    region.memory_space,
                    f"device/{region.device}",
                    _properties(capacity_bytes=region.capacity_bytes, alignment_bytes=region.alignment_bytes),
                )
            )
            if region.device in refs:
                edges.append(IRGraphEdge(refs[region.device], ref, "placement", "memory"))
        for buffer in ir.buffers:
            ref = _ref(self.stage, digest, "buffer", buffer.id)
            refs[buffer.id] = ref
            nodes.append(
                IRGraphNode(
                    ref,
                    str(buffer.id),
                    f"memory/{buffer.memory_region}",
                    _properties(
                        offset=buffer.offset_bytes, size_bytes=buffer.size_bytes, alignment=buffer.alignment_bytes
                    ),
                )
            )
            if buffer.memory_region in refs:
                edges.append(IRGraphEdge(refs[buffer.memory_region], ref, "allocation"))
        token_refs: dict[Any, EntityRef] = {}
        for command in ir.commands:
            queue_group = str(command.queue) if command.queue is not None else "host"
            ref = _ref(self.stage, digest, "command", command.id)
            refs[command.id] = ref
            nodes.append(
                IRGraphNode(
                    ref,
                    command.implementation.key if command.implementation is not None else command.kind.value,
                    f"queue/{queue_group}",
                    _properties(kind=command.kind, queue=command.queue, implementation=command.implementation),
                )
            )
            if command.queue in refs:
                edges.append(IRGraphEdge(refs[command.queue], ref, "placement", "command"))
        for command in ir.commands:
            command_ref = refs[command.id]
            for dependency in command.dependencies:
                if dependency in refs:
                    edges.append(IRGraphEdge(refs[dependency], command_ref, "dependency"))
            for use in command.buffers:
                if use.buffer not in refs:
                    continue
                if use.access.value == "read":
                    edges.append(IRGraphEdge(refs[use.buffer], command_ref, "buffer", "read"))
                else:
                    edges.append(IRGraphEdge(command_ref, refs[use.buffer], "buffer", use.access.value))
            for token in command.signal_tokens + command.wait_tokens:
                if token not in token_refs:
                    token_ref = _ref(self.stage, digest, "sync_token", token)
                    token_refs[token] = token_ref
                    nodes.append(IRGraphNode(token_ref, str(token), "synchronization"))
            edges.extend(
                IRGraphEdge(command_ref, token_refs[item], "synchronization", "signal")
                for item in command.signal_tokens
            )
            edges.extend(
                IRGraphEdge(token_refs[item], command_ref, "synchronization", "wait") for item in command.wait_tokens
            )
        return IRGraphView(self.stage, digest, tuple(nodes), tuple(edges))

    def lineage_entities(self, ir: CanonicalIRMixin) -> tuple[_LineageEntity, ...]:
        assert isinstance(ir, ConcretePlanIR)
        digest = ir.digest
        return tuple(
            [_LineageEntity(_ref(self.stage, digest, "command", item.id), item.lineage) for item in ir.commands]
            + [
                _LineageEntity(
                    _ref(self.stage, digest, "buffer", item.id),
                    item.lineage,
                    (str(item.source_buffer),) if item.source_buffer is not None else (),
                )
                for item in ir.buffers
            ]
        )


class _MachineAdapter:
    ir_type = MachineIR
    stage = CanonicalIRStage.MACHINE

    def graph(self, ir: CanonicalIRMixin) -> IRGraphView:
        assert isinstance(ir, MachineIR)
        digest = ir.digest
        nodes = []
        edges = []
        instruction_refs = {}
        section_refs = {}
        for section in ir.sections:
            ref = _ref(self.stage, digest, "section", f"section:{section.name}")
            section_refs[section.name] = ref
            nodes.append(
                IRGraphNode(
                    ref,
                    section.name,
                    f"section/{section.kind.value}",
                    _properties(kind=section.kind, alignment=section.alignment_bytes, data_bytes=len(section.data)),
                )
            )
            for instruction in section.instructions:
                instruction_ref = _ref(self.stage, digest, "instruction", instruction.id)
                instruction_refs[instruction.id] = instruction_ref
                nodes.append(
                    IRGraphNode(
                        instruction_ref,
                        str(instruction.opcode),
                        f"section/{section.name}",
                        _properties(opcode=instruction.opcode, operands=instruction.operands),
                    )
                )
                edges.append(IRGraphEdge(ref, instruction_ref, "contains"))
        for instruction in ir.instructions:
            for dependency in instruction.dependencies:
                if dependency in instruction_refs:
                    edges.append(
                        IRGraphEdge(instruction_refs[dependency], instruction_refs[instruction.id], "dependency")
                    )
        for entry in ir.entry_points:
            ref = _ref(self.stage, digest, "entry_point", f"entry:{entry.name}")
            nodes.append(IRGraphNode(ref, entry.name, "entry_points"))
            if entry.instruction in instruction_refs:
                edges.append(IRGraphEdge(ref, instruction_refs[entry.instruction], "entry"))
        return IRGraphView(self.stage, digest, tuple(nodes), tuple(edges))

    def lineage_entities(self, ir: CanonicalIRMixin) -> tuple[_LineageEntity, ...]:
        assert isinstance(ir, MachineIR)
        digest = ir.digest
        return tuple(
            _LineageEntity(
                _ref(self.stage, digest, "instruction", item.id),
                item.lineage,
                (str(item.source_command),) if item.source_command is not None else (),
            )
            for item in ir.instructions
        )


_ADAPTERS = cast(
    tuple[IRVisualizationAdapter, ...],
    (
        _ModelAdapter(),
        _DistributedAdapter(),
        _PortableAdapter(),
        _ConcreteAdapter(),
        _MachineAdapter(),
    ),
)


def adapter_for(ir: CanonicalIRMixin) -> IRVisualizationAdapter:
    for adapter in _ADAPTERS:
        if type(ir) is adapter.ir_type:
            return adapter
    raise TypeError(f"no IR visualization adapter for {type(ir).__name__}")


def graph_view(ir: CanonicalIRMixin) -> IRGraphView:
    return adapter_for(ir).graph(ir)


def _stage_diagnostics(ir: CanonicalIRMixin) -> tuple[TraceDiagnostic, ...]:
    return tuple(
        TraceDiagnostic(item.code, item.message, item.severity.value, "/".join(item.path))
        for item in ir.diagnostics().diagnostics
    )


def _mapping_summary(
    source_graph: IRGraphView,
    target_entities: Sequence[_LineageEntity],
    relations: Sequence[LineageRelation],
) -> MappingSummary:
    target_by_source: dict[str, set[str]] = defaultdict(set)
    source_counts_by_target = {}
    generated = 0
    dangling = 0
    mapped_targets = 0
    for relation in relations:
        if not relation.sources and not relation.unresolved_sources:
            generated += 1
        if relation.sources:
            mapped_targets += 1
        dangling += len(relation.unresolved_sources)
        source_counts_by_target[relation.target.key] = len(relation.sources)
        for source in relation.sources:
            target_by_source[source.key].add(relation.target.key)
    one_to_one = one_to_many = many_to_one = many_to_many = 0
    for relation in relations:
        source_count = source_counts_by_target[relation.target.key]
        max_targets = max((len(target_by_source[item.key]) for item in relation.sources), default=0)
        if source_count == 1 and max_targets == 1:
            one_to_one += 1
        elif source_count == 1 and max_targets > 1:
            one_to_many += 1
        elif source_count > 1 and max_targets <= 1:
            many_to_one += 1
        elif source_count > 1 and max_targets > 1:
            many_to_many += 1
    return MappingSummary(
        source_entities=len(source_graph.nodes),
        mapped_source_entities=len(target_by_source),
        target_entities=len(target_entities),
        mapped_target_entities=mapped_targets,
        generated_targets=generated,
        dangling_sources=dangling,
        one_to_one=one_to_one,
        one_to_many=one_to_many,
        many_to_one=many_to_one,
        many_to_many=many_to_many,
    )


def boundary_view(
    source: CanonicalIRMixin,
    target: CanonicalIRMixin,
    pass_name: str,
    transition_report: TransitionReport | None = None,
) -> IRBoundaryView:
    source_adapter = adapter_for(source)
    target_adapter = adapter_for(target)
    source_graph = source_adapter.graph(source)
    source_by_id: dict[str, list[EntityRef]] = defaultdict(list)
    for node in source_graph.nodes:
        source_by_id[node.ref.entity_id].append(node.ref)
    target_entities = target_adapter.lineage_entities(target)
    if transition_report is not None and transition_report.relations:
        if transition_report.source_digest != source.digest or transition_report.target_digest != target.digest:
            raise ValueError("transition report digests do not match the requested boundary")
        target_by_id = {item.ref.entity_id: item for item in target_entities}
        relations = []
        for verified in transition_report.relations:
            target_entity = target_by_id.get(verified.target_id)
            if target_entity is None:
                raise ValueError(f"transition report references unknown target {verified.target_id}")
            resolved = []
            for source_id in verified.source_ids:
                candidates = source_by_id.get(source_id, ())
                if len(candidates) != 1:
                    raise ValueError(f"verified transition source {source_id} is unavailable in the source graph")
                resolved.append(candidates[0])
            relations.append(
                LineageRelation(
                    target=target_entity.ref,
                    sources=tuple(resolved),
                    unresolved_sources=(),
                    lineage_kind=verified.lineage_kind.value,
                    transform=verified.transform,
                    explicit_source_mismatch=False,
                    verified_claims=tuple(item.name for item in verified.evidence),
                )
            )
        summary = _mapping_summary(source_graph, target_entities, relations)
        return IRBoundaryView(
            source_adapter.stage,
            target_adapter.stage,
            source.digest,
            target.digest,
            pass_name,
            tuple(relations),
            summary,
            (),
        )
    relations = []
    diagnostics = []
    for item in target_entities:
        resolved = []
        unresolved = []
        lineage_sources = tuple(str(source_id) for source_id in item.lineage.sources)
        for source_id in lineage_sources:
            candidates = source_by_id.get(source_id, ())
            if len(candidates) == 1:
                resolved.append(candidates[0])
            else:
                unresolved.append(source_id)
                diagnostics.append(
                    TraceDiagnostic(
                        "lineage.source_unresolved",
                        f"{item.ref.entity_id} references unavailable source {source_id}",
                        "warning",
                        item.ref.entity_id,
                    )
                )
        mismatch = bool(item.explicit_sources and set(item.explicit_sources) != set(lineage_sources))
        if mismatch:
            diagnostics.append(
                TraceDiagnostic(
                    "lineage.explicit_source_mismatch",
                    f"{item.ref.entity_id} explicit source disagrees with Lineage.sources",
                    "warning",
                    item.ref.entity_id,
                )
            )
        if not lineage_sources and item.lineage.kind.value != "generated":
            diagnostics.append(
                TraceDiagnostic(
                    "lineage.source_missing",
                    f"{item.ref.entity_id} has no upstream source",
                    "info",
                    item.ref.entity_id,
                )
            )
        relations.append(
            LineageRelation(
                target=item.ref,
                sources=tuple(resolved),
                unresolved_sources=tuple(unresolved),
                lineage_kind=item.lineage.kind.value,
                transform=item.lineage.transform,
                explicit_source_mismatch=mismatch,
            )
        )
    diagnostics.extend(_transition_rules(source, target, relations))
    summary = _mapping_summary(source_graph, target_entities, relations)
    return IRBoundaryView(
        source_adapter.stage,
        target_adapter.stage,
        source.digest,
        target.digest,
        pass_name,
        tuple(relations),
        summary,
        tuple(diagnostics),
    )


def _transition_rules(
    source: CanonicalIRMixin,
    target: CanonicalIRMixin,
    relations: Sequence[LineageRelation],
) -> tuple[TraceDiagnostic, ...]:
    """Run narrow, schema-aware audit rules without changing pass acceptance."""

    diagnostics = []
    if isinstance(source, ModelIR) and isinstance(target, DistributedTaskIR):
        source_operations = {str(item.id) for item in source.operations}
        mapped_operations = {
            item.entity_id
            for relation in relations
            if relation.target.kind == "task"
            for item in relation.sources
            if item.kind == "operation"
        }
        for identifier in sorted(source_operations - mapped_operations):
            diagnostics.append(
                TraceDiagnostic(
                    "transformer.model_operation_unmapped",
                    f"model operation {identifier} has no distributed task lineage",
                    "warning",
                    identifier,
                )
            )
    if isinstance(source, DistributedTaskIR) and isinstance(target, PortablePlanIR):
        source_tasks = {str(item.id): item for item in source.tasks}
        target_tasks = {str(item.id): item for item in target.tasks}
        relation_by_target = {item.target.entity_id: item for item in relations}
        for target_id, task in target_tasks.items():
            relation = relation_by_target.get(target_id)
            if relation is None or len(relation.sources) != 1:
                continue
            source_task = source_tasks.get(relation.sources[0].entity_id)
            if source_task is None:
                continue
            if source_task.operation != task.operation:
                diagnostics.append(
                    TraceDiagnostic(
                        "portable.operation_changed",
                        f"portable task {target_id} changed semantic operation",
                        "warning",
                        target_id,
                    )
                )
            if source_task.ranks != task.logical_ranks:
                diagnostics.append(
                    TraceDiagnostic(
                        "portable.rank_mapping_changed",
                        f"portable task {target_id} changed logical ranks",
                        "warning",
                        target_id,
                    )
                )
            invocation = getattr(source_task.semantic, "invocation", None)
            work = getattr(invocation, "work", None)
            if work is not None:
                expected = (
                    getattr(work, "operations", None),
                    getattr(work, "read_bytes", None),
                    getattr(work, "write_bytes", None),
                    getattr(work, "message_bytes", None),
                )
                actual = (
                    task.workload.operations,
                    task.workload.read_bytes,
                    task.workload.write_bytes,
                    task.workload.message_bytes,
                )
                if expected != actual:
                    diagnostics.append(
                        TraceDiagnostic(
                            "portable.workload_not_conserved",
                            f"portable task {target_id} does not preserve typed workload facts",
                            "warning",
                            target_id,
                        )
                    )
    return tuple(diagnostics)


def _contract_view(derivation_pass: Any) -> PassContractView:
    contract = derivation_pass.contract
    return PassContractView(
        name=contract.name,
        input_schema=(
            f"{contract.input_type.SCHEMA_NAME}@{contract.input_schema.minimum}..{contract.input_schema.maximum}"
        ),
        output_schema=f"{contract.output_type.SCHEMA_NAME}@{contract.output_schema}",
        required_bindings=tuple(sorted(item.value for item in contract.required_bindings)),
        required_analyses=tuple(sorted(str(item) for item in contract.required_analyses)),
        produced_analyses=tuple(sorted(str(item) for item in contract.produced_analyses)),
        mutation_model=contract.mutation_model.value,
        verification=contract.verification.value,
        deterministic=contract.deterministic,
        revision=contract.revision,
        preserved_analyses=tuple(sorted(str(item) for item in contract.preserved_analyses)),
        uses_session_seed=contract.uses_session_seed,
        contract_digest=contract.digest,
        normal_form=contract.normalizer_identity,
        rules=tuple(
            PassRuleView(
                item.transform,
                item.source_entity,
                item.target_entity,
                item.rewrite,
                item.preservation_names,
                item.introduces,
                item.forbids,
                contract._callable_identity(item.verifier) or None,
            )
            for item in contract.rules
        ),
    )


def build_derivation_trace(
    source: CanonicalIRMixin,
    pipeline: PassPipeline,
    result: PipelineResult[Any],
    *,
    request_digest: str,
    session_fingerprint: str,
    source_duration_ns: int,
    branch: str = "training",
) -> DerivationTrace:
    if len(pipeline.passes) != len(result.checkpoints):
        raise ValueError("pipeline and checkpoint counts differ")
    source_adapter = adapter_for(source)
    stages = [
        DerivationStage(
            source_adapter.stage,
            branch,
            _STAGE_LABELS[source_adapter.stage],
            "frontend-import",
            source_duration_ns,
            source,
            _stage_diagnostics(source),
        )
    ]
    transitions = []
    previous = source
    for derivation_pass, checkpoint in zip(pipeline.passes, result.checkpoints):
        output = checkpoint.ir
        output_adapter = adapter_for(output)
        stages.append(
            DerivationStage(
                output_adapter.stage,
                branch,
                _STAGE_LABELS[output_adapter.stage],
                checkpoint.record.pass_name,
                checkpoint.record.duration_ns,
                output,
                _stage_diagnostics(output),
            )
        )
        boundary = boundary_view(
            previous,
            output,
            checkpoint.record.pass_name,
            checkpoint.record.transition_report,
        )
        transitions.append(
            DerivationTransition(
                previous.digest,
                output.digest,
                _contract_view(derivation_pass),
                checkpoint.record.duration_ns,
                boundary,
                checkpoint.record.transition_report.status.value,
                checkpoint.record.transition_report.verified_relations,
                checkpoint.record.transition_report.verified_claims,
                (
                    checkpoint.record.transition_report.canonical_conformance.verifier
                    if checkpoint.record.transition_report.canonical_conformance is not None
                    else None
                ),
            )
        )
        previous = output
    return DerivationTrace(
        request_digest=request_digest,
        session_fingerprint=session_fingerprint,
        stages=tuple(stages),
        transitions=tuple(transitions),
    )


def portable_cost_overlay(
    plan_digest: str,
    tasks: Iterable[Any],
    *,
    provider: str,
    revision: str,
) -> DerivedOverlay:
    return DerivedOverlay(
        name="portable-task-cost",
        stage_digest=plan_digest,
        provider=provider,
        revision=revision,
        entities=tuple(
            EntityOverlay(
                entity_id=item.task_id,
                metrics=(
                    ("total_seconds", float(item.total_seconds)),
                    ("compute_seconds", float(item.compute_seconds)),
                    ("memory_seconds", float(item.memory_seconds)),
                    ("network_seconds", float(item.network_seconds)),
                ),
            )
            for item in tasks
        ),
    )


class DerivationDebugBundleCodec:
    SCHEMA = "blueprinting.derivation-debug-bundle.v0"
    MAX_BYTES = 50 * 1024 * 1024

    @classmethod
    def dumps(cls, trace: DerivationTrace) -> str:
        payload = {
            "schema": cls.SCHEMA,
            "request_digest": trace.request_digest,
            "session_fingerprint": trace.session_fingerprint,
            "stages": [
                {
                    "stage": item.stage.value,
                    "branch": item.branch,
                    "label": item.label,
                    "pass_name": item.pass_name,
                    "duration_ns": item.duration_ns,
                    "snapshot_json": item.snapshot_json,
                }
                for item in trace.stages
            ],
            "transitions": [
                {
                    "source_digest": item.source_digest,
                    "target_digest": item.target_digest,
                    "duration_ns": item.duration_ns,
                    "verification_status": item.verification_status,
                    "verified_relations": item.verified_relations,
                    "verified_claims": item.verified_claims,
                    "canonical_conformance": item.canonical_conformance,
                    "contract": {
                        "name": item.contract.name,
                        "revision": item.contract.revision,
                        "input_schema": item.contract.input_schema,
                        "output_schema": item.contract.output_schema,
                        "required_bindings": item.contract.required_bindings,
                        "required_analyses": item.contract.required_analyses,
                        "produced_analyses": item.contract.produced_analyses,
                        "preserved_analyses": item.contract.preserved_analyses,
                        "mutation_model": item.contract.mutation_model,
                        "verification": item.contract.verification,
                        "deterministic": item.contract.deterministic,
                        "uses_session_seed": item.contract.uses_session_seed,
                        "contract_digest": item.contract.contract_digest,
                        "normal_form": item.contract.normal_form,
                        "rules": [
                            {
                                "transform": rule.transform,
                                "source_entity": rule.source_entity,
                                "target_entity": rule.target_entity,
                                "rewrite": rule.rewrite,
                                "preserves": rule.preserves,
                                "introduces": rule.introduces,
                                "forbids": rule.forbids,
                                "semantic_invariant": rule.semantic_invariant,
                            }
                            for rule in item.contract.rules
                        ],
                    },
                }
                for item in trace.transitions
            ],
            "overlays": [
                {
                    "name": item.name,
                    "stage_digest": item.stage_digest,
                    "provider": item.provider,
                    "revision": item.revision,
                    "entities": [
                        {"entity_id": entity.entity_id, "metrics": dict(entity.metrics)} for entity in item.entities
                    ],
                }
                for item in trace.overlays
            ],
        }
        rendered = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if len(rendered.encode("utf-8")) > cls.MAX_BYTES:
            raise ValueError("derivation debug bundle exceeds 50 MiB")
        return rendered

    @classmethod
    def loads(cls, payload: str) -> DerivationTrace:
        if len(payload.encode("utf-8")) > cls.MAX_BYTES:
            raise ValueError("derivation debug bundle exceeds 50 MiB")
        value = json.loads(payload)
        if not isinstance(value, dict) or value.get("schema") != cls.SCHEMA:
            raise ValueError("unsupported derivation debug bundle schema")
        stage_rows = value.get("stages")
        if not isinstance(stage_rows, list) or not stage_rows:
            raise ValueError("derivation debug bundle requires stages")
        stage_types: Mapping[CanonicalIRStage, type[CanonicalIRMixin]] = {
            CanonicalIRStage.MODEL: ModelIR,
            CanonicalIRStage.DISTRIBUTED: DistributedTaskIR,
            CanonicalIRStage.PORTABLE: PortablePlanIR,
            CanonicalIRStage.CONCRETE: ConcretePlanIR,
            CanonicalIRStage.MACHINE: MachineIR,
        }
        stages = []
        for row in stage_rows:
            if not isinstance(row, dict):
                raise ValueError("bundle stage must be an object")
            stage = CanonicalIRStage(str(row["stage"]))
            snapshot_json = row.get("snapshot_json")
            if not isinstance(snapshot_json, str):
                raise ValueError("bundle stage snapshot_json must be a string")
            ir = stage_types[stage].require_from_json(snapshot_json)
            if adapter_for(ir).stage is not stage:
                raise ValueError("bundle stage does not match snapshot schema")
            stages.append(
                DerivationStage(
                    stage,
                    str(row.get("branch", "training")),
                    str(row.get("label", _STAGE_LABELS[stage])),
                    str(row.get("pass_name", "imported")),
                    int(row.get("duration_ns", 0)),
                    ir,
                    _stage_diagnostics(ir),
                )
            )
        if len({(item.branch, item.stage) for item in stages}) != len(stages):
            raise ValueError("bundle repeats a canonical stage within one branch")
        expected_pairs: set[tuple[str, str]] = set()
        for branch in dict.fromkeys(item.branch for item in stages):
            branch_stages = [item for item in stages if item.branch == branch]
            indices = [CANONICAL_STAGE_ORDER.index(item.stage) for item in branch_stages]
            if indices != list(range(len(indices))):
                raise ValueError("bundle branches must start at ModelIR and contain contiguous canonical stages")
            expected_pairs.update(
                (source.digest, target.digest) for source, target in zip(branch_stages, branch_stages[1:])
            )
        by_digest = {item.digest: item for item in stages}
        transition_rows = value.get("transitions", [])
        if not isinstance(transition_rows, list):
            raise ValueError("bundle transitions must be a list")
        actual_pairs = {
            (str(row.get("source_digest")), str(row.get("target_digest")))
            for row in transition_rows
            if isinstance(row, dict)
        }
        if actual_pairs != expected_pairs or len(actual_pairs) != len(transition_rows):
            raise ValueError("bundle transitions must cover every adjacent stage exactly once")
        transitions = []
        for row in transition_rows:
            source = by_digest.get(str(row.get("source_digest")))
            target = by_digest.get(str(row.get("target_digest")))
            if source is None or target is None:
                raise ValueError("bundle transition references unavailable stage")
            source_index = CANONICAL_STAGE_ORDER.index(source.stage)
            target_index = CANONICAL_STAGE_ORDER.index(target.stage)
            if target_index != source_index + 1:
                raise ValueError("bundle transition must connect adjacent canonical stages")
            if source.digest not in target.ir.header.parent_digests:
                raise ValueError("bundle transition target does not retain source digest")
            contract_row = row.get("contract", {})
            rule_rows = contract_row.get("rules", ())
            if not isinstance(rule_rows, (list, tuple)):
                raise ValueError("bundle pass contract rules must be a list")
            contract = PassContractView(
                name=str(contract_row.get("name", target.pass_name)),
                input_schema=str(contract_row.get("input_schema", source.schema)),
                output_schema=str(contract_row.get("output_schema", target.schema)),
                required_bindings=tuple(str(item) for item in contract_row.get("required_bindings", ())),
                required_analyses=tuple(str(item) for item in contract_row.get("required_analyses", ())),
                produced_analyses=tuple(str(item) for item in contract_row.get("produced_analyses", ())),
                mutation_model=str(contract_row.get("mutation_model", "immutable")),
                verification=str(contract_row.get("verification", "both")),
                deterministic=bool(contract_row.get("deterministic", True)),
                revision=str(contract_row.get("revision", "1")),
                preserved_analyses=tuple(str(item) for item in contract_row.get("preserved_analyses", ())),
                uses_session_seed=bool(contract_row.get("uses_session_seed", False)),
                contract_digest=str(contract_row.get("contract_digest", "")),
                normal_form=(str(contract_row["normal_form"]) if contract_row.get("normal_form") is not None else None),
                rules=tuple(
                    PassRuleView(
                        transform=str(rule.get("transform", "")),
                        source_entity=str(rule.get("source_entity", "")),
                        target_entity=str(rule.get("target_entity", "")),
                        rewrite=str(rule.get("rewrite", "")),
                        preserves=tuple(str(item) for item in rule.get("preserves", ())),
                        introduces=tuple(str(item) for item in rule.get("introduces", ())),
                        forbids=tuple(str(item) for item in rule.get("forbids", ())),
                        semantic_invariant=(
                            str(rule["semantic_invariant"]) if rule.get("semantic_invariant") is not None else None
                        ),
                    )
                    for rule in rule_rows
                    if isinstance(rule, dict)
                ),
            )
            boundary = boundary_view(source.ir, target.ir, contract.name)
            transitions.append(
                DerivationTransition(
                    source.digest,
                    target.digest,
                    contract,
                    int(row.get("duration_ns", target.duration_ns)),
                    boundary,
                    str(row.get("verification_status", "structural_only")),
                    int(row.get("verified_relations", 0)),
                    int(row.get("verified_claims", 0)),
                    (str(row["canonical_conformance"]) if row.get("canonical_conformance") is not None else None),
                )
            )
        overlays = []
        overlay_rows = value.get("overlays", [])
        if not isinstance(overlay_rows, list):
            raise ValueError("bundle overlays must be a list")
        for row in overlay_rows:
            if row.get("stage_digest") not in by_digest:
                raise ValueError("bundle overlay references unavailable stage")
            overlays.append(
                DerivedOverlay(
                    name=str(row["name"]),
                    stage_digest=str(row["stage_digest"]),
                    provider=str(row["provider"]),
                    revision=str(row["revision"]),
                    entities=tuple(
                        EntityOverlay(
                            str(entity["entity_id"]),
                            tuple((str(key), float(metric)) for key, metric in entity.get("metrics", {}).items()),
                        )
                        for entity in row.get("entities", ())
                    ),
                )
            )
        trace = DerivationTrace(
            request_digest=str(value.get("request_digest", "imported")),
            session_fingerprint=str(value.get("session_fingerprint", "imported")),
            stages=tuple(stages),
            transitions=tuple(transitions),
            overlays=tuple(overlays),
        )
        cls.dumps(trace)
        return trace


def with_overlays(trace: DerivationTrace, *overlays: DerivedOverlay) -> DerivationTrace:
    return replace(trace, overlays=trace.overlays + tuple(overlays))


def stage_label(stage: CanonicalIRStage) -> str:
    return _STAGE_LABELS[stage]


__all__ = [
    "CANONICAL_STAGE_ORDER",
    "CanonicalIRStage",
    "DerivationDebugBundleCodec",
    "DerivationStage",
    "DerivationTrace",
    "DerivationTransition",
    "DerivedOverlay",
    "EntityOverlay",
    "EntityRef",
    "IRBoundaryView",
    "IRGraphEdge",
    "IRGraphNode",
    "IRGraphView",
    "IRVisualizationAdapter",
    "LineageRelation",
    "MappingSummary",
    "PassContractView",
    "PassRuleView",
    "TraceDiagnostic",
    "adapter_for",
    "boundary_view",
    "build_derivation_trace",
    "graph_view",
    "portable_cost_overlay",
    "stage_label",
    "with_overlays",
]

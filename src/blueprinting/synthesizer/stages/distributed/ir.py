"""Logical distributed program over a virtual device mesh."""

from __future__ import annotations

from dataclasses import field
from enum import Enum
from typing import ClassVar, TypeAlias

from typing_extensions import assert_never

from blueprinting.schema.authoring import VariantSpec, adt, record, require_adt_variant, seal_adt, variant
from blueprinting.schema.codec import enum_type
from blueprinting.schema.frozen import FrozenDict

from ...errors import DiagnosticBag, VerificationReport

# Scalar's forward references must remain visible to annotation deriving.
from ...expr import Scalar, ScalarExpr, Symbol  # noqa: F401
from ...ids import Lineage, NodeId, ValueId
from ...semantics import EMPTY_SEMANTIC, DistributedTaskSemantic, ProgramSemantic
from ..common import (
    CanonicalIRMixin,
    Effect,
    IRHeader,
    OperationName,
    SchemaVersion,
    TensorType,
    frozen_map,
    is_content_digest,
    make_header,
    reject_reserved_attributes,
    require_instance,
    typed_tuple,
    verify_known_references,
    verify_nonnegative_scalar,
    verify_ordered_dag,
    verify_unique_ids,
)
from ..model.ir import ValueRole


@record("blueprinting.ir.distributed-task.mesh-axis")
class MeshAxis:
    """One named dimension of the logical, target-neutral device mesh."""

    name: str
    size: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("mesh axis name must not be empty")
        if isinstance(self.size, bool) or not isinstance(self.size, int) or self.size <= 0:
            raise ValueError("mesh axis size must be a positive integer")


@record("blueprinting.ir.distributed-task.logical-mesh")
class LogicalMesh:
    """Cartesian logical-rank space used by sharding and collectives."""

    name: str
    axes: tuple[MeshAxis, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "axes", typed_tuple(self.axes, MeshAxis, "logical mesh axes"))
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("logical mesh name must not be empty")
        names = tuple(axis.name for axis in self.axes)
        if not names or len(set(names)) != len(names):
            raise ValueError("logical mesh axes must be non-empty and uniquely named")

    @property
    def size(self) -> int:
        result = 1
        for axis in self.axes:
            result *= axis.size
        return result

    @property
    def axis_names(self) -> tuple[str, ...]:
        return tuple(axis.name for axis in self.axes)


@record("blueprinting.ir.distributed-task.sharding")
class ShardingSpec:
    """Mapping from tensor dimensions to logical mesh axes."""

    dimension_axes: tuple[tuple[str, ...], ...]
    replicated_axes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "dimension_axes", tuple(tuple(item) for item in self.dimension_axes))
        object.__setattr__(self, "replicated_axes", tuple(self.replicated_axes))
        axes = tuple(axis for dimensions in self.dimension_axes for axis in dimensions) + self.replicated_axes
        if any(not isinstance(axis, str) or not axis for axis in axes):
            raise ValueError("sharding axes must be non-empty strings")

    @classmethod
    def replicated(cls, rank: int, axes: tuple[str, ...]) -> ShardingSpec:
        return cls(dimension_axes=tuple(() for _ in range(rank)), replicated_axes=axes)


@enum_type("blueprinting.ir.distributed-task.collective-kind")
class CollectiveKind(Enum):
    """Logical collective semantics independent of a communication library."""

    ALL_REDUCE = "all_reduce"
    ALL_GATHER = "all_gather"
    REDUCE_SCATTER = "reduce_scatter"
    ALL_TO_ALL = "all_to_all"
    BROADCAST = "broadcast"


@enum_type("blueprinting.ir.distributed-task.reduction-kind")
class ReductionKind(Enum):
    """Associative reduction operation required by a logical collective."""

    SUM = "sum"
    MAX = "max"
    MIN = "min"
    PRODUCT = "product"


@adt(wire="blueprinting.ir.distributed-task.collective")
class CollectiveSpec:
    """Closed collective semantics without conditional reduction/root fields."""


def _validate_collective_participants(participants: tuple[int, ...]) -> tuple[int, ...]:
    normalized = tuple(participants)
    if any(isinstance(rank, bool) or not isinstance(rank, int) or rank < 0 for rank in normalized):
        raise ValueError("collective participants must be non-negative integer ranks")
    if not normalized or len(set(normalized)) != len(normalized):
        raise ValueError("collective participants must be non-empty and unique")
    return normalized


@variant("all-reduce")
class AllReduce(CollectiveSpec):
    participants: tuple[int, ...]
    message_bytes: Scalar
    reduction: ReductionKind

    def __post_init__(self) -> None:
        object.__setattr__(self, "participants", _validate_collective_participants(self.participants))
        require_instance(self.reduction, ReductionKind, "all-reduce reduction")


@variant("reduce-scatter")
class ReduceScatter(CollectiveSpec):
    participants: tuple[int, ...]
    message_bytes: Scalar
    reduction: ReductionKind

    def __post_init__(self) -> None:
        object.__setattr__(self, "participants", _validate_collective_participants(self.participants))
        require_instance(self.reduction, ReductionKind, "reduce-scatter reduction")


@variant("all-gather")
class AllGather(CollectiveSpec):
    participants: tuple[int, ...]
    message_bytes: Scalar

    def __post_init__(self) -> None:
        object.__setattr__(self, "participants", _validate_collective_participants(self.participants))


@variant("all-to-all")
class AllToAll(CollectiveSpec):
    participants: tuple[int, ...]
    message_bytes: Scalar

    def __post_init__(self) -> None:
        object.__setattr__(self, "participants", _validate_collective_participants(self.participants))


@variant("broadcast")
class Broadcast(CollectiveSpec):
    participants: tuple[int, ...]
    message_bytes: Scalar
    root: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "participants", _validate_collective_participants(self.participants))
        if isinstance(self.root, bool) or not isinstance(self.root, int) or self.root < 0:
            raise ValueError("broadcast root must be a non-negative integer rank")
        if self.root not in self.participants:
            raise ValueError("broadcast root must be one of its participants")


CollectiveSpecVariant: TypeAlias = AllReduce | ReduceScatter | AllGather | AllToAll | Broadcast
seal_adt(CollectiveSpec, CollectiveSpecVariant)


def collective_kind(spec: CollectiveSpecVariant) -> CollectiveKind:
    match spec:
        case AllReduce():
            return CollectiveKind.ALL_REDUCE
        case ReduceScatter():
            return CollectiveKind.REDUCE_SCATTER
        case AllGather():
            return CollectiveKind.ALL_GATHER
        case AllToAll():
            return CollectiveKind.ALL_TO_ALL
        case Broadcast():
            return CollectiveKind.BROADCAST
    assert_never(spec)


def make_collective_spec(
    kind: CollectiveKind,
    participants: tuple[int, ...],
    message_bytes: Scalar,
    *,
    reduction: ReductionKind | None = None,
    root: int | None = None,
) -> CollectiveSpecVariant:
    """Boundary adapter from enum-oriented inputs into the canonical ADT."""

    match kind:
        case CollectiveKind.ALL_REDUCE:
            if reduction is None or root is not None:
                raise ValueError("all-reduce requires reduction and does not accept root")
            return AllReduce(participants, message_bytes, reduction)
        case CollectiveKind.REDUCE_SCATTER:
            if reduction is None or root is not None:
                raise ValueError("reduce-scatter requires reduction and does not accept root")
            return ReduceScatter(participants, message_bytes, reduction)
        case CollectiveKind.ALL_GATHER:
            if reduction is not None or root is not None:
                raise ValueError("all-gather does not accept reduction or root")
            return AllGather(participants, message_bytes)
        case CollectiveKind.ALL_TO_ALL:
            if reduction is not None or root is not None:
                raise ValueError("all-to-all does not accept reduction or root")
            return AllToAll(participants, message_bytes)
        case CollectiveKind.BROADCAST:
            if reduction is not None or root is None:
                raise ValueError("broadcast requires root and does not accept reduction")
            return Broadcast(participants, message_bytes, root)
    assert_never(kind)


@record("blueprinting.ir.distributed-task.peer-transfer")
class PeerTransfer:
    """Logical point-to-point transfer between two virtual ranks."""

    source_rank: int
    destination_rank: int
    message_bytes: Scalar
    channel: str = "default"

    def __post_init__(self) -> None:
        if not isinstance(self.channel, str):
            raise TypeError("peer transfer channel must be a string")
        for endpoint in (self.source_rank, self.destination_rank):
            if isinstance(endpoint, bool) or not isinstance(endpoint, int) or endpoint < 0:
                raise ValueError("peer transfer endpoints must be non-negative integer ranks")
        if self.source_rank == self.destination_rank:
            raise ValueError("peer transfer endpoints must differ")
        if not self.channel:
            raise ValueError("peer transfer channel must not be empty")


@adt(wire="blueprinting.ir.distributed-task.task")
class TaskBody:
    """Closed family of mutually exclusive distributed task semantics."""

    __variant_spec__: ClassVar[VariantSpec]


@variant("local-compute")
class LocalCompute(TaskBody):
    """Execute a target-neutral operation independently on the logical ranks."""


@variant("collective")
class Collective(TaskBody):
    """Task body carrying a well-formed logical collective specification."""

    spec: CollectiveSpecVariant


@variant("point-to-point")
class PointToPoint(TaskBody):
    """Task body carrying one logical peer transfer."""

    spec: PeerTransfer


@variant("reshard")
class Reshard(TaskBody):
    """Change logical ownership/sharding through explicit dataflow values."""


@variant("control")
class Control(TaskBody):
    """Represent a dependency-only logical coordination task."""


TaskBodyVariant: TypeAlias = LocalCompute | Collective | PointToPoint | Reshard | Control
seal_adt(TaskBody, TaskBodyVariant)


@record("blueprinting.ir.distributed-task.value")
class DistributedValue:
    """Model value specialized with logical ownership and sharding."""

    id: ValueId
    type: TensorType
    role: ValueRole
    sharding: ShardingSpec
    owners: tuple[int, ...]
    lineage: Lineage
    source_value: ValueId | None = None
    attributes: FrozenDict = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        require_instance(self.id, ValueId, "distributed value ID")
        require_instance(self.type, TensorType, "distributed value type")
        require_instance(self.role, ValueRole, "distributed value role")
        require_instance(self.sharding, ShardingSpec, "distributed value sharding")
        require_instance(self.lineage, Lineage, "distributed value lineage")
        if self.source_value is not None:
            require_instance(self.source_value, ValueId, "distributed source value")
        object.__setattr__(self, "owners", tuple(self.owners))
        object.__setattr__(self, "attributes", frozen_map(self.attributes))
        if any(isinstance(rank, bool) or not isinstance(rank, int) or rank < 0 for rank in self.owners):
            raise ValueError("distributed value owners must be non-negative integer ranks")


@record("blueprinting.ir.distributed-task.task-envelope")
class DistributedTask:
    """Common graph envelope around one typed distributed task body."""

    id: NodeId
    body: TaskBodyVariant
    operation: OperationName
    ranks: tuple[int, ...]
    inputs: tuple[ValueId, ...]
    outputs: tuple[ValueId, ...]
    dependencies: tuple[NodeId, ...]
    lineage: Lineage
    effects: tuple[Effect, ...] = ()
    semantic: DistributedTaskSemantic = EMPTY_SEMANTIC
    attributes: FrozenDict = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        require_instance(self.id, NodeId, "distributed task ID")
        require_adt_variant(self.body, TaskBody, "distributed task body")
        require_instance(self.operation, OperationName, "distributed task operation")
        require_instance(self.lineage, Lineage, "distributed task lineage")
        require_instance(self.semantic, DistributedTaskSemantic, "distributed task semantic")
        object.__setattr__(self, "ranks", tuple(self.ranks))
        object.__setattr__(self, "inputs", typed_tuple(self.inputs, ValueId, "distributed task inputs"))
        object.__setattr__(self, "outputs", typed_tuple(self.outputs, ValueId, "distributed task outputs"))
        object.__setattr__(
            self,
            "dependencies",
            typed_tuple(self.dependencies, NodeId, "distributed task dependencies"),
        )
        object.__setattr__(self, "effects", typed_tuple(self.effects, Effect, "distributed task effects"))
        object.__setattr__(self, "attributes", frozen_map(self.attributes))
        if any(isinstance(rank, bool) or not isinstance(rank, int) or rank < 0 for rank in self.ranks):
            raise ValueError("distributed task ranks must be non-negative integers")

    @property
    def body_tag(self) -> str:
        """Stable short constructor identity derived from the ADT manifest."""

        return self.body.__variant_spec__.local_tag


_DISTRIBUTED_RESERVED = frozenset(
    {
        "model_spec",
        "workload_spec",
        "mapping_spec",
        "inference_mapping_spec",
        "inference_phase",
        "batch_size",
        "query_tokens",
        "context_tokens",
        "datatype",
        "invocation",
        "block_memory",
        "scope",
        "physical_device",
        "device_id",
        "route",
        "queue",
        "stream",
        "kernel",
        "implementation_id",
        "start",
        "start_time",
        "end",
        "end_time",
        "duration",
        "latency",
        "bandwidth",
    }
)


@record("blueprinting.ir.distributed-task")
class DistributedTaskIR(CanonicalIRMixin):
    """Logical task graph whose ranks are virtual, never physical devices."""

    SCHEMA_NAME: ClassVar[str] = "blueprinting.distributed-task"
    SCHEMA_VERSION: ClassVar[SchemaVersion] = SchemaVersion(0, 0, 0)

    name: str
    source_model_digest: str
    mesh: LogicalMesh
    values: tuple[DistributedValue, ...]
    tasks: tuple[DistributedTask, ...]
    inputs: tuple[ValueId, ...]
    outputs: tuple[ValueId, ...]
    semantic: ProgramSemantic = EMPTY_SEMANTIC
    attributes: FrozenDict = field(default_factory=FrozenDict)
    header: IRHeader = field(
        default_factory=lambda: make_header(DistributedTaskIR.SCHEMA_NAME, DistributedTaskIR.SCHEMA_VERSION)
    )

    def __post_init__(self) -> None:
        require_instance(self.header, IRHeader, "distributed header")
        require_instance(self.mesh, LogicalMesh, "distributed logical mesh")
        require_instance(self.semantic, ProgramSemantic, "distributed program semantic")
        if not isinstance(self.name, str):
            raise TypeError("distributed program name must be a string")
        object.__setattr__(self, "values", typed_tuple(self.values, DistributedValue, "distributed values"))
        object.__setattr__(self, "tasks", typed_tuple(self.tasks, DistributedTask, "distributed tasks"))
        object.__setattr__(self, "inputs", typed_tuple(self.inputs, ValueId, "distributed inputs"))
        object.__setattr__(self, "outputs", typed_tuple(self.outputs, ValueId, "distributed outputs"))
        object.__setattr__(self, "attributes", frozen_map(self.attributes))
        if (
            not self.header.parent_digests
            and self.header.schema_name == self.SCHEMA_NAME
            and self.header.schema_version == self.SCHEMA_VERSION
            and is_content_digest(self.source_model_digest)
        ):
            object.__setattr__(self, "header", self.header.with_parents(self.source_model_digest))

    def diagnostics(self) -> VerificationReport:
        bag = DiagnosticBag()
        self._verify_common(bag)
        if not self.name:
            bag.error("distributed.name", "distributed program name must not be empty", "name")
        if not self.tasks or not self.values:
            bag.error("distributed.empty", "distributed program requires tasks and values", "tasks")
        if not is_content_digest(self.source_model_digest):
            bag.error(
                "distributed.source_digest", "source_model_digest must be a canonical digest", "source_model_digest"
            )
        elif self.source_model_digest not in self.header.parent_digests:
            bag.error(
                "distributed.parent_digest",
                "source model digest must be retained in header.parent_digests",
                "header",
                "parent_digests",
            )

        verify_unique_ids(bag, self.values, lambda item: item.id, "values")
        verify_unique_ids(bag, self.tasks, lambda item: item.id, "tasks")
        verify_ordered_dag(bag, self.tasks, lambda item: item.id, lambda item: item.dependencies, "tasks")
        value_ids = {item.id for item in self.values}
        verify_known_references(bag, self.inputs, value_ids, "inputs")
        verify_known_references(bag, self.outputs, value_ids, "outputs")
        if len(set(self.inputs)) != len(self.inputs):
            bag.error("distributed.duplicate_input", "distributed inputs must be unique", "inputs")
        if len(set(self.outputs)) != len(self.outputs):
            bag.error("distributed.duplicate_output", "distributed outputs must be unique", "outputs")
        mesh_axes = set(self.mesh.axis_names)
        valid_ranks = set(range(self.mesh.size))

        for index, value in enumerate(self.values):
            path = ("values", str(index))
            if len(value.sharding.dimension_axes) != value.type.rank:
                bag.error(
                    "sharding.rank",
                    "sharding dimension count must equal tensor rank",
                    *path,
                    "sharding",
                    "dimension_axes",
                )
            used_axes = tuple(axis for axes in value.sharding.dimension_axes for axis in axes)
            used_axes += value.sharding.replicated_axes
            if len(set(used_axes)) != len(used_axes):
                bag.error("sharding.axis_reuse", "a mesh axis may appear only once", *path, "sharding")
            for axis in used_axes:
                if axis not in mesh_axes:
                    bag.error("sharding.unknown_axis", f"unknown mesh axis {axis!r}", *path, "sharding")
            if not value.owners or len(set(value.owners)) != len(value.owners):
                bag.error("ownership.invalid", "owners must be non-empty and unique", *path, "owners")
            for rank in value.owners:
                if rank not in valid_ranks:
                    bag.error("rank.unknown", f"owner rank {rank} is outside the logical mesh", *path, "owners")
            reject_reserved_attributes(bag, value.attributes, _DISTRIBUTED_RESERVED, *path, "attributes")
            reject_reserved_attributes(
                bag,
                value.type.attributes,
                _DISTRIBUTED_RESERVED,
                *path,
                "type",
                "attributes",
            )

        defined = set(self.inputs)
        for index, task in enumerate(self.tasks):
            path = ("tasks", str(index))
            if not task.ranks or len(set(task.ranks)) != len(task.ranks):
                bag.error("rank.invalid", "task ranks must be non-empty and unique", *path, "ranks")
            for rank in task.ranks:
                if rank not in valid_ranks:
                    bag.error("rank.unknown", f"task rank {rank} is outside the logical mesh", *path, "ranks")
            if task.operation.dialect.lower() in {"cuda", "nccl", "rocm", "rccl", "lpu"}:
                bag.error(
                    "distributed.target_dialect",
                    f"target dialect {task.operation.dialect!r} is illegal in DistributedTaskIR",
                    *path,
                    "operation",
                )
            verify_known_references(bag, task.inputs, value_ids, *path, "inputs")
            verify_known_references(bag, task.outputs, value_ids, *path, "outputs")
            if len(set(task.inputs)) != len(task.inputs):
                bag.error("task.duplicate_input", "task inputs must be unique", *path, "inputs")
            if len(set(task.outputs)) != len(task.outputs):
                bag.error("task.duplicate_output", "task outputs must be unique", *path, "outputs")
            for input_id in task.inputs:
                if input_id in value_ids and input_id not in defined:
                    bag.error("dataflow.use_before_definition", f"value {input_id} is not yet defined", *path, "inputs")
            for output_id in task.outputs:
                if output_id in defined:
                    bag.error(
                        "dataflow.multiple_definition", f"value {output_id} has multiple definitions", *path, "outputs"
                    )
                defined.add(output_id)

            match task.body:
                case Collective(spec=collective):
                    if set(collective.participants) != set(task.ranks):
                        bag.error(
                            "collective.participants", "collective participants must equal task ranks", *path, "body"
                        )
                    verify_nonnegative_scalar(bag, collective.message_bytes, *path, "body", "message_bytes")
                    if isinstance(collective, Broadcast) and collective.root not in collective.participants:
                        bag.error("collective.root", "broadcast root must be a participant", *path, "body", "root")
                case PointToPoint(spec=transfer):
                    if {transfer.source_rank, transfer.destination_rank} != set(task.ranks):
                        bag.error("peer.endpoints", "peer transfer endpoints must equal task ranks", *path, "body")
                    verify_nonnegative_scalar(bag, transfer.message_bytes, *path, "body", "message_bytes")
                case LocalCompute() | Reshard() | Control():
                    pass
                case _:
                    bag.error(
                        "task.body.unknown",
                        f"unsupported distributed task body {type(task.body).__name__}",
                        *path,
                        "body",
                    )
            reject_reserved_attributes(bag, task.attributes, _DISTRIBUTED_RESERVED, *path, "attributes")

        for output_id in self.outputs:
            if output_id in value_ids and output_id not in defined:
                bag.error("dataflow.undefined_output", f"distributed output {output_id} is not defined", "outputs")
        reject_reserved_attributes(bag, self.attributes, _DISTRIBUTED_RESERVED, "attributes")
        return bag.report()


__all__ = [
    "AllGather",
    "AllReduce",
    "AllToAll",
    "Broadcast",
    "Collective",
    "CollectiveKind",
    "CollectiveSpec",
    "CollectiveSpecVariant",
    "Control",
    "DistributedTask",
    "DistributedTaskIR",
    "DistributedValue",
    "LocalCompute",
    "LogicalMesh",
    "MeshAxis",
    "PeerTransfer",
    "PointToPoint",
    "ReductionKind",
    "ReduceScatter",
    "Reshard",
    "ShardingSpec",
    "TaskBody",
    "TaskBodyVariant",
    "collective_kind",
    "make_collective_spec",
]

"""Logical distributed program over a virtual device mesh."""

from __future__ import annotations

from dataclasses import field
from enum import Enum
from typing import Annotated, ClassVar, TypeAlias

from typing_extensions import assert_never

from blueprinting.schema.authoring import (
    NonEmptyText,
    NonNegativeInt,
    PositiveInt,
    ValueConstraint,
    VariantSpec,
    adt,
    enum,
    record,
    seal_adt,
    variant,
)
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
    is_content_digest,
    is_known_target_dialect,
    make_header,
    reject_reserved_attributes,
    verify_known_references,
    verify_nonnegative_scalar,
    verify_ordered_dag,
    verify_unique_ids,
)
from ..model.ir import ValueRole


@record("blueprinting.ir.distributed-task.mesh-axis")
class MeshAxis:
    """One named dimension of the logical, target-neutral device mesh."""

    name: NonEmptyText
    size: PositiveInt


MeshAxes: TypeAlias = Annotated[tuple[MeshAxis, ...], ValueConstraint.NON_EMPTY]


@record("blueprinting.ir.distributed-task.logical-mesh")
class LogicalMesh:
    """Cartesian logical-rank space used by sharding and collectives."""

    name: NonEmptyText
    axes: MeshAxes

    def __post_init__(self) -> None:
        names = tuple(axis.name for axis in self.axes)
        if len(set(names)) != len(names):
            raise ValueError("logical mesh axes must be uniquely named")

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

    dimension_axes: tuple[tuple[NonEmptyText, ...], ...]
    replicated_axes: tuple[NonEmptyText, ...] = ()

    @classmethod
    def replicated(cls, rank: int, axes: tuple[str, ...]) -> ShardingSpec:
        return cls(dimension_axes=tuple(() for _ in range(rank)), replicated_axes=axes)


@enum("blueprinting.ir.distributed-task.collective-kind")
class CollectiveKind(Enum):
    """Logical collective semantics independent of a communication library."""

    ALL_REDUCE = "all_reduce"
    ALL_GATHER = "all_gather"
    REDUCE_SCATTER = "reduce_scatter"
    ALL_TO_ALL = "all_to_all"
    BROADCAST = "broadcast"


@enum("blueprinting.ir.distributed-task.reduction-kind")
class ReductionKind(Enum):
    """Associative reduction operation required by a logical collective."""

    SUM = "sum"
    MAX = "max"
    MIN = "min"
    PRODUCT = "product"


@adt(wire="blueprinting.ir.distributed-task.collective")
class CollectiveSpec:
    """Closed collective semantics without conditional reduction/root fields."""


CollectiveParticipants: TypeAlias = Annotated[
    tuple[int, ...],
    ValueConstraint.NON_EMPTY,
    ValueConstraint.UNIQUE_ITEMS,
    ValueConstraint.NON_NEGATIVE_ITEMS,
]


@variant("all-reduce")
class AllReduce(CollectiveSpec):
    participants: CollectiveParticipants
    message_bytes: Scalar
    reduction: ReductionKind


@variant("reduce-scatter")
class ReduceScatter(CollectiveSpec):
    participants: CollectiveParticipants
    message_bytes: Scalar
    reduction: ReductionKind


@variant("all-gather")
class AllGather(CollectiveSpec):
    participants: CollectiveParticipants
    message_bytes: Scalar


@variant("all-to-all")
class AllToAll(CollectiveSpec):
    participants: CollectiveParticipants
    message_bytes: Scalar


@variant("broadcast")
class Broadcast(CollectiveSpec):
    participants: CollectiveParticipants
    message_bytes: Scalar
    root: NonNegativeInt

    def __post_init__(self) -> None:
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

    source_rank: NonNegativeInt
    destination_rank: NonNegativeInt
    message_bytes: Scalar
    channel: NonEmptyText = "default"

    def __post_init__(self) -> None:
        if self.source_rank == self.destination_rank:
            raise ValueError("peer transfer endpoints must differ")


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
    owners: tuple[NonNegativeInt, ...]
    lineage: Lineage
    source_value: ValueId | None = None
    attributes: FrozenDict = field(default_factory=FrozenDict)


@record("blueprinting.ir.distributed-task.task-envelope")
class DistributedTask:
    """Common graph envelope around one typed distributed task body."""

    id: NodeId
    body: TaskBodyVariant
    operation: OperationName
    ranks: tuple[NonNegativeInt, ...]
    inputs: tuple[ValueId, ...]
    outputs: tuple[ValueId, ...]
    dependencies: tuple[NodeId, ...]
    lineage: Lineage
    effects: tuple[Effect, ...] = ()
    semantic: DistributedTaskSemantic = EMPTY_SEMANTIC
    attributes: FrozenDict = field(default_factory=FrozenDict)

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
            if is_known_target_dialect(task.operation):
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

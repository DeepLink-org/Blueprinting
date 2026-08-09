"""Logical distributed program over a virtual device mesh."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar

from ..codec import enum_type, record_type
from ..errors import DiagnosticBag, VerificationReport
from ..expr import Scalar
from ..frozen import FrozenDict
from ..ids import Lineage, NodeId, ValueId
from .common import (
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
from .model import ValueRole


@record_type("compiler.distributed.mesh_axis")
@dataclass(frozen=True)
class MeshAxis:
    name: str
    size: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("mesh axis name must not be empty")
        if isinstance(self.size, bool) or not isinstance(self.size, int) or self.size <= 0:
            raise ValueError("mesh axis size must be a positive integer")


@record_type("compiler.distributed.logical_mesh")
@dataclass(frozen=True)
class LogicalMesh:
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


@record_type("compiler.distributed.sharding")
@dataclass(frozen=True)
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


@enum_type("compiler.distributed.collective_kind")
class CollectiveKind(Enum):
    ALL_REDUCE = "all_reduce"
    ALL_GATHER = "all_gather"
    REDUCE_SCATTER = "reduce_scatter"
    ALL_TO_ALL = "all_to_all"
    BROADCAST = "broadcast"


@enum_type("compiler.distributed.reduction_kind")
class ReductionKind(Enum):
    SUM = "sum"
    MAX = "max"
    MIN = "min"
    PRODUCT = "product"


@record_type("compiler.distributed.collective")
@dataclass(frozen=True)
class CollectiveSpec:
    kind: CollectiveKind
    participants: tuple[int, ...]
    message_bytes: Scalar
    reduction: ReductionKind | None = None
    root: int | None = None

    def __post_init__(self) -> None:
        require_instance(self.kind, CollectiveKind, "collective kind")
        if self.reduction is not None:
            require_instance(self.reduction, ReductionKind, "collective reduction")
        object.__setattr__(self, "participants", tuple(self.participants))
        if any(isinstance(rank, bool) or not isinstance(rank, int) or rank < 0 for rank in self.participants):
            raise ValueError("collective participants must be non-negative integer ranks")
        if not self.participants or len(set(self.participants)) != len(self.participants):
            raise ValueError("collective participants must be non-empty and unique")
        reduction_kinds = {CollectiveKind.ALL_REDUCE, CollectiveKind.REDUCE_SCATTER}
        if self.kind in reduction_kinds and self.reduction is None:
            raise ValueError(f"{self.kind.value} requires a reduction operation")
        if self.kind not in reduction_kinds and self.reduction is not None:
            raise ValueError(f"{self.kind.value} does not accept a reduction operation")
        if self.kind is CollectiveKind.BROADCAST and self.root is None:
            raise ValueError("broadcast requires a root rank")
        if self.root is not None and (isinstance(self.root, bool) or not isinstance(self.root, int) or self.root < 0):
            raise ValueError("collective root must be a non-negative integer rank")
        if self.kind is not CollectiveKind.BROADCAST and self.root is not None:
            raise ValueError(f"{self.kind.value} does not accept a root rank")


@record_type("compiler.distributed.peer_transfer")
@dataclass(frozen=True)
class PeerTransfer:
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


@enum_type("compiler.distributed.task_kind")
class DistributedTaskKind(Enum):
    LOCAL_COMPUTE = "local_compute"
    COLLECTIVE = "collective"
    POINT_TO_POINT = "point_to_point"
    RESHARD = "reshard"
    CONTROL = "control"


@record_type("compiler.distributed.value")
@dataclass(frozen=True)
class DistributedValue:
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


@record_type("compiler.distributed.task")
@dataclass(frozen=True)
class DistributedTask:
    id: NodeId
    kind: DistributedTaskKind
    operation: OperationName
    ranks: tuple[int, ...]
    inputs: tuple[ValueId, ...]
    outputs: tuple[ValueId, ...]
    dependencies: tuple[NodeId, ...]
    lineage: Lineage
    collective: CollectiveSpec | None = None
    peer_transfer: PeerTransfer | None = None
    effects: tuple[Effect, ...] = ()
    attributes: FrozenDict = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        require_instance(self.id, NodeId, "distributed task ID")
        require_instance(self.kind, DistributedTaskKind, "distributed task kind")
        require_instance(self.operation, OperationName, "distributed task operation")
        require_instance(self.lineage, Lineage, "distributed task lineage")
        if self.collective is not None:
            require_instance(self.collective, CollectiveSpec, "distributed collective")
        if self.peer_transfer is not None:
            require_instance(self.peer_transfer, PeerTransfer, "distributed peer transfer")
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


_DISTRIBUTED_RESERVED = frozenset(
    {
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


@record_type("compiler.ir.distributed_task.v1")
@dataclass(frozen=True)
class DistributedTaskIR(CanonicalIRMixin):
    """Logical task graph whose ranks are virtual, never physical devices."""

    SCHEMA_NAME: ClassVar[str] = "blueprinting.distributed-task"
    SCHEMA_VERSION: ClassVar[SchemaVersion] = SchemaVersion(1, 0, 0)

    name: str
    source_model_digest: str
    mesh: LogicalMesh
    values: tuple[DistributedValue, ...]
    tasks: tuple[DistributedTask, ...]
    inputs: tuple[ValueId, ...]
    outputs: tuple[ValueId, ...]
    attributes: FrozenDict = field(default_factory=FrozenDict)
    header: IRHeader = field(
        default_factory=lambda: make_header(DistributedTaskIR.SCHEMA_NAME, DistributedTaskIR.SCHEMA_VERSION)
    )

    def __post_init__(self) -> None:
        require_instance(self.header, IRHeader, "distributed header")
        require_instance(self.mesh, LogicalMesh, "distributed logical mesh")
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

    def verify(self) -> VerificationReport:
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

            if task.kind is DistributedTaskKind.COLLECTIVE:
                if task.collective is None or task.peer_transfer is not None:
                    bag.error("task.collective_contract", "collective task requires only collective metadata", *path)
                elif set(task.collective.participants) != set(task.ranks):
                    bag.error(
                        "collective.participants", "collective participants must equal task ranks", *path, "collective"
                    )
            elif task.kind is DistributedTaskKind.POINT_TO_POINT:
                if task.peer_transfer is None or task.collective is not None:
                    bag.error("task.peer_contract", "point-to-point task requires only peer transfer metadata", *path)
                elif {task.peer_transfer.source_rank, task.peer_transfer.destination_rank} != set(task.ranks):
                    bag.error("peer.endpoints", "peer transfer endpoints must equal task ranks", *path, "peer_transfer")
            elif task.collective is not None or task.peer_transfer is not None:
                bag.error("task.communication_contract", "non-communication task has communication metadata", *path)

            if task.collective is not None:
                verify_nonnegative_scalar(bag, task.collective.message_bytes, *path, "collective", "message_bytes")
                if task.collective.root is not None and task.collective.root not in task.collective.participants:
                    bag.error("collective.root", "broadcast root must be a participant", *path, "collective", "root")
            if task.peer_transfer is not None:
                verify_nonnegative_scalar(
                    bag, task.peer_transfer.message_bytes, *path, "peer_transfer", "message_bytes"
                )
            reject_reserved_attributes(bag, task.attributes, _DISTRIBUTED_RESERVED, *path, "attributes")

        for output_id in self.outputs:
            if output_id in value_ids and output_id not in defined:
                bag.error("dataflow.undefined_output", f"distributed output {output_id} is not defined", "outputs")
        reject_reserved_attributes(bag, self.attributes, _DISTRIBUTED_RESERVED, "attributes")
        return bag.report()

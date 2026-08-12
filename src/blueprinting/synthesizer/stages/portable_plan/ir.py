"""Target-neutral deployment plan IR.

Portable plans describe *what resources and capabilities are required* without
choosing a physical device, queue, implementation, address, or timestamp.
"""

from __future__ import annotations

from dataclasses import field
from enum import Enum
from typing import Annotated, ClassVar, TypeAlias

from typing_extensions import assert_never

from blueprinting.schema.authoring import (
    NonEmptyText,
    NonNegativeInt,
    PositiveFiniteFloat,
    PositiveInt,
    ValueConstraint,
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
from ...ids import BufferId, Lineage, NodeId
from ...semantics import EMPTY_SEMANTIC, BufferSemantic, PlanTaskSemantic, ProgramSemantic
from ..common import (
    CanonicalIRMixin,
    Effect,
    IRHeader,
    OperationName,
    SchemaVersion,
    is_content_digest,
    is_known_target_dialect,
    make_header,
    reject_reserved_attributes,
    verify_known_references,
    verify_nonnegative_scalar,
    verify_ordered_dag,
    verify_unique_ids,
)


@enum("blueprinting.ir.portable-plan.resource-kind")
class ResourceKind(Enum):
    """Target-neutral class of a resource demand."""

    COMPUTE = "compute"
    MEMORY_CAPACITY = "memory_capacity"
    MEMORY_BANDWIDTH = "memory_bandwidth"
    NETWORK = "network"
    STORAGE = "storage"
    HOST_CONTROL = "host_control"
    SYNCHRONIZATION = "synchronization"


@enum("blueprinting.ir.portable-plan.resource-scope")
class ResourceScope(Enum):
    """Replication scope used when accounting a resource demand."""

    PER_TASK = "per_task"
    PER_RANK = "per_rank"
    SHARED = "shared"


@record("blueprinting.ir.portable-plan.resource-requirement")
class ResourceRequirement:
    """A typed, target-neutral quantity and its required capabilities."""

    kind: ResourceKind
    quantity: Scalar
    scope: ResourceScope = ResourceScope.PER_TASK
    capabilities: FrozenDict = field(default_factory=FrozenDict)


ImplementationAlternatives: TypeAlias = Annotated[
    tuple[NonEmptyText, ...],
    ValueConstraint.UNIQUE_ITEMS,
]


@record("blueprinting.ir.portable-plan.implementation-requirement")
class ImplementationRequirement:
    """Target-neutral capability request with semantic alternatives."""

    capability: NonEmptyText
    alternatives: ImplementationAlternatives = ()
    constraints: FrozenDict = field(default_factory=FrozenDict)


@record("blueprinting.ir.portable-plan.workload-facts")
class WorkloadFacts:
    """Exact or symbolic work quantities, never performance estimates."""

    operations: Scalar = 0
    read_bytes: Scalar = 0
    write_bytes: Scalar = 0
    message_bytes: Scalar = 0
    temporary_bytes: Scalar = 0
    persistent_bytes: Scalar = 0
    attributes: FrozenDict = field(default_factory=FrozenDict)


def require_concrete_quantity(value: Scalar, subject: str) -> int:
    """Return a bound non-negative integer workload quantity.

    Canonical portable plans may carry symbolic quantities before all workload
    bindings are available. Consumers that execute or cost a plan must cross
    this explicit gate instead of relying on truthiness or implicit numeric
    coercion.
    """

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{subject} must be a concrete integer, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{subject} must be non-negative")
    return value


@enum("blueprinting.ir.portable-plan.buffer-role")
class PlanBufferRole(Enum):
    """Semantic lifetime role of a portable buffer."""

    INPUT = "input"
    OUTPUT = "output"
    VALUE = "value"
    CONSTANT = "constant"
    STATE = "state"
    WORKSPACE = "workspace"
    COMMUNICATION = "communication"


@enum("blueprinting.ir.portable-plan.storage-class")
class AbstractStorageClass(Enum):
    """Storage capability required without selecting a physical memory."""

    TRANSIENT = "transient"
    PERSISTENT = "persistent"
    HOST_VISIBLE = "host_visible"
    DEVICE_LOCAL = "device_local"
    COMMUNICATION = "communication"


@record("blueprinting.ir.portable-plan.buffer")
class PlanBuffer:
    """A target-neutral buffer with exact size, lifetime links, and lineage."""

    id: BufferId
    size_bytes: Scalar
    role: PlanBufferRole
    storage_class: AbstractStorageClass
    lineage: Lineage
    producer: NodeId | None = None
    consumers: tuple[NodeId, ...] = ()
    alignment_bytes: PositiveInt = 1
    semantic: BufferSemantic = EMPTY_SEMANTIC
    attributes: FrozenDict = field(default_factory=FrozenDict)


@enum("blueprinting.ir.portable-plan.task-kind")
class PlanTaskKind(Enum):
    """Execution-domain category of a portable task."""

    COMPUTE = "compute"
    COLLECTIVE = "collective"
    TRANSFER = "transfer"
    BARRIER = "barrier"
    HOST = "host"


@adt(wire="blueprinting.ir.portable-plan.task-body")
class PlanTaskBody:
    """Closed execution-domain semantics for one portable task."""


@variant("compute")
class ComputeTask(PlanTaskBody):
    pass


@variant("collective")
class CollectiveTask(PlanTaskBody):
    pass


@variant("transfer")
class TransferTask(PlanTaskBody):
    pass


@variant("barrier")
class BarrierTask(PlanTaskBody):
    pass


@variant("host")
class HostTask(PlanTaskBody):
    pass


PlanTaskBodyVariant = ComputeTask | CollectiveTask | TransferTask | BarrierTask | HostTask
seal_adt(PlanTaskBody, PlanTaskBodyVariant)


@record("blueprinting.ir.portable-plan.task")
class PlanTask:
    """A target-neutral unit of exact work in the selected strategy DAG."""

    id: NodeId
    body: PlanTaskBodyVariant
    operation: OperationName
    dependencies: tuple[NodeId, ...]
    inputs: tuple[BufferId, ...]
    outputs: tuple[BufferId, ...]
    logical_ranks: tuple[NonNegativeInt, ...]
    workload: WorkloadFacts
    lineage: Lineage
    resources: tuple[ResourceRequirement, ...] = ()
    implementations: tuple[ImplementationRequirement, ...] = ()
    concurrency_group: NonEmptyText | None = None
    effects: tuple[Effect, ...] = ()
    semantic: PlanTaskSemantic = EMPTY_SEMANTIC
    attributes: FrozenDict = field(default_factory=FrozenDict)

    @property
    def kind(self) -> PlanTaskKind:
        """Compatibility/presentation view derived from the canonical body."""

        match self.body:
            case ComputeTask():
                return PlanTaskKind.COMPUTE
            case CollectiveTask():
                return PlanTaskKind.COLLECTIVE
            case TransferTask():
                return PlanTaskKind.TRANSFER
            case BarrierTask():
                return PlanTaskKind.BARRIER
            case HostTask():
                return PlanTaskKind.HOST
        assert_never(self.body)


@enum("blueprinting.ir.portable-plan.objective-kind")
class ObjectiveKind(Enum):
    """Quantity optimized while exploring portable plans."""

    LATENCY = "latency"
    THROUGHPUT = "throughput"
    PEAK_MEMORY = "peak_memory"
    ENERGY = "energy"


@enum("blueprinting.ir.portable-plan.objective-direction")
class ObjectiveDirection(Enum):
    """Optimization direction for a portable-plan objective."""

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


@record("blueprinting.ir.portable-plan.objective")
class PlanObjective:
    """A weighted objective retained as search intent, not measured evidence."""

    kind: ObjectiveKind
    direction: ObjectiveDirection
    weight: PositiveFiniteFloat = 1.0


_PORTABLE_RESERVED = frozenset(
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
        "name",
        "engine",
        "phase",
        "primitive",
        "source_layer",
        "collective",
        "semantic",
        "bound",
        "target",
        "target_id",
        "physical_device",
        "device_id",
        "queue",
        "queue_id",
        "stream",
        "engine_id",
        "memory_bank",
        "memory_address",
        "memory_offset",
        "implementation_id",
        "kernel_id",
        "start",
        "start_time",
        "end",
        "end_time",
        "duration",
        "latency",
        "estimated_time",
        "peak_flops",
        "peak_bandwidth",
    }
)


@record("blueprinting.ir.portable-plan")
class PortablePlanIR(CanonicalIRMixin):
    """One target-neutral plan candidate with explicit dependency and buffer DAGs."""

    SCHEMA_NAME: ClassVar[str] = "blueprinting.portable-plan"
    SCHEMA_VERSION: ClassVar[SchemaVersion] = SchemaVersion(0, 0, 0)

    name: str
    source_distributed_digest: str
    strategy_fingerprint: str
    planner_revision: str
    tasks: tuple[PlanTask, ...]
    buffers: tuple[PlanBuffer, ...]
    inputs: tuple[BufferId, ...]
    outputs: tuple[BufferId, ...]
    objectives: tuple[PlanObjective, ...]
    semantic: ProgramSemantic = EMPTY_SEMANTIC
    attributes: FrozenDict = field(default_factory=FrozenDict)
    header: IRHeader = field(
        default_factory=lambda: make_header(PortablePlanIR.SCHEMA_NAME, PortablePlanIR.SCHEMA_VERSION)
    )

    def diagnostics(self) -> VerificationReport:
        bag = DiagnosticBag()
        self._verify_common(bag)
        if not self.name:
            bag.error("portable.name", "portable plan name must not be empty", "name")
        for field_name in ("source_distributed_digest", "strategy_fingerprint", "planner_revision"):
            if not getattr(self, field_name):
                bag.error("portable.identity", f"{field_name} must not be empty", field_name)
        if not is_content_digest(self.source_distributed_digest):
            bag.error(
                "portable.source_digest",
                "source_distributed_digest must be a canonical digest",
                "source_distributed_digest",
            )
        elif self.source_distributed_digest not in self.header.parent_digests:
            bag.error("portable.parent_digest", "source digest must be retained in header", "header", "parent_digests")
        if not is_content_digest(self.strategy_fingerprint):
            bag.error(
                "portable.strategy_fingerprint",
                "strategy_fingerprint must be a canonical binding digest",
                "strategy_fingerprint",
            )
        if not self.objectives:
            bag.error("portable.objectives", "at least one planning objective is required", "objectives")
        if not self.tasks or not self.buffers:
            bag.error("portable.empty", "portable plan requires tasks and buffers", "tasks")
        if len({item.kind for item in self.objectives}) != len(self.objectives):
            bag.error("portable.duplicate_objective", "objective kinds must be unique", "objectives")

        verify_unique_ids(bag, self.tasks, lambda item: item.id, "tasks")
        verify_unique_ids(bag, self.buffers, lambda item: item.id, "buffers")
        verify_ordered_dag(bag, self.tasks, lambda item: item.id, lambda item: item.dependencies, "tasks")
        task_ids = {item.id for item in self.tasks}
        buffer_ids = {item.id for item in self.buffers}
        verify_known_references(bag, self.inputs, buffer_ids, "inputs")
        verify_known_references(bag, self.outputs, buffer_ids, "outputs")
        if len(set(self.inputs)) != len(self.inputs):
            bag.error("portable.duplicate_input", "plan inputs must be unique", "inputs")
        if len(set(self.outputs)) != len(self.outputs):
            bag.error("portable.duplicate_output", "plan outputs must be unique", "outputs")

        task_by_id = {item.id: item for item in self.tasks}
        buffer_by_id = {item.id: item for item in self.buffers}
        ancestors: dict[NodeId, set[NodeId]] = {}
        for task in self.tasks:
            inherited = set(task.dependencies)
            for dependency in task.dependencies:
                inherited.update(ancestors.get(dependency, ()))
            ancestors[task.id] = inherited

        for index, buffer in enumerate(self.buffers):
            path = ("buffers", str(index))
            verify_nonnegative_scalar(bag, buffer.size_bytes, *path, "size_bytes")
            if isinstance(buffer.size_bytes, (int, float)) and buffer.size_bytes == 0:
                bag.error("buffer.empty", "planned buffer size must be greater than zero", *path, "size_bytes")
            if buffer.producer is not None and buffer.producer not in task_ids:
                bag.error("reference.unknown", f"unknown producer {buffer.producer}", *path, "producer")
            verify_known_references(bag, buffer.consumers, task_ids, *path, "consumers")
            if len(set(buffer.consumers)) != len(buffer.consumers):
                bag.error("buffer.duplicate_consumer", "buffer consumers must be unique", *path, "consumers")
            if (
                buffer.producer is None
                and buffer.id not in self.inputs
                and buffer.role
                not in {
                    PlanBufferRole.CONSTANT,
                    PlanBufferRole.STATE,
                    PlanBufferRole.WORKSPACE,
                }
            ):
                bag.error("buffer.missing_producer", "non-input buffer requires a producer", *path, "producer")
            if (
                buffer.producer is not None
                and buffer.producer in task_by_id
                and buffer.id not in task_by_id[buffer.producer].outputs
            ):
                bag.error("buffer.producer_mismatch", "producer task does not declare this output", *path, "producer")
            for consumer in buffer.consumers:
                if consumer in task_by_id and buffer.id not in task_by_id[consumer].inputs:
                    bag.error(
                        "buffer.consumer_mismatch",
                        f"consumer {consumer} does not declare this input",
                        *path,
                        "consumers",
                    )
            reject_reserved_attributes(bag, buffer.attributes, _PORTABLE_RESERVED, *path, "attributes")

        for index, task in enumerate(self.tasks):
            path = ("tasks", str(index))
            verify_known_references(bag, task.inputs, buffer_ids, *path, "inputs")
            verify_known_references(bag, task.outputs, buffer_ids, *path, "outputs")
            if len(set(task.inputs)) != len(task.inputs):
                bag.error("task.duplicate_input", "task inputs must be unique", *path, "inputs")
            if len(set(task.outputs)) != len(task.outputs):
                bag.error("task.duplicate_output", "task outputs must be unique", *path, "outputs")
            if not task.logical_ranks or len(set(task.logical_ranks)) != len(task.logical_ranks):
                bag.error(
                    "portable.logical_ranks", "logical ranks must be non-empty and unique", *path, "logical_ranks"
                )
            for rank in task.logical_ranks:
                if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
                    bag.error(
                        "portable.logical_rank",
                        "logical ranks must be non-negative integers",
                        *path,
                        "logical_ranks",
                    )
            if is_known_target_dialect(task.operation):
                bag.error(
                    "portable.target_dialect",
                    f"target dialect {task.operation.dialect!r} is illegal in PortablePlanIR",
                    *path,
                    "operation",
                )
            for buffer_id in task.inputs:
                input_buffer = buffer_by_id.get(buffer_id)
                if input_buffer is None:
                    continue
                if task.id not in input_buffer.consumers:
                    bag.error("buffer.consumer_mismatch", f"buffer {buffer_id} omits this task", *path, "inputs")
                if input_buffer.producer is not None and input_buffer.producer not in ancestors.get(task.id, set()):
                    bag.error(
                        "task.missing_data_dependency",
                        f"producer {input_buffer.producer} of {buffer_id} is not a dependency ancestor",
                        *path,
                        "dependencies",
                    )
            for buffer_id in task.outputs:
                output_buffer = buffer_by_id.get(buffer_id)
                if output_buffer is not None and output_buffer.producer != task.id:
                    bag.error(
                        "buffer.producer_mismatch", f"buffer {buffer_id} names another producer", *path, "outputs"
                    )

            facts = task.workload
            for name in (
                "operations",
                "read_bytes",
                "write_bytes",
                "message_bytes",
                "temporary_bytes",
                "persistent_bytes",
            ):
                verify_nonnegative_scalar(bag, getattr(facts, name), *path, "workload", name)
            for resource_index, resource in enumerate(task.resources):
                resource_path = path + ("resources", str(resource_index), "quantity")
                verify_nonnegative_scalar(bag, resource.quantity, *resource_path)
                if isinstance(resource.quantity, (int, float)) and resource.quantity == 0:
                    bag.error("resource.empty", "resource quantity must be greater than zero", *resource_path)
                reject_reserved_attributes(
                    bag,
                    resource.capabilities,
                    _PORTABLE_RESERVED,
                    *path,
                    "resources",
                    str(resource_index),
                    "capabilities",
                )
            capabilities = tuple(item.capability for item in task.implementations)
            if len(set(capabilities)) != len(capabilities):
                bag.error("implementation.duplicate_requirement", "capability requirements must be unique", *path)
            for implementation_index, implementation in enumerate(task.implementations):
                reject_reserved_attributes(
                    bag,
                    implementation.constraints,
                    _PORTABLE_RESERVED,
                    *path,
                    "implementations",
                    str(implementation_index),
                    "constraints",
                )
            reject_reserved_attributes(bag, facts.attributes, _PORTABLE_RESERVED, *path, "workload", "attributes")
            reject_reserved_attributes(bag, task.attributes, _PORTABLE_RESERVED, *path, "attributes")

        reject_reserved_attributes(bag, self.attributes, _PORTABLE_RESERVED, "attributes")
        return bag.report()


__all__ = [
    "AbstractStorageClass",
    "ImplementationRequirement",
    "ObjectiveDirection",
    "ObjectiveKind",
    "PlanBuffer",
    "PlanBufferRole",
    "PlanObjective",
    "PlanTask",
    "PlanTaskBody",
    "PlanTaskBodyVariant",
    "ComputeTask",
    "CollectiveTask",
    "TransferTask",
    "BarrierTask",
    "HostTask",
    "PlanTaskKind",
    "PortablePlanIR",
    "ResourceKind",
    "ResourceRequirement",
    "ResourceScope",
    "WorkloadFacts",
]

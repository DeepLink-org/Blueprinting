"""Target- and deployment-bound authoritative command plan."""

from __future__ import annotations

from dataclasses import field
from enum import Enum
from typing import ClassVar, TypeAlias

from blueprinting.schema.authoring import VariantSpec, adt, record, require_adt_variant, seal_adt, variant
from blueprinting.schema.codec import enum_type
from blueprinting.schema.frozen import FrozenDict

from ...errors import DiagnosticBag, VerificationReport
from ...ids import (
    BufferId,
    CommandId,
    DeviceId,
    Lineage,
    MemoryRegionId,
    QueueId,
    TokenId,
)
from ..common import (
    CanonicalIRMixin,
    IRHeader,
    SchemaVersion,
    frozen_map,
    is_content_digest,
    make_header,
    reject_reserved_attributes,
    require_instance,
    typed_tuple,
    verify_known_references,
    verify_ordered_dag,
    verify_unique_ids,
)


@record("blueprinting.ir.concrete-plan.device")
class DevicePlacement:
    """Binding from a logical rank to a physical target device."""

    id: DeviceId
    logical_rank: int
    target_device: str
    attributes: FrozenDict = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        require_instance(self.id, DeviceId, "device placement ID")
        object.__setattr__(self, "attributes", frozen_map(self.attributes))
        if isinstance(self.logical_rank, bool) or not isinstance(self.logical_rank, int) or self.logical_rank < 0:
            raise ValueError("logical rank must be a non-negative integer")
        if not isinstance(self.target_device, str) or not self.target_device:
            raise ValueError("target device identity must not be empty")


@enum_type("blueprinting.ir.concrete-plan.queue-kind")
class QueueKind(Enum):
    """Target execution-engine category represented by a command queue."""

    COMPUTE = "compute"
    COLLECTIVE = "collective"
    TRANSFER = "transfer"
    HOST = "host"


@record("blueprinting.ir.concrete-plan.queue")
class QueueSpec:
    """A target-owned command queue attached to one physical device."""

    id: QueueId
    device: DeviceId
    kind: QueueKind
    engine: str
    ordered: bool = True
    attributes: FrozenDict = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        require_instance(self.id, QueueId, "queue ID")
        require_instance(self.device, DeviceId, "queue device")
        require_instance(self.kind, QueueKind, "queue kind")
        object.__setattr__(self, "attributes", frozen_map(self.attributes))
        if not isinstance(self.engine, str) or not self.engine:
            raise ValueError("queue engine must not be empty")
        if not isinstance(self.ordered, bool):
            raise TypeError("queue ordered flag must be boolean")


@record("blueprinting.ir.concrete-plan.memory-region")
class MemoryRegion:
    """A finite physical memory region available to concrete buffers."""

    id: MemoryRegionId
    device: DeviceId
    memory_space: str
    capacity_bytes: int
    alignment_bytes: int = 1
    attributes: FrozenDict = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        require_instance(self.id, MemoryRegionId, "memory region ID")
        require_instance(self.device, DeviceId, "memory region device")
        object.__setattr__(self, "attributes", frozen_map(self.attributes))
        if not isinstance(self.memory_space, str) or not self.memory_space:
            raise ValueError("memory space must not be empty")
        for name in ("capacity_bytes", "alignment_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@record("blueprinting.ir.concrete-plan.buffer-binding")
class BufferBinding:
    """Physical placement of one portable buffer with preserved lineage."""

    id: BufferId
    memory_region: MemoryRegionId
    offset_bytes: int
    size_bytes: int
    alignment_bytes: int
    lineage: Lineage
    source_buffer: BufferId | None = None
    attributes: FrozenDict = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        require_instance(self.id, BufferId, "buffer binding ID")
        require_instance(self.memory_region, MemoryRegionId, "buffer memory region")
        require_instance(self.lineage, Lineage, "buffer binding lineage")
        if self.source_buffer is not None:
            require_instance(self.source_buffer, BufferId, "source buffer")
        object.__setattr__(self, "attributes", frozen_map(self.attributes))
        for name in ("offset_bytes", "size_bytes", "alignment_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.offset_bytes < 0:
            raise ValueError("buffer offset must not be negative")
        if self.size_bytes <= 0 or self.alignment_bytes <= 0:
            raise ValueError("buffer size and alignment must be positive")


@record("blueprinting.ir.concrete-plan.implementation-ref")
class ImplementationRef:
    """Versioned target implementation and ABI selected for a command."""

    namespace: str
    name: str
    version: str
    abi: str
    variant: str = "default"

    def __post_init__(self) -> None:
        for field_name in ("namespace", "name", "version", "abi", "variant"):
            if not isinstance(getattr(self, field_name), str):
                raise TypeError("implementation identity fields must be strings")
        if any(not getattr(self, item) for item in ("namespace", "name", "version", "abi", "variant")):
            raise ValueError("implementation identity fields must not be empty")

    @property
    def key(self) -> str:
        return f"{self.namespace}:{self.name}:{self.version}:{self.variant}@{self.abi}"


@enum_type("blueprinting.ir.concrete-plan.access-mode")
class AccessMode(Enum):
    """Concrete command access performed on a bound buffer."""

    READ = "read"
    WRITE = "write"
    READ_WRITE = "read_write"


@record("blueprinting.ir.concrete-plan.buffer-use")
class BufferUse:
    """Typed buffer access declared by one concrete command."""

    buffer: BufferId
    access: AccessMode

    def __post_init__(self) -> None:
        require_instance(self.buffer, BufferId, "buffer use ID")
        require_instance(self.access, AccessMode, "buffer access mode")


@enum_type("blueprinting.ir.concrete-plan.command-kind")
class CommandKind(Enum):
    """Derived command category used by generic consumers and diagnostics."""

    LAUNCH = "launch"
    COLLECTIVE = "collective"
    TRANSFER = "transfer"
    BARRIER = "barrier"
    SIGNAL = "signal"
    WAIT = "wait"
    HOST_CALL = "host_call"


@adt(wire="blueprinting.ir.concrete-plan.command-body")
class CommandBody:
    """Closed family of mutually exclusive target command semantics."""

    __variant_spec__: ClassVar[VariantSpec]


@variant("launch")
class Launch(CommandBody):
    implementation: ImplementationRef
    queue: QueueId | None = None


@variant("collective")
class CollectiveCommand(CommandBody):
    implementation: ImplementationRef
    queue: QueueId | None = None


@variant("transfer")
class Transfer(CommandBody):
    queue: QueueId | None = None


@variant("barrier")
class Barrier(CommandBody):
    pass


@variant("signal")
class Signal(CommandBody):
    tokens: tuple[TokenId, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "tokens", typed_tuple(self.tokens, TokenId, "signal tokens"))
        if not self.tokens or len(set(self.tokens)) != len(self.tokens):
            raise ValueError("signal tokens must be non-empty and unique")


@variant("wait")
class Wait(CommandBody):
    tokens: tuple[TokenId, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "tokens", typed_tuple(self.tokens, TokenId, "wait tokens"))
        if not self.tokens or len(set(self.tokens)) != len(self.tokens):
            raise ValueError("wait tokens must be non-empty and unique")


@variant("host-call")
class HostCall(CommandBody):
    implementation: ImplementationRef
    queue: QueueId | None = None


CommandBodyVariant: TypeAlias = Launch | CollectiveCommand | Transfer | Barrier | Signal | Wait | HostCall
seal_adt(CommandBody, CommandBodyVariant)


@adt(wire="blueprinting.ir.concrete-plan.synchronization")
class CommandSynchronization:
    """Closed synchronization clause orthogonal to executable command semantics."""

    __variant_spec__: ClassVar[VariantSpec]


@variant("none")
class Unsynchronized(CommandSynchronization):
    pass


@variant("wait")
class WaitFor(CommandSynchronization):
    tokens: tuple[TokenId, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "tokens", typed_tuple(self.tokens, TokenId, "command wait tokens"))
        if not self.tokens or len(set(self.tokens)) != len(self.tokens):
            raise ValueError("command wait tokens must be non-empty and unique")


@variant("signal")
class SignalAfter(CommandSynchronization):
    tokens: tuple[TokenId, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "tokens", typed_tuple(self.tokens, TokenId, "command signal tokens"))
        if not self.tokens or len(set(self.tokens)) != len(self.tokens):
            raise ValueError("command signal tokens must be non-empty and unique")


@variant("wait-and-signal")
class WaitAndSignal(CommandSynchronization):
    wait: tuple[TokenId, ...]
    signal: tuple[TokenId, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "wait", typed_tuple(self.wait, TokenId, "command wait tokens"))
        object.__setattr__(self, "signal", typed_tuple(self.signal, TokenId, "command signal tokens"))
        if (
            not self.wait
            or not self.signal
            or len(set(self.wait)) != len(self.wait)
            or len(set(self.signal)) != len(self.signal)
        ):
            raise ValueError("wait-and-signal token sets must be non-empty and individually unique")


CommandSynchronizationVariant: TypeAlias = Unsynchronized | WaitFor | SignalAfter | WaitAndSignal
seal_adt(CommandSynchronization, CommandSynchronizationVariant)


@record("blueprinting.ir.concrete-plan.command")
class ConcreteCommand:
    """Graph envelope around one typed implementation-bound command body."""

    id: CommandId
    body: CommandBodyVariant
    dependencies: tuple[CommandId, ...]
    buffers: tuple[BufferUse, ...]
    lineage: Lineage
    synchronization: CommandSynchronizationVariant = field(default_factory=Unsynchronized)
    attributes: FrozenDict = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        require_instance(self.id, CommandId, "command ID")
        require_adt_variant(self.body, CommandBody, "command body")
        require_adt_variant(self.synchronization, CommandSynchronization, "command synchronization")
        require_instance(self.lineage, Lineage, "command lineage")
        object.__setattr__(
            self,
            "dependencies",
            typed_tuple(self.dependencies, CommandId, "command dependencies"),
        )
        object.__setattr__(self, "buffers", typed_tuple(self.buffers, BufferUse, "command buffers"))
        object.__setattr__(self, "attributes", frozen_map(self.attributes))
        if isinstance(self.body, (Signal, Wait)) and not isinstance(self.synchronization, Unsynchronized):
            raise ValueError("standalone signal/wait bodies cannot carry an additional synchronization clause")

    @property
    def kind(self) -> CommandKind:
        return {
            Launch: CommandKind.LAUNCH,
            CollectiveCommand: CommandKind.COLLECTIVE,
            Transfer: CommandKind.TRANSFER,
            Barrier: CommandKind.BARRIER,
            Signal: CommandKind.SIGNAL,
            Wait: CommandKind.WAIT,
            HostCall: CommandKind.HOST_CALL,
        }[type(self.body)]

    @property
    def queue(self) -> QueueId | None:
        if isinstance(self.body, (Launch, CollectiveCommand, Transfer, HostCall)):
            return self.body.queue
        return None

    @property
    def implementation(self) -> ImplementationRef | None:
        if isinstance(self.body, (Launch, CollectiveCommand, HostCall)):
            return self.body.implementation
        return None

    @property
    def wait_tokens(self) -> tuple[TokenId, ...]:
        if isinstance(self.body, Wait):
            return self.body.tokens
        if isinstance(self.synchronization, WaitFor):
            return self.synchronization.tokens
        if isinstance(self.synchronization, WaitAndSignal):
            return self.synchronization.wait
        return ()

    @property
    def signal_tokens(self) -> tuple[TokenId, ...]:
        if isinstance(self.body, Signal):
            return self.body.tokens
        if isinstance(self.synchronization, SignalAfter):
            return self.synchronization.tokens
        if isinstance(self.synchronization, WaitAndSignal):
            return self.synchronization.signal
        return ()


class TargetScheduleExtension:
    """Marker for registered target-owned scheduling correctness semantics."""


@record("blueprinting.ir.concrete-plan.queue-issue-order")
class QueueIssueOrder:
    """Correctness-significant issue order for one target queue."""

    queue: QueueId
    commands: tuple[CommandId, ...]

    def __post_init__(self) -> None:
        require_instance(self.queue, QueueId, "issue-order queue")
        object.__setattr__(self, "commands", typed_tuple(self.commands, CommandId, "issue-order commands"))
        if len(set(self.commands)) != len(self.commands):
            raise ValueError("one queue issue order cannot contain duplicate commands")


@record("blueprinting.ir.concrete-plan.queue-schedule-extension")
class QueueScheduleExtension(TargetScheduleExtension):
    """Typed target extension for queue-ordered execution semantics."""

    orders: tuple[QueueIssueOrder, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "orders", typed_tuple(self.orders, QueueIssueOrder, "queue issue orders"))
        if len({item.queue for item in self.orders}) != len(self.orders):
            raise ValueError("queue schedule must define each queue at most once")


@record("blueprinting.ir.concrete-plan.issue-slot")
class IssueSlot:
    """Exact cycle and slot assigned to a command by a slot target."""

    cycle: int
    slot: int
    command: CommandId

    def __post_init__(self) -> None:
        for name in ("cycle", "slot"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"issue {name} must be a non-negative integer")
        require_instance(self.command, CommandId, "issue-slot command")


@record("blueprinting.ir.concrete-plan.route-constraint")
class RouteConstraint:
    """Correctness-significant physical route required by a transfer command."""

    command: CommandId
    source_device: DeviceId
    destination_device: DeviceId
    channel: str

    def __post_init__(self) -> None:
        require_instance(self.command, CommandId, "route command")
        require_instance(self.source_device, DeviceId, "route source device")
        require_instance(self.destination_device, DeviceId, "route destination device")
        if self.source_device == self.destination_device:
            raise ValueError("route endpoints must differ")
        if not isinstance(self.channel, str) or not self.channel:
            raise ValueError("route channel must not be empty")


@record("blueprinting.ir.concrete-plan.slot-dataflow-extension")
class SlotDataflowExtension(TargetScheduleExtension):
    """Typed target extension for slot issue and routed dataflow semantics."""

    issue_slots: tuple[IssueSlot, ...]
    routes: tuple[RouteConstraint, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "issue_slots", typed_tuple(self.issue_slots, IssueSlot, "issue slots"))
        object.__setattr__(self, "routes", typed_tuple(self.routes, RouteConstraint, "route constraints"))
        positions = tuple((item.cycle, item.slot) for item in self.issue_slots)
        if len(set(positions)) != len(positions):
            raise ValueError("issue cycle/slot pairs must be unique")
        if len({item.command for item in self.issue_slots}) != len(self.issue_slots):
            raise ValueError("each command may occupy only one issue slot")
        if len({item.command for item in self.routes}) != len(self.routes):
            raise ValueError("each routed command may have only one route constraint")


_CONCRETE_RESERVED = frozenset(
    {
        "start",
        "start_time",
        "predicted_start",
        "end",
        "end_time",
        "predicted_end",
        "duration",
        "latency",
        "estimated_time",
        "predicted_duration",
    }
)


@record("blueprinting.ir.concrete-plan")
class ConcretePlanIR(CanonicalIRMixin):
    """Dependency-driven plan consumed by both simulation and emission."""

    SCHEMA_NAME: ClassVar[str] = "blueprinting.concrete-plan"
    SCHEMA_VERSION: ClassVar[SchemaVersion] = SchemaVersion(0, 0, 0)

    name: str
    source_portable_digest: str
    target_fingerprint: str
    deployment_fingerprint: str
    abi_revision: str
    evidence_revision: str
    planner_revision: str
    devices: tuple[DevicePlacement, ...]
    queues: tuple[QueueSpec, ...]
    memory_regions: tuple[MemoryRegion, ...]
    buffers: tuple[BufferBinding, ...]
    commands: tuple[ConcreteCommand, ...]
    target_extension: TargetScheduleExtension
    attributes: FrozenDict = field(default_factory=FrozenDict)
    header: IRHeader = field(
        default_factory=lambda: make_header(ConcretePlanIR.SCHEMA_NAME, ConcretePlanIR.SCHEMA_VERSION)
    )

    def __post_init__(self) -> None:
        require_instance(self.header, IRHeader, "concrete header")
        identity_fields = (
            "name",
            "source_portable_digest",
            "target_fingerprint",
            "deployment_fingerprint",
            "abi_revision",
            "evidence_revision",
            "planner_revision",
        )
        if any(not isinstance(getattr(self, field_name), str) for field_name in identity_fields):
            raise TypeError("concrete plan identity fields must be strings")
        object.__setattr__(self, "devices", typed_tuple(self.devices, DevicePlacement, "concrete devices"))
        object.__setattr__(self, "queues", typed_tuple(self.queues, QueueSpec, "concrete queues"))
        object.__setattr__(
            self,
            "memory_regions",
            typed_tuple(self.memory_regions, MemoryRegion, "concrete memory regions"),
        )
        object.__setattr__(self, "buffers", typed_tuple(self.buffers, BufferBinding, "concrete buffers"))
        object.__setattr__(self, "commands", typed_tuple(self.commands, ConcreteCommand, "concrete commands"))
        require_instance(self.target_extension, TargetScheduleExtension, "concrete target extension")
        object.__setattr__(self, "attributes", frozen_map(self.attributes))
        if (
            not self.header.parent_digests
            and self.header.schema_name == self.SCHEMA_NAME
            and self.header.schema_version == self.SCHEMA_VERSION
            and is_content_digest(self.source_portable_digest)
        ):
            object.__setattr__(self, "header", self.header.with_parents(self.source_portable_digest))

    def diagnostics(self) -> VerificationReport:
        bag = DiagnosticBag()
        self._verify_common(bag)
        if not self.name:
            bag.error("concrete.name", "concrete plan name must not be empty", "name")
        identity_fields = (
            "source_portable_digest",
            "target_fingerprint",
            "deployment_fingerprint",
            "abi_revision",
            "evidence_revision",
            "planner_revision",
        )
        for field_name in identity_fields:
            if not getattr(self, field_name):
                bag.error("concrete.identity", f"{field_name} must not be empty", field_name)
        if not is_content_digest(self.source_portable_digest):
            bag.error(
                "concrete.source_digest", "source_portable_digest must be a canonical digest", "source_portable_digest"
            )
        elif self.source_portable_digest not in self.header.parent_digests:
            bag.error("concrete.parent_digest", "source digest must be retained in header", "header", "parent_digests")
        for field_name in ("target_fingerprint", "deployment_fingerprint"):
            if not is_content_digest(getattr(self, field_name)):
                bag.error(
                    "concrete.binding_fingerprint",
                    f"{field_name} must be a canonical binding digest",
                    field_name,
                )
        if not self.devices or not self.commands:
            bag.error("concrete.empty", "concrete plan requires devices and commands", "commands")

        verify_unique_ids(bag, self.devices, lambda item: item.id, "devices")
        verify_unique_ids(bag, self.queues, lambda item: item.id, "queues")
        verify_unique_ids(bag, self.memory_regions, lambda item: item.id, "memory_regions")
        verify_unique_ids(bag, self.buffers, lambda item: item.id, "buffers")
        verify_unique_ids(bag, self.commands, lambda item: item.id, "commands")
        verify_ordered_dag(bag, self.commands, lambda item: item.id, lambda item: item.dependencies, "commands")

        device_ids = {item.id for item in self.devices}
        queue_ids = {item.id for item in self.queues}
        buffer_ids = {item.id for item in self.buffers}
        if len({item.logical_rank for item in self.devices}) != len(self.devices):
            bag.error("device.duplicate_rank", "logical ranks must map to one physical device each", "devices")
        if len({item.target_device for item in self.devices}) != len(self.devices):
            bag.error("device.duplicate_target", "target device identities must be unique", "devices")
        for index, device in enumerate(self.devices):
            reject_reserved_attributes(
                bag,
                device.attributes,
                _CONCRETE_RESERVED,
                "devices",
                str(index),
                "attributes",
            )

        queue_by_id = {item.id: item for item in self.queues}
        for index, queue in enumerate(self.queues):
            if queue.device not in device_ids:
                bag.error("reference.unknown", f"unknown queue device {queue.device}", "queues", str(index), "device")
            reject_reserved_attributes(bag, queue.attributes, _CONCRETE_RESERVED, "queues", str(index), "attributes")

        region_by_id = {item.id: item for item in self.memory_regions}
        for index, region in enumerate(self.memory_regions):
            if region.device not in device_ids:
                bag.error(
                    "reference.unknown",
                    f"unknown memory-region device {region.device}",
                    "memory_regions",
                    str(index),
                    "device",
                )
            reject_reserved_attributes(
                bag,
                region.attributes,
                _CONCRETE_RESERVED,
                "memory_regions",
                str(index),
                "attributes",
            )

        buffer_by_id = {item.id: item for item in self.buffers}
        for index, buffer in enumerate(self.buffers):
            path = ("buffers", str(index))
            buffer_region = region_by_id.get(buffer.memory_region)
            if buffer_region is None:
                bag.error("reference.unknown", f"unknown memory region {buffer.memory_region}", *path, "memory_region")
            else:
                if buffer.offset_bytes % max(buffer.alignment_bytes, buffer_region.alignment_bytes) != 0:
                    bag.error(
                        "buffer.alignment", "buffer offset violates buffer or region alignment", *path, "offset_bytes"
                    )
                if buffer.offset_bytes + buffer.size_bytes > buffer_region.capacity_bytes:
                    bag.error("buffer.out_of_bounds", "buffer allocation exceeds memory region capacity", *path)
            reject_reserved_attributes(bag, buffer.attributes, _CONCRETE_RESERVED, *path, "attributes")

        signaled = {}
        implementation_kinds = {CommandKind.LAUNCH, CommandKind.COLLECTIVE, CommandKind.HOST_CALL}
        queue_kinds = {CommandKind.LAUNCH, CommandKind.COLLECTIVE, CommandKind.TRANSFER, CommandKind.HOST_CALL}
        expected_queue_kind = {
            CommandKind.LAUNCH: QueueKind.COMPUTE,
            CommandKind.COLLECTIVE: QueueKind.COLLECTIVE,
            CommandKind.TRANSFER: QueueKind.TRANSFER,
            CommandKind.HOST_CALL: QueueKind.HOST,
        }
        for index, command in enumerate(self.commands):
            path = ("commands", str(index))
            if (
                isinstance(self.target_extension, QueueScheduleExtension)
                and command.kind in queue_kinds
                and command.queue is None
            ):
                bag.error("command.missing_queue", f"{command.kind.value} command requires a queue", *path, "queue")
            if command.queue is not None and command.queue not in queue_ids:
                bag.error("reference.unknown", f"unknown queue {command.queue}", *path, "queue")
            elif command.queue is not None and command.kind in expected_queue_kind:
                actual_kind = queue_by_id[command.queue].kind
                if actual_kind is not expected_queue_kind[command.kind]:
                    bag.error(
                        "command.queue_kind",
                        f"{command.kind.value} requires a {expected_queue_kind[command.kind].value} queue",
                        *path,
                        "queue",
                    )
            if command.kind in implementation_kinds and command.implementation is None:
                bag.error(
                    "command.missing_implementation",
                    f"{command.kind.value} command requires a selected implementation",
                    *path,
                    "implementation",
                )
            if command.kind not in implementation_kinds and command.implementation is not None:
                bag.error(
                    "command.unexpected_implementation",
                    f"{command.kind.value} command cannot select an implementation",
                    *path,
                    "implementation",
                )
            used_buffers = tuple(item.buffer for item in command.buffers)
            verify_known_references(bag, used_buffers, buffer_ids, *path, "buffers")
            if len(set(used_buffers)) != len(used_buffers):
                bag.error("command.duplicate_buffer", "one command may list each buffer only once", *path, "buffers")
            if command.kind is CommandKind.LAUNCH and command.queue in queue_by_id:
                queue_device = queue_by_id[command.queue].device
                for buffer_id in used_buffers:
                    binding = buffer_by_id.get(buffer_id)
                    binding_region = region_by_id.get(binding.memory_region) if binding is not None else None
                    if binding_region is not None and binding_region.device != queue_device:
                        bag.error(
                            "command.device_mismatch",
                            f"launch buffer {buffer_id} is not resident on queue device",
                            *path,
                            "buffers",
                        )

            if len(set(command.wait_tokens)) != len(command.wait_tokens):
                bag.error("token.duplicate_wait", "wait tokens must be unique", *path, "wait_tokens")
            if len(set(command.signal_tokens)) != len(command.signal_tokens):
                bag.error("token.duplicate_signal", "signal tokens must be unique", *path, "signal_tokens")
            for token in command.wait_tokens:
                if token not in signaled:
                    bag.error("token.wait_before_signal", f"token {token} has not been signaled", *path, "wait_tokens")
            for token in command.signal_tokens:
                if token in signaled:
                    bag.error("token.multiple_signal", f"token {token} has multiple signalers", *path, "signal_tokens")
                signaled[token] = command.id
            if command.kind is CommandKind.SIGNAL and not command.signal_tokens:
                bag.error(
                    "token.empty_signal", "signal command must produce at least one token", *path, "signal_tokens"
                )
            if command.kind is CommandKind.WAIT and not command.wait_tokens:
                bag.error("token.empty_wait", "wait command must consume at least one token", *path, "wait_tokens")
            reject_reserved_attributes(bag, command.attributes, _CONCRETE_RESERVED, *path, "attributes")

        reject_reserved_attributes(bag, self.attributes, _CONCRETE_RESERVED, "attributes")
        _verify_target_extension(self, bag)
        return bag.report()


def _verify_target_extension(plan: ConcretePlanIR, bag: DiagnosticBag) -> None:
    command_by_id = {item.id: item for item in plan.commands}
    if isinstance(plan.target_extension, QueueScheduleExtension):
        queue_ids = {item.id for item in plan.queues}
        ordered_commands = tuple(command for order in plan.target_extension.orders for command in order.commands)
        expected = tuple(item.id for item in plan.commands if item.queue is not None)
        if len(ordered_commands) != len(set(ordered_commands)):
            bag.error("target.queue.duplicate_command", "queued commands must appear exactly once", "target_extension")
        if set(ordered_commands) != set(expected):
            bag.error("target.queue.coverage", "queue orders must cover every queued command", "target_extension")
        for order_index, order in enumerate(plan.target_extension.orders):
            if order.queue not in queue_ids:
                bag.error(
                    "reference.unknown", f"unknown extension queue {order.queue}", "target_extension", str(order_index)
                )
                continue
            position = {command: index for index, command in enumerate(order.commands)}
            for command_id in order.commands:
                command = command_by_id.get(command_id)
                if command is None:
                    bag.error("reference.unknown", f"unknown extension command {command_id}", "target_extension")
                    continue
                if command.queue != order.queue:
                    bag.error("target.queue.mismatch", "command is listed under a different queue", "target_extension")
                for dependency in command.dependencies:
                    if dependency in position and position[dependency] >= position[command_id]:
                        bag.error("target.queue.order", "queue order violates a command dependency", "target_extension")
        return

    if isinstance(plan.target_extension, SlotDataflowExtension):
        if plan.queues or any(item.queue is not None for item in plan.commands):
            bag.error("target.slot.queue", "slot/dataflow plans cannot carry queue semantics", "target_extension")
        slot_by_command = {item.command: item for item in plan.target_extension.issue_slots}
        if set(slot_by_command) != set(command_by_id):
            bag.error("target.slot.coverage", "issue slots must cover every command exactly once", "target_extension")
        for command in plan.commands:
            current = slot_by_command.get(command.id)
            if current is None:
                continue
            for dependency in command.dependencies:
                previous = slot_by_command.get(dependency)
                if previous is not None and (previous.cycle, previous.slot) >= (current.cycle, current.slot):
                    bag.error("target.slot.order", "issue slots violate a command dependency", "target_extension")
        device_ids = {item.id for item in plan.devices}
        routes = {item.command: item for item in plan.target_extension.routes}
        transfers = {item.id for item in plan.commands if item.kind is CommandKind.TRANSFER}
        if set(routes) != transfers:
            bag.error("target.route.coverage", "every transfer command requires exactly one route", "target_extension")
        for route in plan.target_extension.routes:
            if route.source_device not in device_ids or route.destination_device not in device_ids:
                bag.error("reference.unknown", "route endpoint is not a concrete device", "target_extension")
        return

    bag.error("target.extension.unknown", "unsupported typed target schedule extension", "target_extension")


__all__ = [
    "AccessMode",
    "Barrier",
    "BufferBinding",
    "BufferUse",
    "CommandKind",
    "CommandBody",
    "CommandBodyVariant",
    "CommandSynchronization",
    "CommandSynchronizationVariant",
    "CollectiveCommand",
    "ConcreteCommand",
    "ConcretePlanIR",
    "DevicePlacement",
    "ImplementationRef",
    "HostCall",
    "IssueSlot",
    "MemoryRegion",
    "Launch",
    "QueueIssueOrder",
    "QueueKind",
    "QueueScheduleExtension",
    "QueueSpec",
    "RouteConstraint",
    "SlotDataflowExtension",
    "Signal",
    "SignalAfter",
    "TargetScheduleExtension",
    "Transfer",
    "Unsynchronized",
    "Wait",
    "WaitAndSignal",
    "WaitFor",
]

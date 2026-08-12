"""Deterministic reference binders for exercising ConcretePlanIR contracts."""

from __future__ import annotations

from blueprinting.schema.codec import content_digest

from ...axes import BindingAxis
from ...ids import CommandId, DeviceId, Lineage, MemoryRegionId, QueueId
from ...passes.authoring import DerivationPass, PassContext, PassRule, RelationCheckContext, derivation, relation
from ...session import SynthesisSession
from ..common import make_header
from ..portable_plan.ir import PlanBuffer, PlanTask, PlanTaskKind, PortablePlanIR
from .ir import (
    AccessMode,
    Barrier,
    BufferBinding,
    BufferUse,
    CollectiveCommand,
    CommandBodyVariant,
    CommandKind,
    ConcreteCommand,
    ConcretePlanIR,
    DevicePlacement,
    HostCall,
    ImplementationRef,
    IssueSlot,
    Launch,
    MemoryRegion,
    QueueIssueOrder,
    QueueKind,
    QueueScheduleExtension,
    QueueSpec,
    RouteConstraint,
    SlotDataflowExtension,
    TargetScheduleExtension,
    Transfer,
)

__all__ = ["BindReferenceQueueTargetPass", "BindReferenceSlotTargetPass"]

_COMMAND_KIND = {
    PlanTaskKind.COMPUTE: CommandKind.LAUNCH,
    PlanTaskKind.COLLECTIVE: CommandKind.COLLECTIVE,
    PlanTaskKind.TRANSFER: CommandKind.TRANSFER,
    PlanTaskKind.BARRIER: CommandKind.BARRIER,
    PlanTaskKind.HOST: CommandKind.HOST_CALL,
}

_QUEUE_KIND = {
    PlanTaskKind.COMPUTE: QueueKind.COMPUTE,
    PlanTaskKind.COLLECTIVE: QueueKind.COLLECTIVE,
    PlanTaskKind.TRANSFER: QueueKind.TRANSFER,
    PlanTaskKind.HOST: QueueKind.HOST,
}


def _buffer_size(buffer: PlanBuffer) -> int:
    if isinstance(buffer.size_bytes, bool) or not isinstance(buffer.size_bytes, int) or buffer.size_bytes <= 0:
        raise TypeError("reference binders require concrete positive buffer sizes")
    return buffer.size_bytes


def _command_body(
    kind: PlanTaskKind,
    implementation: ImplementationRef | None,
    queue: QueueId | None,
) -> CommandBodyVariant:
    match kind:
        case PlanTaskKind.COMPUTE:
            if implementation is None:
                raise ValueError("compute task requires a selected implementation")
            return Launch(implementation, queue)
        case PlanTaskKind.COLLECTIVE:
            if implementation is None:
                raise ValueError("collective task requires a selected implementation")
            return CollectiveCommand(implementation, queue)
        case PlanTaskKind.TRANSFER:
            return Transfer(queue)
        case PlanTaskKind.BARRIER:
            return Barrier()
        case PlanTaskKind.HOST:
            if implementation is None:
                raise ValueError("host task requires a selected implementation")
            return HostCall(implementation, queue)


def _verify_reference_buffer(
    source: PlanBuffer,
    target: BufferBinding,
    _context: RelationCheckContext,
) -> None:
    expected_size = _buffer_size(source)
    if target.id != source.id or target.source_buffer != source.id:
        raise ValueError("concrete buffer must retain its portable buffer identity")
    if target.size_bytes != expected_size:
        raise ValueError("concrete buffer size differs from its portable capacity obligation")
    if target.alignment_bytes < source.alignment_bytes or target.offset_bytes % target.alignment_bytes:
        raise ValueError("concrete buffer does not satisfy portable alignment")


def _verify_reference_command(
    source: PlanTask,
    target: ConcreteCommand,
    context: RelationCheckContext,
) -> None:
    expected_kind = _COMMAND_KIND[source.kind]
    if target.kind is not expected_kind:
        raise ValueError("concrete command kind differs from its portable task body")
    expected_dependencies = tuple(context.only_target_for(item, "ConcreteCommand") for item in source.dependencies)
    if target.dependencies != expected_dependencies:
        raise ValueError("concrete command dependencies do not preserve the portable task DAG")
    expected_buffers = tuple(BufferUse(item, AccessMode.READ) for item in source.inputs) + tuple(
        BufferUse(item, AccessMode.WRITE) for item in source.outputs
    )
    if target.buffers != expected_buffers:
        raise ValueError("concrete command buffer accesses differ from its portable task")
    if expected_kind in {CommandKind.LAUNCH, CommandKind.COLLECTIVE, CommandKind.HOST_CALL}:
        implementation = target.implementation
        target_profile = context.session.bindings.target
        if implementation is None or target_profile is None:
            raise ValueError("implementation-bound command is missing target identity")
        if implementation.name != str(source.operation) or implementation.abi != target_profile.target_abi:
            raise ValueError("selected implementation does not match portable operation and target ABI")
    elif target.implementation is not None:
        raise ValueError("non-implementation command unexpectedly selected an implementation")


def _rules(prefix: str) -> tuple[PassRule, PassRule]:
    return (
        relation(
            f"{prefix}-buffer",
            "Bind an abstract portable buffer to a reference memory region",
            source=PlanBuffer,
            target=BufferBinding,
            verifier=_verify_reference_buffer,
            introduces=("memory region", "offset"),
        ),
        relation(
            f"{prefix}-command",
            "Select a reference implementation and materialize command dependencies",
            source=PlanTask,
            target=ConcreteCommand,
            verifier=_verify_reference_command,
            introduces=("implementation", "target schedule identity"),
        ),
    )


_QUEUE_RULES = _rules("reference-queue")
_SLOT_RULES = _rules("reference-slot")


def _common(
    ir: PortablePlanIR,
    session: SynthesisSession,
    *,
    planner: str,
    buffer_transform: str,
    command_transform: str,
    queued: bool,
) -> ConcretePlanIR:
    target = session.bindings.target
    deployment = session.bindings.deployment
    if target is None or deployment is None:
        raise ValueError("reference target binding requires target and deployment profiles")
    devices = tuple(
        DevicePlacement(
            DeviceId.derive(ir.digest, planner, "device", rank),
            rank,
            f"{target.name}:{rank}",
        )
        for rank in range(deployment.device_count)
    )
    region_id = MemoryRegionId.derive(ir.digest, planner, "memory", 0)
    offsets = []
    offset = 0
    for buffer in ir.buffers:
        alignment = max(16, buffer.alignment_bytes)
        offset = ((offset + alignment - 1) // alignment) * alignment
        offsets.append(offset)
        offset += _buffer_size(buffer)
    declared_capacity = deployment.available_memory_bytes[0] if deployment.available_memory_bytes else offset
    if declared_capacity < offset:
        raise ValueError("reference deployment does not have enough memory for portable buffers")
    region = MemoryRegion(region_id, devices[0].id, "reference-local", max(declared_capacity, 1), 16)
    buffers = tuple(
        BufferBinding(
            buffer.id,
            region_id,
            buffer_offset,
            _buffer_size(buffer),
            max(16, buffer.alignment_bytes),
            Lineage.lowered(buffer_transform, (buffer.id,)),
            source_buffer=buffer.id,
        )
        for buffer, buffer_offset in zip(ir.buffers, offsets)
    )
    command_ids = {
        task.id: CommandId.derive(ir.digest, planner, "command", index, task.id) for index, task in enumerate(ir.tasks)
    }
    queue_by_kind = {}
    queues = []
    extension: TargetScheduleExtension
    if queued:
        for kind in tuple(dict.fromkeys(task.kind for task in ir.tasks if task.kind in _QUEUE_KIND)):
            queue_id = QueueId.derive(ir.digest, planner, "queue", kind.value)
            queue_by_kind[kind] = queue_id
            queues.append(QueueSpec(queue_id, devices[0].id, _QUEUE_KIND[kind], f"reference:{kind.value}"))
    commands = []
    for task in ir.tasks:
        command_kind = _COMMAND_KIND[task.kind]
        implementation = None
        if command_kind in {CommandKind.LAUNCH, CommandKind.COLLECTIVE, CommandKind.HOST_CALL}:
            implementation = ImplementationRef(
                "reference",
                str(task.operation),
                "0",
                target.target_abi,
            )
        uses = tuple(BufferUse(item, AccessMode.READ) for item in task.inputs) + tuple(
            BufferUse(item, AccessMode.WRITE) for item in task.outputs
        )
        commands.append(
            ConcreteCommand(
                command_ids[task.id],
                _command_body(task.kind, implementation, queue_by_kind.get(task.kind) if queued else None),
                tuple(command_ids[item] for item in task.dependencies),
                uses,
                Lineage.lowered(command_transform, (task.id,)),
            )
        )
    if queued:
        orders = tuple(
            QueueIssueOrder(
                queue.id,
                tuple(command.id for command in commands if command.queue == queue.id),
            )
            for queue in queues
        )
        extension = QueueScheduleExtension(orders)
    else:
        issue_slots = tuple(IssueSlot(index, 0, command.id) for index, command in enumerate(commands))
        transfer_commands = tuple(item for item in commands if item.kind is CommandKind.TRANSFER)
        if transfer_commands and len(devices) < 2:
            raise ValueError("slot/dataflow transfer routes require at least two deployment devices")
        routes = tuple(
            RouteConstraint(command.id, devices[0].id, devices[1].id, "reference-link") for command in transfer_commands
        )
        extension = SlotDataflowExtension(issue_slots, routes)
    return ConcretePlanIR(
        name=f"{ir.name}-{planner}",
        source_portable_digest=ir.digest,
        target_fingerprint=target.fingerprint,
        deployment_fingerprint=deployment.fingerprint,
        abi_revision=target.target_abi,
        evidence_revision=session.evidence_snapshot,
        planner_revision=planner,
        devices=devices,
        queues=tuple(queues),
        memory_regions=(region,),
        buffers=buffers,
        commands=tuple(commands),
        target_extension=extension,
        header=make_header(
            ConcretePlanIR.SCHEMA_NAME,
            ConcretePlanIR.SCHEMA_VERSION,
            parent_digests=(ir.digest,),
        ),
    )


def _queue_normal_form(ir: PortablePlanIR, session: SynthesisSession) -> ConcretePlanIR:
    return _common(
        ir,
        session,
        planner="reference-queue",
        buffer_transform="reference-queue-buffer",
        command_transform="reference-queue-command",
        queued=True,
    )


def _slot_normal_form(ir: PortablePlanIR, session: SynthesisSession) -> ConcretePlanIR:
    return _common(
        ir,
        session,
        planner="reference-slot",
        buffer_transform="reference-slot-buffer",
        command_transform="reference-slot-command",
        queued=False,
    )


@derivation(
    "reference-queue-bind",
    revision="1",
    bindings=(BindingAxis.TARGET, BindingAxis.DEPLOYMENT),
    rules=_QUEUE_RULES,
    normalizer=_queue_normal_form,
)
class BindReferenceQueueTargetPass(DerivationPass[PortablePlanIR, ConcretePlanIR]):
    r"""Bind a portable plan to the deterministic queue-reference contract.

    This is a contract reference implementation, not a paper-derived scheduler.
    Buffers are laid out in canonical order.  For buffer ``i`` with size ``s_i``
    and required alignment ``r_i``, define ``a_i=max(16,r_i)`` and

    $$
    o_0=0,\qquad
    o_i=\left\lceil\frac{o_{i-1}+s_{i-1}}{a_i}\right\rceil a_i.
    $$

    Hence ``o_i mod a_i = 0`` and ``o_i ≥ o_{i-1}+s_{i-1}``.  Tasks retain
    topological order and are partitioned into compute, collective, transfer,
    and host queues; issue order is their stable subsequence in each queue.
    The commit gate re-evaluates this canonical construction and requires exact
    equality, so size, alignment, access mode, implementation, task category,
    dependencies, and target extension are checked as one derivation law.

    References:
        - Internal deterministic reference-target contract; no paper-derived
          performance or scheduling algorithm is claimed.
    """

    def run(self, ir: PortablePlanIR, context: PassContext) -> ConcretePlanIR:
        return _queue_normal_form(ir, context.session)


@derivation(
    "reference-slot-bind",
    revision="1",
    bindings=(BindingAxis.TARGET, BindingAxis.DEPLOYMENT),
    rules=_SLOT_RULES,
    normalizer=_slot_normal_form,
)
class BindReferenceSlotTargetPass(DerivationPass[PortablePlanIR, ConcretePlanIR]):
    r"""Bind a portable plan to the deterministic slot/dataflow reference contract.

    This pass shares the aligned buffer recurrence of
    ``BindReferenceQueueTargetPass``.  Given the portable topological task order
    ``T=(t_0,...,t_{n-1})``, its reference issue assignment is

    $$
    \operatorname{issue}(t_i)=(\operatorname{cycle}=i,
                                \operatorname{slot}=0).
    $$

    Therefore every dependency ``t_j → t_i`` already proved by PortablePlanIR
    satisfies ``j<i`` and consequently ``cycle(t_j)<cycle(t_i)``.  Transfers
    receive explicit source/destination route constraints.  This construction
    exists to exercise typed target extensions and their verifier; it is not a
    throughput-optimal slot scheduler and has no claimed research-paper result.

    References:
        - Internal deterministic slot/dataflow contract; no paper-derived
          performance or scheduling algorithm is claimed.
    """

    def run(self, ir: PortablePlanIR, context: PassContext) -> ConcretePlanIR:
        return _slot_normal_form(ir, context.session)


REFERENCE_QUEUE_TARGET_REVISION = content_digest(
    (BindReferenceQueueTargetPass.contract.name, BindReferenceQueueTargetPass.contract.revision),
    "reference-target",
)
REFERENCE_SLOT_TARGET_REVISION = content_digest(
    (BindReferenceSlotTargetPass.contract.name, BindReferenceSlotTargetPass.contract.revision),
    "reference-target",
)

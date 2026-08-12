"""Evidence-aware cost view for a static inference phase plan."""

from __future__ import annotations

from dataclasses import dataclass

from blueprinting.mapping import NetworkTierBinding, TransformerInferenceMappingSpec
from blueprinting.schema.frozen import FrozenDict
from blueprinting.synthesizer.dialects.transformer import (
    EngineKind,
    InferenceInvocation,
    PhaseWork,
    TransformerBufferSemantic,
    TransformerInferencePlanSemantic,
    TransformerInferencePlanTaskSemantic,
)
from blueprinting.workload import TransformerDataType, TransformerModelSpec

from ..synthesizer.bindings import InferencePhase
from ..synthesizer.stages.portable_plan.ir import (
    PlanBuffer,
    PlanTask,
    PortablePlanIR,
    require_concrete_quantity,
)
from ..system import SystemProfile
from .cost import CostQuery, CostQueryContext, CostResolver, CostSubject, EstimateUncertainty
from .cost_model import CalibrationMode
from .inference_evidence import InferenceEvidenceQuery, inference_cost_operation


@dataclass(frozen=True)
class InferenceTaskEstimate:
    invocation: InferenceInvocation
    compute_seconds: float
    memory_seconds: float
    network_seconds: float
    analytical_seconds: float
    total_seconds: float
    evidence_provider: str
    evidence_revision: str
    evidence_source_revision: str
    evidence_record_ids: tuple[str, ...]
    evidence_match: str
    evidence_method: str
    evidence_uncertainty: EstimateUncertainty
    evidence_assumptions: tuple[str, ...]


@dataclass(frozen=True)
class InferencePhaseMemory:
    weights: int
    kv_cache: int
    working_upper_bound: int
    pipeline_buffers: int

    @property
    def total(self) -> int:
        return self.weights + self.kv_cache + self.working_upper_bound + self.pipeline_buffers


@dataclass(frozen=True)
class InferencePhaseEstimate:
    mode: CalibrationMode
    phase: InferencePhase
    batch_size: int
    query_tokens: int
    context_tokens: int
    tasks: tuple[InferenceTaskEstimate, ...]
    memory: InferencePhaseMemory
    block_seconds: float
    transformer_seconds: float
    pipeline_seconds: float
    total_seconds: float
    evidence_revisions: FrozenDict


def inference_evidence_query_for(
    invocation: InferenceInvocation,
    *,
    hardware: SystemProfile,
    mapping: TransformerInferenceMappingSpec,
    datatype: TransformerDataType,
    model: TransformerModelSpec,
    batch_size: int,
    query_tokens: int,
    context_tokens: int,
) -> InferenceEvidenceQuery:
    """Build the one normalized query shared by cost and baseline paths."""

    return InferenceEvidenceQuery(
        phase=invocation.phase,
        primitive=invocation.primitive,
        source_layer=invocation.source_layer,
        model_name=model.name,
        hardware_name=hardware.name,
        model_sequence_length=model.sequence_length,
        hidden_size=model.hidden_size,
        feedforward_size=model.feedforward_size,
        attention_heads=model.attention_heads,
        batch_size=batch_size,
        query_tokens=query_tokens,
        context_tokens=context_tokens,
        tensor_parallel=mapping.tensor_parallel,
        datatype=datatype,
    )


def _merge_dimensions(base: dict[str, object], extra: FrozenDict) -> FrozenDict:
    overlap = set(base).intersection(extra)
    conflicts = tuple(key for key in overlap if base[key] != extra[key])
    if conflicts:
        raise ValueError(f"cost context conflicts with canonical workload dimensions: {sorted(conflicts)}")
    base.update(extra)
    return FrozenDict(base)


def cost_query_for_inference_task(
    task: PlanTask,
    *,
    hardware: SystemProfile,
    mapping: TransformerInferenceMappingSpec,
    network_binding: NetworkTierBinding,
    datatype: TransformerDataType,
    model: TransformerModelSpec,
    batch_size: int,
    query_tokens: int,
    context_tokens: int,
    context: CostQueryContext = CostQueryContext(),
) -> CostQuery:
    """Build a normalized cost request directly from portable workload facts."""

    if not isinstance(context, CostQueryContext):
        raise TypeError("context must be CostQueryContext")
    invocation = _invocation_from_plan_task(task)
    primitive = invocation.primitive
    operation = inference_cost_operation(primitive)
    subject = CostSubject.COMMUNICATION if invocation.engine is EngineKind.COLLECTIVE else CostSubject.OPERATOR
    tensor_parallel = mapping.tensor_parallel
    dimensions: dict[str, object] = {
        "semantic_operation": primitive,
        "source_layer": invocation.source_layer,
        "phase": invocation.phase.value,
        "model_name": model.name,
        "model_sequence_length": model.sequence_length,
        "hidden_size": model.hidden_size,
        "feedforward_size": model.feedforward_size,
        "attention_heads": model.attention_heads,
        "kv_heads": model.attention_heads,
        "local_attention_heads": model.attention_heads // tensor_parallel,
        "local_kv_heads": model.attention_heads // tensor_parallel,
        "head_size": model.attention_head_size,
        "batch_size": batch_size,
        "query_tokens": query_tokens,
        "context_tokens": context_tokens,
        "tensor_parallel": tensor_parallel,
        "num_tokens": batch_size * query_tokens,
        "use_gated_mlp": False,
        "beam_width": 1,
        "window_size": 0,
        "kv_cache_datatype": datatype,
    }
    if operation == "gemm":
        tokens = batch_size * query_tokens
        local_hidden = model.hidden_size // tensor_parallel
        local_feedforward = model.feedforward_size // tensor_parallel
        m = tokens
        if primitive == "attention_pre_projection":
            n, k = 3 * local_hidden, model.hidden_size
        elif primitive == "attention_post_projection":
            n, k = model.hidden_size, local_hidden
        elif primitive == "mlp_up_projection":
            n, k = local_feedforward, model.hidden_size
        else:
            n, k = model.hidden_size, local_feedforward
        dimensions.update({"m": m, "n": n, "k": k})
    query_dimensions = _merge_dimensions(dimensions, context.dimensions_for(primitive, operation))
    return CostQuery(
        subject=subject,
        operation=operation,
        hardware=hardware.name,
        datatype=datatype,
        operations=require_concrete_quantity(task.workload.operations, f"task {task.id} operations"),
        read_bytes=require_concrete_quantity(task.workload.read_bytes, f"task {task.id} read_bytes"),
        write_bytes=require_concrete_quantity(task.workload.write_bytes, f"task {task.id} write_bytes"),
        message_bytes=require_concrete_quantity(task.workload.message_bytes, f"task {task.id} message_bytes"),
        participants=tensor_parallel if subject is CostSubject.COMMUNICATION else 1,
        network_tier=network_binding.tensor_parallel,
        engine=invocation.engine.value,
        hardware_revision=hardware.evidence_revision,
        implementation=context.implementation_for(primitive, operation),
        implementation_revision=context.implementation_revision_for(primitive, operation),
        runtime=context.runtime,
        runtime_revision=context.runtime_revision,
        topology=context.topology,
        power_mode=context.power_mode,
        dimensions=query_dimensions,
    )


def _task_estimate(
    task: PlanTask,
    *,
    hardware: SystemProfile,
    mapping: TransformerInferenceMappingSpec,
    network_binding: NetworkTierBinding,
    datatype: TransformerDataType,
    model: TransformerModelSpec,
    batch_size: int,
    query_tokens: int,
    context_tokens: int,
    mode: CalibrationMode,
    cost_resolver: CostResolver | None,
    cost_context: CostQueryContext,
) -> InferenceTaskEstimate:
    invocation = _invocation_from_plan_task(task)
    work = invocation.work
    processor = hardware.matrix if invocation.engine is EngineKind.MATRIX else hardware.vector
    apply_efficiency = mode is CalibrationMode.SYSTEM_EVIDENCE
    compute_seconds = (
        work.operations / processor.throughput(work.operations, apply_efficiency=apply_efficiency)
        if work.operations
        else 0.0
    )
    memory_seconds = (
        work.memory_bytes / hardware.memory.throughput(work.memory_bytes, apply_efficiency=apply_efficiency)
        if work.memory_bytes
        else 0.0
    )
    network_seconds = 0.0
    if invocation.engine is EngineKind.COLLECTIVE:
        if invocation.collective is None:
            raise ValueError("collective invocation is missing its collective kind")
        try:
            network = hardware.networks[network_binding.tensor_parallel]
        except IndexError as error:
            raise ValueError("tensor_parallel network tier is not defined by the bound system") from error
        network_seconds = network.time(
            invocation.collective.value,
            work.message_bytes,
            mapping.tensor_parallel,
            apply_efficiency=apply_efficiency,
        )
    analytical_seconds = hardware.processing_time(compute_seconds, memory_seconds) + network_seconds
    provider_name = "analytical-system-profile"
    revision = hardware.evidence_revision
    source_revision = hardware.evidence_revision
    record_ids: tuple[str, ...] = ()
    match = mode.value
    method = "analytical"
    uncertainty = EstimateUncertainty()
    assumptions: tuple[str, ...] = ()
    total_seconds = analytical_seconds
    if cost_resolver is not None:
        resolution = cost_resolver.require(
            cost_query_for_inference_task(
                task,
                model=model,
                mapping=mapping,
                network_binding=network_binding,
                datatype=datatype,
                hardware=hardware,
                batch_size=batch_size,
                query_tokens=query_tokens,
                context_tokens=context_tokens,
                context=cost_context,
            )
        )
        evidence = resolution.estimate
        total_seconds = evidence.seconds
        provider_name = evidence.provider
        revision = evidence.provider_revision
        source_revision = evidence.source_revision
        record_ids = evidence.raw_record_ids
        match = evidence.match.value
        method = evidence.method.value
        uncertainty = evidence.uncertainty
        assumptions = evidence.assumptions
    return InferenceTaskEstimate(
        invocation=invocation,
        compute_seconds=compute_seconds,
        memory_seconds=memory_seconds,
        network_seconds=network_seconds,
        analytical_seconds=analytical_seconds,
        total_seconds=total_seconds,
        evidence_provider=provider_name,
        evidence_revision=revision,
        evidence_source_revision=source_revision,
        evidence_record_ids=record_ids,
        evidence_match=match,
        evidence_method=method,
        evidence_uncertainty=uncertainty,
        evidence_assumptions=assumptions,
    )


def _invocation_from_plan_task(task: PlanTask) -> InferenceInvocation:
    """Reconstruct comparison metadata from canonical portable workload facts.

    Costing reads operation and byte counts only from ``PlanTask.workload``.
    The earlier lowering-only ``InferenceInvocation`` is deliberately not
    embedded in ``PortablePlanIR`` as a second source of workload truth.
    """

    semantic = task.semantic
    if not isinstance(semantic, TransformerInferencePlanTaskSemantic):
        raise ValueError(f"portable inference task {task.id} is missing typed Transformer semantics")

    return InferenceInvocation(
        name=semantic.name,
        source_layer=semantic.source_layer,
        primitive=semantic.primitive,
        phase=semantic.phase,
        engine=semantic.engine,
        work=PhaseWork(
            operations=require_concrete_quantity(task.workload.operations, f"task {task.id} operations"),
            read_bytes=require_concrete_quantity(task.workload.read_bytes, f"task {task.id} read_bytes"),
            write_bytes=require_concrete_quantity(task.workload.write_bytes, f"task {task.id} write_bytes"),
            message_bytes=require_concrete_quantity(task.workload.message_bytes, f"task {task.id} message_bytes"),
        ),
        collective=semantic.collective,
    )


def _concrete_buffer_size(buffer: PlanBuffer) -> int:
    size = buffer.size_bytes
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise TypeError(f"portable inference buffer {buffer.id} must have a concrete non-negative size")
    return size


def _semantic_buffer_size(plan: PortablePlanIR, semantic: str) -> int:
    buffers = tuple(
        buffer
        for buffer in plan.buffers
        if isinstance(buffer.semantic, TransformerBufferSemantic) and buffer.semantic.role == semantic
    )
    if len(buffers) != 1:
        raise ValueError(f"portable inference plan must contain exactly one {semantic!r} buffer")
    return _concrete_buffer_size(buffers[0])


def estimate_inference_phase(
    plan: PortablePlanIR,
    hardware: SystemProfile,
    mode: CalibrationMode = CalibrationMode.SYSTEM_EVIDENCE,
    *,
    network_binding: NetworkTierBinding,
    cost_resolver: CostResolver | None = None,
    cost_context: CostQueryContext = CostQueryContext(),
) -> InferencePhaseEstimate:
    """Cost one prefill or decode phase point without queueing assumptions."""

    semantic = plan.semantic
    if not isinstance(semantic, TransformerInferencePlanSemantic):
        raise TypeError("portable inference plan is missing typed Transformer inference semantics")
    model = semantic.model
    mapping = semantic.mapping
    phase = semantic.phase
    datatype = semantic.datatype
    if hardware.datatype != datatype:
        raise ValueError("system profile datatype does not match inference workload datatype")
    if not isinstance(network_binding, NetworkTierBinding):
        raise TypeError("network_binding must be NetworkTierBinding")
    if not isinstance(cost_context, CostQueryContext):
        raise TypeError("cost_context must be CostQueryContext")
    batch_size = semantic.batch_size
    query_tokens = semantic.query_tokens
    context_tokens = semantic.context_tokens

    task_estimates = []
    for task in plan.tasks:
        task_estimates.append(
            _task_estimate(
                task,
                hardware=hardware,
                mapping=mapping,
                network_binding=network_binding,
                datatype=datatype,
                model=model,
                batch_size=batch_size,
                query_tokens=query_tokens,
                context_tokens=context_tokens,
                mode=mode,
                cost_resolver=cost_resolver,
                cost_context=cost_context,
            )
        )

    block_seconds = sum(item.total_seconds for item in task_estimates)
    transformer_seconds = block_seconds * model.block_count
    pipeline_seconds = 0.0
    pipeline_estimate = None
    if mapping.pipeline_parallel > 1:
        boundary_bytes = _concrete_buffer_size(next(buffer for buffer in plan.buffers if buffer.id == plan.inputs[0]))
        if cost_resolver is None:
            try:
                network = hardware.networks[network_binding.pipeline_parallel]
            except IndexError as error:
                raise ValueError("pipeline_parallel network tier is not defined by the bound system") from error
            one_hop_seconds = network.time(
                "p2p",
                boundary_bytes,
                2,
                apply_efficiency=mode is CalibrationMode.SYSTEM_EVIDENCE,
            )
        else:
            pipeline_dimensions: dict[str, object] = {
                "semantic_operation": "p2p",
                "phase": phase.value,
                "model_name": model.name,
                "batch_size": batch_size,
                "query_tokens": query_tokens,
                "context_tokens": context_tokens,
                "pipeline_parallel": mapping.pipeline_parallel,
            }
            resolved_pipeline_dimensions = _merge_dimensions(
                pipeline_dimensions,
                cost_context.dimensions_for("p2p", "p2p"),
            )
            pipeline_estimate = cost_resolver.require(
                CostQuery(
                    subject=CostSubject.COMMUNICATION,
                    operation="p2p",
                    hardware=hardware.name,
                    datatype=datatype,
                    message_bytes=boundary_bytes,
                    participants=2,
                    network_tier=network_binding.pipeline_parallel,
                    engine=EngineKind.COLLECTIVE.value,
                    hardware_revision=hardware.evidence_revision,
                    implementation=cost_context.implementation_for("p2p", "p2p"),
                    implementation_revision=cost_context.implementation_revision_for("p2p", "p2p"),
                    runtime=cost_context.runtime,
                    runtime_revision=cost_context.runtime_revision,
                    topology=cost_context.topology,
                    power_mode=cost_context.power_mode,
                    dimensions=resolved_pipeline_dimensions,
                )
            ).estimate
            one_hop_seconds = pipeline_estimate.seconds
        pipeline_seconds = (mapping.pipeline_parallel - 1) * one_hop_seconds

    blocks_per_stage = model.block_count // mapping.pipeline_parallel
    boundary_bytes = _concrete_buffer_size(next(buffer for buffer in plan.buffers if buffer.id == plan.inputs[0]))
    memory = InferencePhaseMemory(
        weights=_semantic_buffer_size(plan, "block_weights") * blocks_per_stage,
        kv_cache=_semantic_buffer_size(plan, "kv_cache") * blocks_per_stage,
        working_upper_bound=_semantic_buffer_size(plan, "block_working_upper_bound"),
        pipeline_buffers=boundary_bytes * (2 if mapping.pipeline_parallel > 1 else 1),
    )
    revisions = {hardware.evidence_revision: "analytical-system-profile"}
    for item in task_estimates:
        revisions[item.evidence_revision] = item.evidence_provider
        if item.evidence_source_revision != item.evidence_revision:
            revisions[item.evidence_source_revision] = f"source:{item.evidence_provider}"
    if cost_resolver is not None:
        revisions[cost_resolver.revision] = "cost-resolver-policy"
    if pipeline_estimate is not None:
        revisions[pipeline_estimate.provider_revision] = pipeline_estimate.provider
        if pipeline_estimate.source_revision != pipeline_estimate.provider_revision:
            revisions[pipeline_estimate.source_revision] = f"source:{pipeline_estimate.provider}"
    return InferencePhaseEstimate(
        mode=mode,
        phase=phase,
        batch_size=batch_size,
        query_tokens=query_tokens,
        context_tokens=context_tokens,
        tasks=tuple(task_estimates),
        memory=memory,
        block_seconds=block_seconds,
        transformer_seconds=transformer_seconds,
        pipeline_seconds=pipeline_seconds,
        total_seconds=transformer_seconds + pipeline_seconds,
        evidence_revisions=FrozenDict(revisions),
    )

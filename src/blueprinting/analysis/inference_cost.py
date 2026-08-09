"""Evidence-aware cost view for a static inference phase plan."""

from __future__ import annotations

from dataclasses import dataclass

from ..synthesizer.bindings import InferencePhase
from ..synthesizer.frozen import FrozenDict
from ..synthesizer.ir import CollectiveKind, PlanBuffer, PlanTask, PortablePlanIR
from ..synthesizer.models.transformer import TransformerModelSpec
from ..synthesizer.models.transformer_inference import TransformerInferenceExecutionSpec
from .cost import CostQuery, CostQueryContext, CostResolver, CostSubject
from .cost_model import CalibrationMode, HardwareProfile
from .inference_evidence import InferenceCostProvider, InferenceEvidenceQuery
from .transformer_inference import InferenceInvocation
from .transformer_workload import EngineKind, PhaseWork


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
    hardware: HardwareProfile,
    execution: TransformerInferenceExecutionSpec,
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
        tensor_parallel=execution.tensor_parallel,
        datatype=execution.datatype,
    )


_GEMM_PRIMITIVES = frozenset(
    {
        "attention_pre_projection",
        "attention_post_projection",
        "mlp_up_projection",
        "mlp_down_projection",
    }
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
    hardware: HardwareProfile,
    execution: TransformerInferenceExecutionSpec,
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
    operation = "gemm" if primitive in _GEMM_PRIMITIVES else primitive
    subject = CostSubject.COMMUNICATION if invocation.engine is EngineKind.COLLECTIVE else CostSubject.OPERATOR
    tensor_parallel = execution.tensor_parallel
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
        "kv_cache_datatype": execution.datatype,
    }
    if primitive in _GEMM_PRIMITIVES:
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
    dimensions = _merge_dimensions(dimensions, context.dimensions_for(primitive, operation)).to_dict()
    return CostQuery(
        subject=subject,
        operation=operation,
        hardware=hardware.name,
        datatype=execution.datatype,
        operations=task.workload.operations,
        read_bytes=task.workload.read_bytes,
        write_bytes=task.workload.write_bytes,
        message_bytes=task.workload.message_bytes,
        participants=tensor_parallel if subject is CostSubject.COMMUNICATION else 1,
        network_tier=invocation.network_tier or 0,
        engine=invocation.engine.value,
        hardware_revision=hardware.evidence_revision,
        implementation=context.implementation_for(primitive, operation),
        implementation_revision=context.implementation_revision_for(primitive, operation),
        runtime=context.runtime,
        runtime_revision=context.runtime_revision,
        topology=context.topology,
        power_mode=context.power_mode,
        dimensions=FrozenDict(dimensions),
    )


def _task_estimate(
    task: PlanTask,
    *,
    hardware: HardwareProfile,
    execution: TransformerInferenceExecutionSpec,
    model: TransformerModelSpec,
    batch_size: int,
    query_tokens: int,
    context_tokens: int,
    mode: CalibrationMode,
    cost_provider: InferenceCostProvider | None,
    cost_resolver: CostResolver | None,
    cost_context: CostQueryContext,
) -> InferenceTaskEstimate:
    invocation = _invocation_from_plan_task(task)
    work = invocation.work
    processor = hardware.matrix if invocation.engine is EngineKind.MATRIX else hardware.vector
    compute_seconds = work.operations / processor.throughput(work.operations, mode) if work.operations else 0.0
    memory_seconds = (
        work.memory_bytes / hardware.memory.throughput(work.memory_bytes, mode) if work.memory_bytes else 0.0
    )
    network_seconds = 0.0
    if invocation.engine is EngineKind.COLLECTIVE:
        if invocation.network_tier is None or invocation.collective is None:
            raise ValueError("collective invocation is missing network facts")
        try:
            network = hardware.networks[invocation.network_tier]
        except IndexError as error:
            raise ValueError(f"hardware profile does not define network tier {invocation.network_tier}") from error
        network_seconds = network.time(
            invocation.collective.value,
            work.message_bytes,
            execution.tensor_parallel,
            mode,
        )
    analytical_seconds = hardware.processing_time(compute_seconds, memory_seconds) + network_seconds
    provider_name = "analytical-system-profile"
    revision = hardware.evidence_revision
    source_revision = hardware.evidence_revision
    record_ids: tuple[str, ...] = ()
    match = mode.value
    method = "analytical"
    total_seconds = analytical_seconds
    if cost_resolver is not None:
        resolution = cost_resolver.resolve(
            cost_query_for_inference_task(
                task,
                model=model,
                execution=execution,
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
    elif cost_provider is not None:
        evidence = cost_provider.resolve(
            inference_evidence_query_for(
                invocation,
                model=model,
                execution=execution,
                hardware=hardware,
                batch_size=batch_size,
                query_tokens=query_tokens,
                context_tokens=context_tokens,
            )
        )
        if evidence is not None:
            total_seconds = evidence.seconds
            provider_name = evidence.provider
            revision = evidence.revision
            source_revision = evidence.revision
            match = evidence.match
            method = "external"
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
    )


def _invocation_from_plan_task(task: PlanTask) -> InferenceInvocation:
    """Reconstruct comparison metadata from canonical portable workload facts.

    Costing reads operation and byte counts only from ``PlanTask.workload``.
    The earlier lowering-only ``InferenceInvocation`` is deliberately not
    embedded in ``PortablePlanIR`` as a second source of workload truth.
    """

    attributes = task.workload.attributes
    try:
        phase = InferencePhase(attributes["phase"])
        engine = EngineKind(attributes["engine"])
        name = attributes["name"]
        primitive = attributes["primitive"]
        source_layer = attributes["source_layer"]
    except (KeyError, ValueError) as error:
        raise ValueError(f"portable inference task {task.id} has invalid semantic workload metadata") from error
    for field_name, value in (("name", name), ("primitive", primitive), ("source_layer", source_layer)):
        if not isinstance(value, str) or not value:
            raise ValueError(f"portable inference task {task.id} has invalid {field_name}")

    collective_value = attributes.get("collective", "")
    network_tier_value = attributes.get("network_tier", -1)
    collective = None
    network_tier = None
    if engine is EngineKind.COLLECTIVE:
        try:
            collective = CollectiveKind(collective_value)
        except ValueError as error:
            raise ValueError(f"portable inference task {task.id} has invalid collective metadata") from error
        if isinstance(network_tier_value, bool) or not isinstance(network_tier_value, int) or network_tier_value < 0:
            raise ValueError(f"portable inference task {task.id} has invalid network tier")
        network_tier = network_tier_value
    elif collective_value != "" or network_tier_value != -1:
        raise ValueError(f"local portable inference task {task.id} carries collective metadata")

    return InferenceInvocation(
        name=name,
        source_layer=source_layer,
        primitive=primitive,
        phase=phase,
        engine=engine,
        work=PhaseWork(
            operations=task.workload.operations,
            read_bytes=task.workload.read_bytes,
            write_bytes=task.workload.write_bytes,
            message_bytes=task.workload.message_bytes,
        ),
        collective=collective,
        network_tier=network_tier,
    )


def _concrete_buffer_size(buffer: PlanBuffer) -> int:
    size = buffer.size_bytes
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise TypeError(f"portable inference buffer {buffer.id} must have a concrete non-negative size")
    return size


def _semantic_buffer_size(plan: PortablePlanIR, semantic: str) -> int:
    buffers = tuple(buffer for buffer in plan.buffers if buffer.attributes.get("semantic") == semantic)
    if len(buffers) != 1:
        raise ValueError(f"portable inference plan must contain exactly one {semantic!r} buffer")
    return _concrete_buffer_size(buffers[0])


def estimate_inference_phase(
    plan: PortablePlanIR,
    hardware: HardwareProfile,
    mode: CalibrationMode = CalibrationMode.SYSTEM_EVIDENCE,
    *,
    cost_provider: InferenceCostProvider | None = None,
    cost_resolver: CostResolver | None = None,
    cost_context: CostQueryContext = CostQueryContext(),
) -> InferencePhaseEstimate:
    """Cost one prefill or decode phase point without queueing assumptions."""

    model = plan.attributes.get("model_spec")
    execution = plan.attributes.get("inference_execution_spec")
    phase = plan.attributes.get("inference_phase")
    if not isinstance(model, TransformerModelSpec):
        raise TypeError("portable inference plan is missing TransformerModelSpec")
    if not isinstance(execution, TransformerInferenceExecutionSpec):
        raise TypeError("portable inference plan is missing TransformerInferenceExecutionSpec")
    if not isinstance(phase, InferencePhase):
        raise TypeError("portable inference plan is missing InferencePhase")
    if hardware.datatype != execution.datatype:
        raise ValueError("hardware profile datatype does not match inference execution datatype")
    if cost_provider is not None and cost_resolver is not None:
        raise ValueError("cost_provider and cost_resolver are mutually exclusive")
    if not isinstance(cost_context, CostQueryContext):
        raise TypeError("cost_context must be CostQueryContext")
    batch_size = plan.attributes.get("batch_size")
    query_tokens = plan.attributes.get("query_tokens")
    context_tokens = plan.attributes.get("context_tokens")
    if any(not isinstance(value, int) for value in (batch_size, query_tokens, context_tokens)):
        raise TypeError("portable inference plan has non-concrete workload facts")

    task_estimates = []
    for task in plan.tasks:
        task_estimates.append(
            _task_estimate(
                task,
                hardware=hardware,
                execution=execution,
                model=model,
                batch_size=batch_size,
                query_tokens=query_tokens,
                context_tokens=context_tokens,
                mode=mode,
                cost_provider=cost_provider,
                cost_resolver=cost_resolver,
                cost_context=cost_context,
            )
        )

    block_seconds = sum(item.total_seconds for item in task_estimates)
    transformer_seconds = block_seconds * model.block_count
    pipeline_seconds = 0.0
    pipeline_estimate = None
    if execution.pipeline_parallel > 1:
        boundary_bytes = _concrete_buffer_size(next(buffer for buffer in plan.buffers if buffer.id == plan.inputs[0]))
        if cost_resolver is None:
            try:
                network = hardware.networks[execution.pipeline_parallel_network]
            except IndexError as error:
                raise ValueError(
                    f"hardware profile does not define network tier {execution.pipeline_parallel_network}"
                ) from error
            one_hop_seconds = network.time("p2p", boundary_bytes, 2, mode)
        else:
            pipeline_dimensions = {
                "semantic_operation": "p2p",
                "phase": phase.value,
                "model_name": model.name,
                "batch_size": batch_size,
                "query_tokens": query_tokens,
                "context_tokens": context_tokens,
                "pipeline_parallel": execution.pipeline_parallel,
            }
            pipeline_dimensions = _merge_dimensions(
                pipeline_dimensions,
                cost_context.dimensions_for("p2p", "p2p"),
            )
            pipeline_estimate = cost_resolver.resolve(
                CostQuery(
                    subject=CostSubject.COMMUNICATION,
                    operation="p2p",
                    hardware=hardware.name,
                    datatype=execution.datatype,
                    message_bytes=boundary_bytes,
                    participants=2,
                    network_tier=execution.pipeline_parallel_network,
                    engine=EngineKind.COLLECTIVE.value,
                    hardware_revision=hardware.evidence_revision,
                    implementation=cost_context.implementation_for("p2p", "p2p"),
                    implementation_revision=cost_context.implementation_revision_for("p2p", "p2p"),
                    runtime=cost_context.runtime,
                    runtime_revision=cost_context.runtime_revision,
                    topology=cost_context.topology,
                    power_mode=cost_context.power_mode,
                    dimensions=pipeline_dimensions,
                )
            ).estimate
            one_hop_seconds = pipeline_estimate.seconds
        pipeline_seconds = (execution.pipeline_parallel - 1) * one_hop_seconds

    blocks_per_stage = model.block_count // execution.pipeline_parallel
    boundary_bytes = _concrete_buffer_size(next(buffer for buffer in plan.buffers if buffer.id == plan.inputs[0]))
    memory = InferencePhaseMemory(
        weights=_semantic_buffer_size(plan, "block_weights") * blocks_per_stage,
        kv_cache=_semantic_buffer_size(plan, "kv_cache") * blocks_per_stage,
        working_upper_bound=_semantic_buffer_size(plan, "block_working_upper_bound"),
        pipeline_buffers=boundary_bytes * (2 if execution.pipeline_parallel > 1 else 1),
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

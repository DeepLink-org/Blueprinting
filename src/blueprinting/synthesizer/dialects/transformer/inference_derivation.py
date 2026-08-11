"""Inference lowerings from a semantic decoder to a phase-local work plan."""

from __future__ import annotations

from typing import cast

from blueprinting.mapping import (
    ForwardOnly,
    PipelineParallel,
    ReplicaParallel,
    SingleStage,
    TensorParallel,
    TransformerInferenceMappingSpec,
    TransformerInferenceParallelism,
)
from blueprinting.workload import TransformerModelSpec

from ...bindings import InferencePhase, InferenceWorkload
from ...ids import BufferId, Lineage, NodeId, ValueId
from ...passes.authoring import RelationCheckContext, relation
from ...session import SynthesisSession
from ...stages.common import Effect, EffectKind, OperationName, TensorType
from ...stages.distributed.ir import (
    Collective,
    CollectiveKind,
    DistributedTask,
    DistributedTaskIR,
    DistributedValue,
    LocalCompute,
    LogicalMesh,
    MeshAxis,
    ReductionKind,
    ShardingSpec,
    collective_kind,
    make_collective_spec,
)
from ...stages.model.ir import ModelIR, ModelOperation, ModelValue, ValueRole
from ...stages.portable_plan.ir import (
    AbstractStorageClass,
    CollectiveTask,
    ComputeTask,
    ImplementationRequirement,
    ObjectiveDirection,
    ObjectiveKind,
    PlanBuffer,
    PlanBufferRole,
    PlanObjective,
    PlanTask,
    PortablePlanIR,
    ResourceKind,
    ResourceRequirement,
    ResourceScope,
    WorkloadFacts,
)
from .common import EngineKind
from .inference import InferenceInvocation, derive_transformer_inference_block
from .semantics import (
    TransformerBufferSemantic,
    TransformerInferenceDistributedTaskSemantic,
    TransformerInferencePlanSemantic,
    TransformerInferenceProgramSemantic,
    TransformerInferenceStrategySemantic,
    TransformerInferenceWorkloadSemantic,
    TransformerModelOperationSemantic,
    inference_task_semantic,
)


def _parallelism(mapping: TransformerInferenceMappingSpec) -> tuple[int, int, int]:
    """Destructure the TP × PP × replica inference strategy."""

    match mapping.parallelism:
        case TransformerInferenceParallelism(
            tensor=TensorParallel(degree=tp),
            pipeline=PipelineParallel(degree=pp, schedule=SingleStage() | ForwardOnly()),
            replicas=ReplicaParallel(degree=replicas),
        ):
            return tp, pp, replicas
        case _:
            raise TypeError("unsupported Transformer inference parallel strategy")


def _semantic_specs(
    ir: ModelIR,
    session: SynthesisSession,
) -> tuple[TransformerModelSpec, TransformerInferenceMappingSpec, InferencePhase, int, int, int, str]:
    if len(ir.operations) != 1 or ir.operations[0].operation != OperationName("transformer", "decoder_inference"):
        raise ValueError("Transformer inference distribution expects one transformer.decoder_inference operation")
    operation_semantic = ir.operations[0].semantic
    if not isinstance(operation_semantic, TransformerModelOperationSemantic):
        raise TypeError("model operation is missing a typed TransformerModelSpec")
    model = operation_semantic.model
    workload = session.bindings.workload
    strategy = session.bindings.strategy
    if workload is None or strategy is None:
        raise ValueError("Transformer inference distribution requires workload and strategy bindings")
    if not isinstance(workload.mode, InferenceWorkload):
        raise ValueError("Transformer inference requires an inference workload binding")
    inference_phase = workload.mode.phase
    strategy_semantic = strategy.semantic
    workload_semantic = workload.semantic
    if not isinstance(strategy_semantic, TransformerInferenceStrategySemantic):
        raise TypeError("strategy binding is missing a typed TransformerInferenceMappingSpec")
    if not isinstance(workload_semantic, TransformerInferenceWorkloadSemantic):
        raise TypeError("workload binding is missing typed Transformer inference facts")
    mapping = strategy_semantic.mapping
    batch_size = workload_semantic.batch_size
    context_tokens = workload_semantic.context_tokens
    query_tokens = context_tokens if inference_phase is InferencePhase.PREFILL else 1
    datatype = workload_semantic.datatype
    mapping.validate_model(model)
    return model, mapping, inference_phase, batch_size, query_tokens, context_tokens, datatype


def _verify_inference_distributed_value(
    source: ModelValue,
    target: DistributedValue,
    context: RelationCheckContext,
) -> None:
    target_ir = context.target_ir
    if not isinstance(target_ir, DistributedTaskIR):
        raise TypeError("inference value invariant requires DistributedTaskIR")
    if target.source_value != source.id or target.role is not source.role:
        raise ValueError("distributed inference value must retain its source identity and role")
    if target.type.dtype != source.type.dtype:
        raise ValueError("distributed inference value must preserve datatype")
    if target.owners != tuple(range(target_ir.mesh.size)):
        raise ValueError("distributed inference value owners must cover the logical mesh")


def _verify_inference_distributed_task(
    _source: ModelOperation,
    target: DistributedTask,
    context: RelationCheckContext,
) -> None:
    target_ir = context.target_ir
    if not isinstance(target_ir, DistributedTaskIR):
        raise TypeError("inference task invariant requires DistributedTaskIR")
    if target.ranks != tuple(range(target_ir.mesh.size)):
        raise ValueError("distributed inference task ranks must cover the logical mesh")
    semantic = target.semantic
    if not isinstance(semantic, TransformerInferenceDistributedTaskSemantic):
        raise TypeError("distributed inference task must retain its typed invocation")
    invocation = semantic.invocation
    if invocation.engine is EngineKind.COLLECTIVE:
        if not isinstance(target.body, Collective) or invocation.collective is None:
            raise ValueError("collective invocation must lower to a collective body")
        if collective_kind(target.body.spec) is not invocation.collective:
            raise ValueError("collective body kind differs from its inference invocation")
        if target.body.spec.message_bytes != invocation.work.message_bytes:
            raise ValueError("collective body must conserve exact message bytes")
    elif not isinstance(target.body, LocalCompute):
        raise ValueError("non-collective inference invocation must lower to local compute")


def _verify_inference_boundary_buffer(
    source: DistributedValue,
    target: PlanBuffer,
    _context: RelationCheckContext,
) -> None:
    expected_role = {
        ValueRole.INPUT: PlanBufferRole.INPUT,
        ValueRole.OUTPUT: PlanBufferRole.OUTPUT,
    }.get(source.role)
    if expected_role is None or target.role is not expected_role:
        raise ValueError("inference boundary buffer must preserve its distributed value role")
    if target.storage_class is not AbstractStorageClass.DEVICE_LOCAL:
        raise ValueError("inference boundary buffer must remain abstract device-local storage")


def _verify_inference_cache_buffer(
    source: DistributedValue,
    target: PlanBuffer,
    context: RelationCheckContext,
) -> None:
    source_ir = context.source_ir
    if not isinstance(source_ir, DistributedTaskIR) or not isinstance(
        source_ir.semantic, TransformerInferenceProgramSemantic
    ):
        raise TypeError("cache invariant requires typed inference program semantics")
    if source.role is not ValueRole.KV_CACHE:
        raise ValueError("cache buffer must originate from a distributed KV-cache value")
    if target.role is not PlanBufferRole.STATE or target.storage_class is not AbstractStorageClass.PERSISTENT:
        raise ValueError("KV cache must remain persistent state")
    if target.size_bytes != source_ir.semantic.block_memory.kv_cache:
        raise ValueError("KV-cache capacity differs from the derived block-memory fact")
    if target.semantic != TransformerBufferSemantic("kv_cache", phase=source_ir.semantic.phase):
        raise ValueError("KV-cache semantic projection is inconsistent")


def _verify_inference_plan_task(
    source: DistributedTask,
    target: PlanTask,
    context: RelationCheckContext,
) -> None:
    source_ir = context.source_ir
    semantic = source.semantic
    if not isinstance(source_ir, DistributedTaskIR) or not isinstance(
        source_ir.semantic, TransformerInferenceProgramSemantic
    ):
        raise TypeError("plan task invariant requires typed inference program semantics")
    if not isinstance(semantic, TransformerInferenceDistributedTaskSemantic):
        raise TypeError("plan task source must retain its typed inference invocation")
    invocation = semantic.invocation
    expected_work = WorkloadFacts(
        operations=invocation.work.operations,
        read_bytes=invocation.work.read_bytes,
        write_bytes=invocation.work.write_bytes,
        message_bytes=invocation.work.message_bytes,
    )
    if target.workload != expected_work:
        raise ValueError("portable inference task must conserve exact workload facts")
    if target.operation != source.operation or target.logical_ranks != source.ranks or target.effects != source.effects:
        raise ValueError("portable inference task must preserve operation, logical ranks, and effects")
    expected_semantic = inference_task_semantic(
        invocation,
        query_tokens=source_ir.semantic.query_tokens,
        context_tokens=source_ir.semantic.context_tokens,
    )
    if target.semantic != expected_semantic:
        raise ValueError("portable inference task semantic projection is inconsistent")
    if invocation.engine is EngineKind.COLLECTIVE:
        if not isinstance(target.body, CollectiveTask):
            raise ValueError("collective invocation must produce CollectiveTask")
    elif not isinstance(target.body, ComputeTask):
        raise ValueError("compute invocation must produce ComputeTask")


def _verify_inference_weight_buffer(
    _sources: DistributedTask | tuple[DistributedTask, ...],
    target: PlanBuffer,
    context: RelationCheckContext,
) -> None:
    source_ir = context.source_ir
    if not isinstance(source_ir, DistributedTaskIR) or not isinstance(
        source_ir.semantic, TransformerInferenceProgramSemantic
    ):
        raise TypeError("weight invariant requires typed inference program semantics")
    if target.role is not PlanBufferRole.CONSTANT or target.storage_class is not AbstractStorageClass.PERSISTENT:
        raise ValueError("inference weights must remain persistent constants")
    if target.size_bytes != source_ir.semantic.block_memory.weights:
        raise ValueError("weight capacity differs from the derived block-memory fact")


def _verify_inference_workspace_buffer(
    _sources: DistributedTask | tuple[DistributedTask, ...],
    target: PlanBuffer,
    context: RelationCheckContext,
) -> None:
    source_ir = context.source_ir
    if not isinstance(source_ir, DistributedTaskIR) or not isinstance(
        source_ir.semantic, TransformerInferenceProgramSemantic
    ):
        raise TypeError("workspace invariant requires typed inference program semantics")
    if target.role is not PlanBufferRole.WORKSPACE or target.storage_class is not AbstractStorageClass.TRANSIENT:
        raise ValueError("inference workspace must remain transient workspace storage")
    if target.size_bytes != source_ir.semantic.block_memory.working_upper_bound:
        raise ValueError("workspace capacity differs from the derived upper bound")


INFERENCE_DISTRIBUTION_RULES = (
    relation(
        "transformer-inference-distribute",
        "Map inference inputs, outputs, and KV state onto the logical tensor-parallel mesh",
        source=ModelValue,
        target=DistributedValue,
        verifier=_verify_inference_distributed_value,
        introduces=("owners", "logical sharding", "mesh scope"),
        forbids=("physical device", "target implementation", "predicted time"),
    ),
    relation(
        "transformer-inference-decompose",
        "Expand one inference phase into observable local and collective tasks",
        source=ModelOperation,
        target=DistributedTask,
        verifier=_verify_inference_distributed_task,
        introduces=("logical ranks", "task kind", "collective spec", "distributed dependencies"),
        forbids=("physical device", "queue", "kernel", "predicted time"),
    ),
)

INFERENCE_PLANNING_RULES = (
    relation(
        "transformer-inference-plan-buffer",
        "Materialize inference boundary values as abstract plan buffers",
        source=DistributedValue,
        target=PlanBuffer,
        verifier=_verify_inference_boundary_buffer,
        introduces=("exact size", "storage class", "alignment", "producer/consumers"),
        forbids=("memory address", "physical memory region", "predicted time"),
    ),
    relation(
        "transformer-inference-plan-cache",
        "Materialize persistent KV-cache state for the inference phase",
        source=DistributedValue,
        target=PlanBuffer,
        verifier=_verify_inference_cache_buffer,
        introduces=("persistent state buffer", "cache consumers", "phase tag"),
        forbids=("physical memory region", "memory address"),
    ),
    relation(
        "transformer-inference-plan-work",
        "Materialize target-neutral inference work and capability requirements",
        source=DistributedTask,
        target=PlanTask,
        verifier=_verify_inference_plan_task,
        introduces=("exact WorkloadFacts", "resources", "implementation capabilities", "concurrency"),
        forbids=("kernel ID", "physical queue", "empirical duration"),
    ),
    relation(
        "transformer-inference-plan-weights",
        "Materialize persistent block-weight capacity required by the task set",
        source="DistributedTask set",
        target=PlanBuffer,
        verifier=_verify_inference_weight_buffer,
        introduces=("persistent constant buffer", "weight consumers", "alignment"),
        forbids=("physical memory region", "memory address"),
    ),
    relation(
        "transformer-inference-plan-workspace",
        "Materialize an abstract upper bound for transient phase workspace",
        source="DistributedTask set",
        target=PlanBuffer,
        verifier=_verify_inference_workspace_buffer,
        introduces=("transient workspace bound", "workspace consumers", "alignment"),
        forbids=("physical memory region", "memory address", "predicted allocation time"),
    ),
)


def normalize_inference_distribution(ir: ModelIR, session: SynthesisSession) -> DistributedTaskIR:
    """Purely derive logical distributed tasks for Transformer inference."""
    model, mapping, phase, batch_size, query_tokens, context_tokens, datatype = _semantic_specs(ir, session)
    invocations, block_memory = derive_transformer_inference_block(
        model,
        mapping,
        phase=phase,
        batch_size=batch_size,
        context_tokens=context_tokens,
        datatype=datatype,
    )
    tp, _pp, _replicas = _parallelism(mapping)
    ranks = tuple(range(tp))
    mesh = LogicalMesh("local-tensor-parallel-group", (MeshAxis("tp", tp),))
    source_input = ir.inputs[0]
    source_output = ir.outputs[0]
    source_cache = next(value.id for value in ir.values if value.role is ValueRole.KV_CACHE)
    input_id = ValueId.derive(ir.digest, phase.value, "transformer-distributed", "input")
    cache_id = ValueId.derive(ir.digest, phase.value, "transformer-distributed", "kv-cache")
    output_id = ValueId.derive(ir.digest, phase.value, "transformer-distributed", "output")
    boundary_type = TensorType((batch_size, query_tokens, model.hidden_size), datatype)
    cache_type = TensorType((2, batch_size, context_tokens, model.hidden_size), datatype)
    boundary_sharding = ShardingSpec.replicated(boundary_type.rank, ("tp",))
    cache_sharding = ShardingSpec(((), (), (), ("tp",)))

    task_ids = tuple(
        NodeId.derive(ir.digest, phase.value, "transformer-distributed", index, invocation.name)
        for index, invocation in enumerate(invocations)
    )
    tasks = []
    cache_primitives = frozenset({"attention_kv_cache_save", "attention_core"})
    for index, (task_id, invocation) in enumerate(zip(task_ids, invocations)):
        body: Collective | LocalCompute
        match invocation:
            case InferenceInvocation(engine=EngineKind.COLLECTIVE, collective=kind) if kind is not None:
                reduction = (
                    ReductionKind.SUM if kind in {CollectiveKind.ALL_REDUCE, CollectiveKind.REDUCE_SCATTER} else None
                )
                body = Collective(make_collective_spec(kind, ranks, invocation.work.message_bytes, reduction=reduction))
                operation = OperationName("collective", kind.value)
            case InferenceInvocation(engine=EngineKind.MATRIX | EngineKind.VECTOR, collective=None):
                body = LocalCompute()
                operation = OperationName("transformer", f"{invocation.primitive}_{phase.value}")
            case _:
                raise TypeError("unsupported Transformer inference invocation")
        inputs = []
        if index == 0:
            inputs.append(input_id)
        if invocation.primitive in cache_primitives:
            inputs.append(cache_id)
        effects: tuple[Effect, ...]
        match invocation.primitive:
            case "attention_kv_cache_save":
                effects = (Effect(EffectKind.WRITE, "kv_cache"),)
            case "attention_core":
                effects = (Effect(EffectKind.READ, "kv_cache"),)
            case _:
                effects = ()
        tasks.append(
            DistributedTask(
                id=task_id,
                body=body,
                operation=operation,
                ranks=ranks,
                inputs=tuple(inputs),
                outputs=(output_id,) if index == len(invocations) - 1 else (),
                dependencies=(task_ids[index - 1],) if index else (),
                lineage=Lineage.lowered("transformer-inference-decompose", (ir.operations[0].id,)),
                effects=effects,
                semantic=TransformerInferenceDistributedTaskSemantic(invocation),
            )
        )

    distributed = DistributedTaskIR(
        name=f"{model.name}-{phase.value}-local-tp-block",
        source_model_digest=ir.digest,
        mesh=mesh,
        values=(
            DistributedValue(
                input_id,
                boundary_type,
                ValueRole.INPUT,
                boundary_sharding,
                ranks,
                Lineage.lowered("transformer-inference-distribute", (source_input,)),
                source_value=source_input,
            ),
            DistributedValue(
                cache_id,
                cache_type,
                ValueRole.KV_CACHE,
                cache_sharding,
                ranks,
                Lineage.lowered("transformer-inference-distribute", (source_cache,)),
                source_value=source_cache,
            ),
            DistributedValue(
                output_id,
                boundary_type,
                ValueRole.OUTPUT,
                boundary_sharding,
                ranks,
                Lineage.lowered("transformer-inference-distribute", (source_output,)),
                source_value=source_output,
            ),
        ),
        tasks=tuple(tasks),
        inputs=(input_id, cache_id),
        outputs=(output_id,),
        semantic=TransformerInferenceProgramSemantic(
            model,
            mapping,
            phase,
            batch_size,
            query_tokens,
            context_tokens,
            datatype,
            "one-local-tensor-parallel-block-phase",
            block_memory,
        ),
    )
    return distributed


def _plan_resources(invocation: InferenceInvocation) -> tuple[ResourceRequirement, ...]:
    resources = []
    if invocation.work.operations:
        resources.append(
            ResourceRequirement(
                ResourceKind.COMPUTE,
                invocation.work.operations,
                ResourceScope.PER_RANK,
            )
        )
    if invocation.work.memory_bytes:
        resources.append(
            ResourceRequirement(ResourceKind.MEMORY_BANDWIDTH, invocation.work.memory_bytes, ResourceScope.PER_RANK)
        )
    if invocation.work.message_bytes:
        resources.append(
            ResourceRequirement(
                ResourceKind.NETWORK,
                invocation.work.message_bytes,
                ResourceScope.PER_RANK,
            )
        )
    return tuple(resources)


def _implementation(invocation: InferenceInvocation) -> ImplementationRequirement:
    match invocation:
        case InferenceInvocation(engine=EngineKind.COLLECTIVE, collective=kind) if kind is not None:
            return ImplementationRequirement(kind.value, alternatives=("collective-library", "network-engine"))
        case InferenceInvocation(engine=EngineKind.COLLECTIVE):
            raise TypeError("collective inference invocation is missing its collective kind")
        case _:
            pass
    alternatives = {
        "attention_core": ("flash-attention", "paged-attention", "dense-attention"),
        "attention_kv_cache_save": ("fused-kv-write", "vector-engine"),
        "attention_pre_projection": ("tensor-core", "matrix-engine"),
        "attention_post_projection": ("tensor-core", "matrix-engine"),
        "mlp_up_projection": ("tensor-core", "matrix-engine"),
        "mlp_down_projection": ("tensor-core", "matrix-engine"),
    }.get(invocation.primitive)
    if alternatives is not None:
        return ImplementationRequirement(invocation.primitive, alternatives=alternatives)
    return ImplementationRequirement(invocation.primitive, alternatives=("vector-engine",))


def normalize_inference_plan(ir: DistributedTaskIR, session: SynthesisSession) -> PortablePlanIR:
    """Purely derive target-neutral exact work for Transformer inference."""
    program = ir.semantic
    if not isinstance(program, TransformerInferenceProgramSemantic):
        raise TypeError("distributed inference IR is missing InferenceBlockMemoryFacts")
    model = program.model
    mapping = program.mapping
    phase = program.phase
    block_memory = program.block_memory
    strategy = session.bindings.strategy
    if strategy is None:
        raise ValueError("portable inference planning requires a strategy binding")
    if not isinstance(strategy.semantic, TransformerInferenceStrategySemantic):
        raise TypeError("strategy binding is missing typed Transformer inference semantics")
    if strategy.semantic.mapping != mapping:
        raise ValueError("strategy binding does not match the distributed program semantics")

    input_id = BufferId.derive(ir.digest, "transformer-inference-portable", "input")
    output_id = BufferId.derive(ir.digest, "transformer-inference-portable", "output")
    weight_id = BufferId.derive(ir.digest, "transformer-inference-portable", "weights")
    cache_id = BufferId.derive(ir.digest, "transformer-inference-portable", "kv-cache")
    workspace_id = BufferId.derive(ir.digest, "transformer-inference-portable", "workspace-upper-bound")
    task_ids = tuple(
        NodeId.derive(ir.digest, "transformer-inference-portable", index, task.id)
        for index, task in enumerate(ir.tasks)
    )
    task_semantics = tuple(task.semantic for task in ir.tasks)
    if any(not isinstance(item, TransformerInferenceDistributedTaskSemantic) for item in task_semantics):
        raise TypeError("distributed inference task is missing InferenceInvocation")
    typed_semantics = cast(tuple[TransformerInferenceDistributedTaskSemantic, ...], task_semantics)
    invocations = tuple(item.invocation for item in typed_semantics)
    weight_consumers = tuple(
        task_id for task_id, invocation in zip(task_ids, invocations) if invocation.engine is EngineKind.MATRIX
    )
    cache_consumers = tuple(
        task_id
        for task_id, invocation in zip(task_ids, invocations)
        if invocation.primitive in {"attention_kv_cache_save", "attention_core"}
    )

    tasks = []
    for index, (source_task, task_id, invocation) in enumerate(zip(ir.tasks, task_ids, invocations)):
        inputs = [workspace_id]
        if index == 0:
            inputs.append(input_id)
        if invocation.engine is EngineKind.MATRIX:
            inputs.append(weight_id)
        if invocation.primitive in {"attention_kv_cache_save", "attention_core"}:
            inputs.append(cache_id)
        tasks.append(
            PlanTask(
                id=task_id,
                body=CollectiveTask() if invocation.engine is EngineKind.COLLECTIVE else ComputeTask(),
                operation=source_task.operation,
                dependencies=(task_ids[index - 1],) if index else (),
                inputs=tuple(inputs),
                outputs=(output_id,) if index == len(ir.tasks) - 1 else (),
                logical_ranks=source_task.ranks,
                workload=WorkloadFacts(
                    operations=invocation.work.operations,
                    read_bytes=invocation.work.read_bytes,
                    write_bytes=invocation.work.write_bytes,
                    message_bytes=invocation.work.message_bytes,
                ),
                lineage=Lineage.lowered("transformer-inference-plan-work", (source_task.id,)),
                resources=_plan_resources(invocation),
                implementations=(_implementation(invocation),),
                concurrency_group=("network" if invocation.engine is EngineKind.COLLECTIVE else "compute"),
                effects=source_task.effects,
                semantic=inference_task_semantic(
                    invocation,
                    query_tokens=program.query_tokens,
                    context_tokens=program.context_tokens,
                ),
            )
        )

    return PortablePlanIR(
        name=f"{model.name}-{phase.value}-local-tp-block-plan",
        source_distributed_digest=ir.digest,
        strategy_fingerprint=strategy.fingerprint,
        planner_revision="transformer-inference-work-analysis",
        tasks=tuple(tasks),
        buffers=(
            PlanBuffer(
                input_id,
                block_memory.boundary,
                PlanBufferRole.INPUT,
                AbstractStorageClass.DEVICE_LOCAL,
                Lineage.lowered("transformer-inference-plan-buffer", (ir.inputs[0],)),
                consumers=(task_ids[0],),
                alignment_bytes=16,
            ),
            PlanBuffer(
                output_id,
                block_memory.boundary,
                PlanBufferRole.OUTPUT,
                AbstractStorageClass.DEVICE_LOCAL,
                Lineage.lowered("transformer-inference-plan-buffer", (ir.outputs[0],)),
                producer=task_ids[-1],
                alignment_bytes=16,
            ),
            PlanBuffer(
                weight_id,
                block_memory.weights,
                PlanBufferRole.CONSTANT,
                AbstractStorageClass.PERSISTENT,
                Lineage.lowered("transformer-inference-plan-weights", tuple(task.id for task in ir.tasks)),
                consumers=weight_consumers,
                alignment_bytes=16,
                semantic=TransformerBufferSemantic("block_weights"),
            ),
            PlanBuffer(
                cache_id,
                block_memory.kv_cache,
                PlanBufferRole.STATE,
                AbstractStorageClass.PERSISTENT,
                Lineage.lowered("transformer-inference-plan-cache", (ir.inputs[1],)),
                consumers=cache_consumers,
                alignment_bytes=16,
                semantic=TransformerBufferSemantic("kv_cache", phase=phase),
            ),
            PlanBuffer(
                workspace_id,
                block_memory.working_upper_bound,
                PlanBufferRole.WORKSPACE,
                AbstractStorageClass.TRANSIENT,
                Lineage.lowered("transformer-inference-plan-workspace", tuple(task.id for task in ir.tasks)),
                consumers=task_ids,
                alignment_bytes=16,
                semantic=TransformerBufferSemantic(
                    "block_working_upper_bound",
                    bound="unfused-score-materialization",
                ),
            ),
        ),
        inputs=(input_id,),
        outputs=(output_id,),
        objectives=(
            PlanObjective(ObjectiveKind.LATENCY, ObjectiveDirection.MINIMIZE),
            PlanObjective(ObjectiveKind.PEAK_MEMORY, ObjectiveDirection.MINIMIZE),
        ),
        semantic=TransformerInferencePlanSemantic(
            model,
            mapping,
            phase,
            program.batch_size,
            program.query_tokens,
            program.context_tokens,
            program.datatype,
            program.scope,
            block_memory,
        ),
    )

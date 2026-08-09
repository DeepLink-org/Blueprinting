"""Inference lowerings from a semantic decoder to a phase-local work plan."""

from __future__ import annotations

from blueprinting.mapping import TransformerInferenceMappingSpec
from blueprinting.schema.frozen import FrozenDict
from blueprinting.workload import TransformerModelSpec

from ..axes import BindingAxis
from ..bindings import InferencePhase, WorkloadMode
from ..dialects.transformer import (
    EngineKind,
    InferenceBlockMemoryFacts,
    InferenceInvocation,
    derive_transformer_inference_block,
)
from ..ids import BufferId, Lineage, NodeId, ValueId
from ..ir import (
    AbstractStorageClass,
    CollectiveKind,
    CollectiveSpec,
    DistributedTask,
    DistributedTaskIR,
    DistributedTaskKind,
    DistributedValue,
    Effect,
    EffectKind,
    ImplementationRequirement,
    LogicalMesh,
    MeshAxis,
    ModelIR,
    ObjectiveDirection,
    ObjectiveKind,
    OperationName,
    PlanBuffer,
    PlanBufferRole,
    PlanObjective,
    PlanTask,
    PlanTaskKind,
    PortablePlanIR,
    ReductionKind,
    ResourceKind,
    ResourceRequirement,
    ResourceScope,
    ShardingSpec,
    TensorType,
    ValueRole,
    WorkloadFacts,
)
from ..passes import DerivationPass, PassContext, PassContract


def _semantic_specs(
    ir: ModelIR,
    context: PassContext,
) -> tuple[TransformerModelSpec, TransformerInferenceMappingSpec, InferencePhase, int, int, int, str]:
    if len(ir.operations) != 1 or ir.operations[0].operation != OperationName("transformer", "decoder_inference"):
        raise ValueError("Transformer inference distribution expects one transformer.decoder_inference operation")
    model = ir.operations[0].attributes.get("model_spec")
    if not isinstance(model, TransformerModelSpec):
        raise TypeError("model operation is missing a typed TransformerModelSpec")
    workload = context.session.bindings.workload
    strategy = context.session.bindings.strategy
    if workload is None or strategy is None:
        raise ValueError("Transformer inference distribution requires workload and strategy bindings")
    if workload.mode is not WorkloadMode.INFERENCE or workload.inference_phase is None:
        raise ValueError("Transformer inference requires an explicit inference phase")
    mapping = strategy.attributes.get("inference_mapping_spec")
    if not isinstance(mapping, TransformerInferenceMappingSpec):
        raise TypeError("strategy binding is missing a typed TransformerInferenceMappingSpec")
    if (
        strategy.tensor_parallel != mapping.tensor_parallel
        or strategy.pipeline_parallel != mapping.pipeline_parallel
        or strategy.data_parallel != mapping.replicas
    ):
        raise ValueError("strategy binding is inconsistent with inference execution facts")
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in (workload.batch_size, workload.sequence_length)
    ):
        raise TypeError("static Transformer inference requires concrete integer workload bindings")
    batch_size = workload.batch_size
    context_tokens = workload.sequence_length
    query_tokens = context_tokens if workload.inference_phase is InferencePhase.PREFILL else 1
    if workload.attributes.get("query_tokens") != query_tokens:
        raise ValueError("workload query_tokens attribute is inconsistent with the inference phase")
    if workload.attributes.get("context_tokens") != context_tokens:
        raise ValueError("workload context_tokens attribute is inconsistent with sequence_length")
    datatype = workload.attributes.get("datatype")
    if not isinstance(datatype, str):
        raise TypeError("inference workload binding is missing a concrete datatype")
    mapping.validate_model(model)
    return model, mapping, workload.inference_phase, batch_size, query_tokens, context_tokens, datatype


class DistributeTransformerInferencePass(DerivationPass[ModelIR, DistributedTaskIR]):
    """Expand one phase into observable local and collective components."""

    contract = PassContract.create(
        "transformer-inference-distribute-v2",
        ModelIR,
        DistributedTaskIR,
        required_bindings=frozenset({BindingAxis.WORKLOAD, BindingAxis.STRATEGY}),
    )

    def run(self, ir: ModelIR, context: PassContext) -> DistributedTaskIR:
        model, mapping, phase, batch_size, query_tokens, context_tokens, datatype = _semantic_specs(ir, context)
        invocations, block_memory = derive_transformer_inference_block(
            model,
            mapping,
            phase=phase,
            batch_size=batch_size,
            context_tokens=context_tokens,
            datatype=datatype,
        )
        ranks = tuple(range(mapping.tensor_parallel))
        mesh = LogicalMesh("local-tensor-parallel-group", (MeshAxis("tp", mapping.tensor_parallel),))
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
            collective = None
            kind = DistributedTaskKind.LOCAL_COMPUTE
            if invocation.engine is EngineKind.COLLECTIVE:
                kind = DistributedTaskKind.COLLECTIVE
                collective = CollectiveSpec(
                    kind=invocation.collective,
                    participants=ranks,
                    message_bytes=invocation.work.message_bytes,
                    reduction=(
                        ReductionKind.SUM
                        if invocation.collective in {CollectiveKind.ALL_REDUCE, CollectiveKind.REDUCE_SCATTER}
                        else None
                    ),
                )
            operation = (
                OperationName("collective", invocation.collective.value)
                if invocation.collective is not None
                else OperationName("transformer", f"{invocation.primitive}_{phase.value}")
            )
            inputs = []
            if index == 0:
                inputs.append(input_id)
            if invocation.primitive in cache_primitives:
                inputs.append(cache_id)
            effects = ()
            if invocation.primitive == "attention_kv_cache_save":
                effects = (Effect(EffectKind.WRITE, "kv_cache"),)
            elif invocation.primitive == "attention_core":
                effects = (Effect(EffectKind.READ, "kv_cache"),)
            tasks.append(
                DistributedTask(
                    id=task_id,
                    kind=kind,
                    operation=operation,
                    ranks=ranks,
                    inputs=tuple(inputs),
                    outputs=(output_id,) if index == len(invocations) - 1 else (),
                    dependencies=(task_ids[index - 1],) if index else (),
                    lineage=Lineage.lowered("transformer-inference-decompose", (ir.operations[0].id,)),
                    collective=collective,
                    effects=effects,
                    attributes=FrozenDict({"invocation": invocation}),
                )
            )

        return DistributedTaskIR(
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
            attributes=FrozenDict(
                {
                    "model_spec": model,
                    "inference_mapping_spec": mapping,
                    "inference_phase": phase,
                    "batch_size": batch_size,
                    "query_tokens": query_tokens,
                    "context_tokens": context_tokens,
                    "datatype": datatype,
                    "block_memory": block_memory,
                    "scope": "one-local-tensor-parallel-block-phase",
                }
            ),
        )


def _plan_resources(invocation: InferenceInvocation) -> tuple[ResourceRequirement, ...]:
    resources = []
    if invocation.work.operations:
        resources.append(
            ResourceRequirement(
                ResourceKind.COMPUTE,
                invocation.work.operations,
                ResourceScope.PER_RANK,
                FrozenDict({"engine": invocation.engine.value}),
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
    if invocation.engine is EngineKind.COLLECTIVE:
        return ImplementationRequirement(
            invocation.collective.value, alternatives=("collective-library", "network-engine")
        )
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


class PlanTransformerInferencePass(DerivationPass[DistributedTaskIR, PortablePlanIR]):
    """Materialize a phase plan without target placement or measured time."""

    contract = PassContract.create(
        "transformer-inference-plan-work-v2",
        DistributedTaskIR,
        PortablePlanIR,
        required_bindings=frozenset({BindingAxis.WORKLOAD, BindingAxis.STRATEGY}),
    )

    def run(self, ir: DistributedTaskIR, context: PassContext) -> PortablePlanIR:
        model = ir.attributes.get("model_spec")
        mapping = ir.attributes.get("inference_mapping_spec")
        phase = ir.attributes.get("inference_phase")
        block_memory = ir.attributes.get("block_memory")
        if not isinstance(model, TransformerModelSpec):
            raise TypeError("distributed inference IR is missing TransformerModelSpec")
        if not isinstance(mapping, TransformerInferenceMappingSpec):
            raise TypeError("distributed inference IR is missing TransformerInferenceMappingSpec")
        if not isinstance(phase, InferencePhase):
            raise TypeError("distributed inference IR is missing InferencePhase")
        if not isinstance(block_memory, InferenceBlockMemoryFacts):
            raise TypeError("distributed inference IR is missing InferenceBlockMemoryFacts")
        strategy = context.session.bindings.strategy
        if strategy is None:
            raise ValueError("portable inference planning requires a strategy binding")

        input_id = BufferId.derive(ir.digest, "transformer-inference-portable", "input")
        output_id = BufferId.derive(ir.digest, "transformer-inference-portable", "output")
        weight_id = BufferId.derive(ir.digest, "transformer-inference-portable", "weights")
        cache_id = BufferId.derive(ir.digest, "transformer-inference-portable", "kv-cache")
        workspace_id = BufferId.derive(ir.digest, "transformer-inference-portable", "workspace-upper-bound")
        task_ids = tuple(
            NodeId.derive(ir.digest, "transformer-inference-portable", index, task.id)
            for index, task in enumerate(ir.tasks)
        )
        invocations = tuple(task.attributes.get("invocation") for task in ir.tasks)
        if any(not isinstance(item, InferenceInvocation) for item in invocations):
            raise TypeError("distributed inference task is missing InferenceInvocation")
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
                    kind=(
                        PlanTaskKind.COLLECTIVE if invocation.engine is EngineKind.COLLECTIVE else PlanTaskKind.COMPUTE
                    ),
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
                        attributes=FrozenDict(
                            {
                                "name": invocation.name,
                                "engine": invocation.engine.value,
                                "phase": invocation.phase.value,
                                "primitive": invocation.primitive,
                                "source_layer": invocation.source_layer,
                                "query_tokens": ir.attributes["query_tokens"],
                                "context_tokens": ir.attributes["context_tokens"],
                                "collective": (
                                    invocation.collective.value if invocation.collective is not None else ""
                                ),
                            }
                        ),
                    ),
                    lineage=Lineage.lowered("transformer-inference-plan-work", (source_task.id,)),
                    resources=_plan_resources(invocation),
                    implementations=(_implementation(invocation),),
                    concurrency_group=("network" if invocation.engine is EngineKind.COLLECTIVE else "compute"),
                    effects=source_task.effects,
                )
            )

        return PortablePlanIR(
            name=f"{model.name}-{phase.value}-local-tp-block-plan",
            source_distributed_digest=ir.digest,
            strategy_fingerprint=strategy.fingerprint,
            planner_revision="transformer-inference-work-analysis-v2",
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
                    attributes=FrozenDict({"semantic": "block_weights"}),
                ),
                PlanBuffer(
                    cache_id,
                    block_memory.kv_cache,
                    PlanBufferRole.STATE,
                    AbstractStorageClass.PERSISTENT,
                    Lineage.lowered("transformer-inference-plan-cache", (ir.inputs[1],)),
                    consumers=cache_consumers,
                    alignment_bytes=16,
                    attributes=FrozenDict({"semantic": "kv_cache", "phase": phase.value}),
                ),
                PlanBuffer(
                    workspace_id,
                    block_memory.working_upper_bound,
                    PlanBufferRole.WORKSPACE,
                    AbstractStorageClass.TRANSIENT,
                    Lineage.lowered("transformer-inference-plan-workspace", tuple(task.id for task in ir.tasks)),
                    consumers=task_ids,
                    alignment_bytes=16,
                    attributes=FrozenDict(
                        {
                            "semantic": "block_working_upper_bound",
                            "bound": "unfused-score-materialization",
                        }
                    ),
                ),
            ),
            inputs=(input_id,),
            outputs=(output_id,),
            objectives=(
                PlanObjective(ObjectiveKind.LATENCY, ObjectiveDirection.MINIMIZE),
                PlanObjective(ObjectiveKind.PEAK_MEMORY, ObjectiveDirection.MINIMIZE),
            ),
            attributes=FrozenDict(
                {
                    "model_spec": model,
                    "inference_mapping_spec": mapping,
                    "inference_phase": phase,
                    "batch_size": ir.attributes["batch_size"],
                    "query_tokens": ir.attributes["query_tokens"],
                    "context_tokens": ir.attributes["context_tokens"],
                    "datatype": ir.attributes["datatype"],
                    "scope": ir.attributes["scope"],
                }
            ),
        )

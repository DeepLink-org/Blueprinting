"""Transformer training lowerings from semantic graph to portable work DAG."""

from __future__ import annotations

from ...analysis.transformer_workload import (
    EngineKind,
    PrimitiveInvocation,
    derive_transformer_block,
)
from ...workload import (
    TensorParallelCommunication,
    TransformerExecutionSpec,
    TransformerModelSpec,
)
from ..axes import BindingAxis
from ..frozen import FrozenDict
from ..ids import BufferId, Lineage, NodeId, ValueId
from ..ir import (
    AbstractStorageClass,
    CollectiveKind,
    CollectiveSpec,
    DistributedTask,
    DistributedTaskIR,
    DistributedTaskKind,
    DistributedValue,
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


def _semantic_specs(ir: ModelIR, context: PassContext) -> tuple[TransformerModelSpec, TransformerExecutionSpec]:
    if len(ir.operations) != 1 or ir.operations[0].operation != OperationName("transformer", "decoder_training"):
        raise ValueError("Transformer distribution expects one transformer.decoder_training operation")
    model = ir.operations[0].attributes.get("model_spec")
    strategy = context.session.bindings.strategy
    if strategy is None:
        raise ValueError("Transformer distribution requires a strategy binding")
    execution = strategy.attributes.get("execution_spec")
    if not isinstance(model, TransformerModelSpec):
        raise TypeError("model operation is missing a typed TransformerModelSpec")
    if not isinstance(execution, TransformerExecutionSpec):
        raise TypeError("strategy binding is missing a typed TransformerExecutionSpec")
    workload = context.session.bindings.workload
    if workload is None:
        raise ValueError("Transformer distribution requires a workload binding")
    expected = (execution.microbatch_size, model.sequence_length, execution.microbatch_count)
    actual = (workload.batch_size, workload.sequence_length, workload.micro_batches)
    if actual != expected:
        raise ValueError(f"workload binding {actual!r} is inconsistent with execution facts {expected!r}")
    if (
        strategy.tensor_parallel != execution.tensor_parallel
        or strategy.pipeline_parallel != execution.pipeline_parallel
        or strategy.data_parallel != execution.data_parallel
    ):
        raise ValueError("strategy binding is inconsistent with Transformer execution facts")
    return model, execution


class DistributeTransformerTrainingPass(DerivationPass[ModelIR, DistributedTaskIR]):
    """Expand one semantic block into explicit local and collective tasks."""

    contract = PassContract.create(
        "transformer-distribute-v1",
        ModelIR,
        DistributedTaskIR,
        required_bindings=frozenset({BindingAxis.WORKLOAD, BindingAxis.STRATEGY}),
    )

    def run(self, ir: ModelIR, context: PassContext) -> DistributedTaskIR:
        model, execution = _semantic_specs(ir, context)
        invocations, block_memory = derive_transformer_block(model, execution)
        ranks = tuple(range(execution.tensor_parallel))
        mesh = LogicalMesh("local-tensor-parallel-group", (MeshAxis("tp", execution.tensor_parallel),))
        source_input = ir.inputs[0]
        source_output = ir.outputs[0]
        input_id = ValueId.derive(ir.digest, "transformer-distributed", "input")
        output_id = ValueId.derive(ir.digest, "transformer-distributed", "output")
        tensor_type = TensorType(
            (execution.microbatch_size, model.sequence_length, model.hidden_size),
            execution.datatype,
        )
        if execution.tensor_parallel_communication is TensorParallelCommunication.REDUCE_SCATTER_ALL_GATHER:
            sharding = ShardingSpec(((), ("tp",), ()))
        else:
            sharding = ShardingSpec.replicated(tensor_type.rank, ("tp",))

        task_ids = tuple(
            NodeId.derive(ir.digest, "transformer-distributed", index, invocation.name)
            for index, invocation in enumerate(invocations)
        )
        tasks = []
        for index, (task_id, invocation) in enumerate(zip(task_ids, invocations)):
            collective = None
            kind = DistributedTaskKind.LOCAL_COMPUTE
            if invocation.engine is EngineKind.COLLECTIVE:
                kind = DistributedTaskKind.COLLECTIVE
                reduction = (
                    ReductionKind.SUM
                    if invocation.collective in {CollectiveKind.ALL_REDUCE, CollectiveKind.REDUCE_SCATTER}
                    else None
                )
                collective = CollectiveSpec(
                    kind=invocation.collective,
                    participants=ranks,
                    message_bytes=invocation.work.message_bytes,
                    reduction=reduction,
                )
            operation = (
                OperationName("collective", invocation.collective.value)
                if invocation.collective is not None
                else OperationName("transformer", f"{invocation.primitive}_{invocation.phase.value}")
            )
            tasks.append(
                DistributedTask(
                    id=task_id,
                    kind=kind,
                    operation=operation,
                    ranks=ranks,
                    inputs=(input_id,) if index == 0 else (),
                    outputs=(output_id,) if index == len(invocations) - 1 else (),
                    dependencies=(task_ids[index - 1],) if index else (),
                    lineage=Lineage.lowered("transformer-decompose", (ir.operations[0].id,)),
                    collective=collective,
                    attributes=FrozenDict({"invocation": invocation}),
                )
            )

        return DistributedTaskIR(
            name=f"{model.name}-local-tp-block",
            source_model_digest=ir.digest,
            mesh=mesh,
            values=(
                DistributedValue(
                    input_id,
                    tensor_type,
                    ValueRole.INPUT,
                    sharding,
                    ranks,
                    Lineage.lowered("transformer-distribute", (source_input,)),
                    source_value=source_input,
                ),
                DistributedValue(
                    output_id,
                    tensor_type,
                    ValueRole.OUTPUT,
                    sharding,
                    ranks,
                    Lineage.lowered("transformer-distribute", (source_output,)),
                    source_value=source_output,
                ),
            ),
            tasks=tuple(tasks),
            inputs=(input_id,),
            outputs=(output_id,),
            attributes=FrozenDict(
                {
                    "model_spec": model,
                    "execution_spec": execution,
                    "block_memory": block_memory,
                    "scope": "one-local-tensor-parallel-block",
                }
            ),
        )


def _plan_resources(invocation: PrimitiveInvocation) -> tuple[ResourceRequirement, ...]:
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
            ResourceRequirement(
                ResourceKind.MEMORY_BANDWIDTH,
                invocation.work.memory_bytes,
                ResourceScope.PER_RANK,
            )
        )
    if invocation.work.message_bytes:
        resources.append(
            ResourceRequirement(
                ResourceKind.NETWORK,
                invocation.work.message_bytes,
                ResourceScope.PER_RANK,
                FrozenDict({"network_tier": invocation.network_tier}),
            )
        )
    return tuple(resources)


class PlanTransformerTrainingPass(DerivationPass[DistributedTaskIR, PortablePlanIR]):
    """Materialize exact WorkloadFacts without choosing a hardware target."""

    contract = PassContract.create(
        "transformer-plan-work-v1",
        DistributedTaskIR,
        PortablePlanIR,
        required_bindings=frozenset({BindingAxis.STRATEGY}),
    )

    def run(self, ir: DistributedTaskIR, context: PassContext) -> PortablePlanIR:
        model = ir.attributes.get("model_spec")
        execution = ir.attributes.get("execution_spec")
        if not isinstance(model, TransformerModelSpec) or not isinstance(execution, TransformerExecutionSpec):
            raise TypeError("distributed Transformer IR is missing typed semantic facts")
        strategy = context.session.bindings.strategy
        if strategy is None:
            raise ValueError("portable planning requires a strategy binding")

        input_id = BufferId.derive(ir.digest, "transformer-portable", "input")
        output_id = BufferId.derive(ir.digest, "transformer-portable", "output")
        boundary_elements = execution.microbatch_size * model.sequence_length * model.hidden_size
        if execution.tensor_parallel_communication is TensorParallelCommunication.REDUCE_SCATTER_ALL_GATHER:
            boundary_elements //= execution.tensor_parallel
        boundary_bytes = boundary_elements * execution.bytes_per_element

        task_ids = tuple(
            NodeId.derive(ir.digest, "transformer-portable", index, task.id) for index, task in enumerate(ir.tasks)
        )
        tasks = []
        for index, (source_task, task_id) in enumerate(zip(ir.tasks, task_ids)):
            invocation = source_task.attributes.get("invocation")
            if not isinstance(invocation, PrimitiveInvocation):
                raise TypeError("distributed task is missing a typed PrimitiveInvocation")
            if invocation.engine is EngineKind.MATRIX:
                capability = "matrix-multiply"
                alternatives = ("tensor-core", "matrix-engine")
            elif invocation.engine is EngineKind.VECTOR:
                capability = "vector-elementwise"
                alternatives = ("vector-engine",)
            else:
                capability = invocation.collective.value
                alternatives = ("collective-library", "network-engine")
            tasks.append(
                PlanTask(
                    id=task_id,
                    kind=(
                        PlanTaskKind.COLLECTIVE if invocation.engine is EngineKind.COLLECTIVE else PlanTaskKind.COMPUTE
                    ),
                    operation=source_task.operation,
                    dependencies=(task_ids[index - 1],) if index else (),
                    inputs=(input_id,) if index == 0 else (),
                    outputs=(output_id,) if index == len(ir.tasks) - 1 else (),
                    logical_ranks=source_task.ranks,
                    workload=WorkloadFacts(
                        operations=invocation.work.operations,
                        read_bytes=invocation.work.read_bytes,
                        write_bytes=invocation.work.write_bytes,
                        message_bytes=invocation.work.message_bytes,
                        attributes=FrozenDict(
                            {
                                "engine": invocation.engine.value,
                                "phase": invocation.phase.value,
                                "primitive": invocation.primitive,
                                "source_layer": invocation.source_layer,
                            }
                        ),
                    ),
                    lineage=Lineage.lowered("transformer-plan-work", (source_task.id,)),
                    resources=_plan_resources(invocation),
                    implementations=(ImplementationRequirement(capability, alternatives=alternatives),),
                    concurrency_group=("network" if invocation.engine is EngineKind.COLLECTIVE else "compute"),
                    attributes=FrozenDict({"invocation": invocation}),
                )
            )

        return PortablePlanIR(
            name=f"{model.name}-local-tp-block-plan",
            source_distributed_digest=ir.digest,
            strategy_fingerprint=strategy.fingerprint,
            planner_revision="transformer-work-analysis-v1",
            tasks=tuple(tasks),
            buffers=(
                PlanBuffer(
                    input_id,
                    boundary_bytes,
                    PlanBufferRole.INPUT,
                    AbstractStorageClass.DEVICE_LOCAL,
                    Lineage.lowered("transformer-plan-buffer", (ir.inputs[0],)),
                    consumers=(task_ids[0],),
                    alignment_bytes=16,
                ),
                PlanBuffer(
                    output_id,
                    boundary_bytes,
                    PlanBufferRole.OUTPUT,
                    AbstractStorageClass.DEVICE_LOCAL,
                    Lineage.lowered("transformer-plan-buffer", (ir.outputs[0],)),
                    producer=task_ids[-1],
                    alignment_bytes=16,
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
                    "execution_spec": execution,
                    "block_memory": ir.attributes["block_memory"],
                    "scope": ir.attributes["scope"],
                }
            ),
        )

"""Transformer training lowerings from semantic graph to portable work DAG."""

from __future__ import annotations

from typing import cast

from blueprinting.mapping import (
    DataParallel,
    ForwardOnly,
    InterleavedOneForwardOneBackward,
    OneForwardOneBackward,
    PipelineParallel,
    RecomputePolicy,
    SingleStage,
    TensorParallel,
    TensorParallelCommunication,
    TransformerTrainingMappingSpec,
    TransformerTrainingParallelism,
)
from blueprinting.workload import TransformerModelSpec, TransformerTrainingWorkloadSpec

from ...bindings import TrainingWorkload
from ...ids import BufferId, Lineage, NodeId, ValueId
from ...passes.authoring import RelationCheckContext, relation
from ...session import SynthesisSession
from ...stages.common import OperationName, TensorType
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
from .semantics import (
    TransformerModelOperationSemantic,
    TransformerTrainingDistributedTaskSemantic,
    TransformerTrainingPlanSemantic,
    TransformerTrainingProgramSemantic,
    TransformerTrainingStrategySemantic,
    TransformerTrainingWorkloadSemantic,
    training_task_semantic,
)
from .training import PrimitiveInvocation, TrainingPhase, derive_transformer_block


def _parallelism(
    mapping: TransformerTrainingMappingSpec,
) -> tuple[int, int, int, int, TensorParallelCommunication, RecomputePolicy]:
    """Destructure the Megatron-style TP × PP × DP strategy."""

    match mapping.parallelism:
        case TransformerTrainingParallelism(
            tensor=TensorParallel(degree=tp, communication=communication),
            pipeline=PipelineParallel(degree=pp, schedule=schedule),
            data=DataParallel(degree=dp),
            recompute=recompute,
        ):
            match schedule:
                case SingleStage() | OneForwardOneBackward():
                    virtual_stages = 1
                case InterleavedOneForwardOneBackward(virtual_stages=count):
                    virtual_stages = count
                case ForwardOnly():
                    raise TypeError("forward-only schedule is invalid for training")
            return tp, pp, dp, virtual_stages, communication, recompute
        case _:
            raise TypeError("unsupported Transformer training parallel strategy")


def _semantic_specs(
    ir: ModelIR,
    session: SynthesisSession,
) -> tuple[TransformerModelSpec, TransformerTrainingWorkloadSpec, TransformerTrainingMappingSpec]:
    if len(ir.operations) != 1 or ir.operations[0].operation != OperationName("transformer", "decoder_training"):
        raise ValueError("Transformer distribution expects one transformer.decoder_training operation")
    operation_semantic = ir.operations[0].semantic
    if not isinstance(operation_semantic, TransformerModelOperationSemantic):
        raise TypeError("model operation is missing a typed TransformerModelSpec")
    model = operation_semantic.model
    workload = session.bindings.workload
    strategy = session.bindings.strategy
    if workload is None or strategy is None:
        raise ValueError("Transformer distribution requires workload and strategy bindings")
    if not isinstance(workload.mode, TrainingWorkload):
        raise ValueError("Transformer training requires a training workload binding")
    workload_semantic = workload.semantic
    strategy_semantic = strategy.semantic
    if not isinstance(workload_semantic, TransformerTrainingWorkloadSemantic):
        raise TypeError("workload binding is missing a typed TransformerTrainingWorkloadSpec")
    if not isinstance(strategy_semantic, TransformerTrainingStrategySemantic):
        raise TypeError("strategy binding is missing a typed TransformerTrainingMappingSpec")
    workload_spec = workload_semantic.workload
    mapping = strategy_semantic.mapping
    mapping.validate_model(model)
    mapping.validate_workload(workload_spec)
    return model, workload_spec, mapping


def _verify_training_distributed_value(
    source: ModelValue,
    target: DistributedValue,
    context: RelationCheckContext,
) -> None:
    target_ir = context.target_ir
    if not isinstance(target_ir, DistributedTaskIR):
        raise TypeError("training value invariant requires DistributedTaskIR")
    if target.source_value != source.id or target.role is not source.role:
        raise ValueError("distributed value must retain its source identity and role")
    if target.type.dtype != source.type.dtype:
        raise ValueError("distributed value must preserve boundary datatype")
    if target.owners != tuple(range(target_ir.mesh.size)):
        raise ValueError("distributed value owners must cover the target logical mesh")


def _verify_training_distributed_task(
    _source: ModelOperation,
    target: DistributedTask,
    context: RelationCheckContext,
) -> None:
    target_ir = context.target_ir
    if not isinstance(target_ir, DistributedTaskIR):
        raise TypeError("training task invariant requires DistributedTaskIR")
    if target.ranks != tuple(range(target_ir.mesh.size)):
        raise ValueError("distributed task ranks must cover the logical mesh")
    semantic = target.semantic
    if not isinstance(semantic, TransformerTrainingDistributedTaskSemantic):
        raise TypeError("distributed task must retain its typed PrimitiveInvocation")
    invocation = semantic.invocation
    if invocation.engine is EngineKind.COLLECTIVE:
        if not isinstance(target.body, Collective) or invocation.collective is None:
            raise ValueError("collective invocation must lower to a collective body")
        if collective_kind(target.body.spec) is not invocation.collective:
            raise ValueError("collective body kind differs from its invocation")
        if target.body.spec.message_bytes != invocation.work.message_bytes:
            raise ValueError("collective body must conserve exact message bytes")
    elif not isinstance(target.body, LocalCompute):
        raise ValueError("non-collective invocation must lower to local compute")


def _verify_training_plan_buffer(
    source: DistributedValue,
    target: PlanBuffer,
    _context: RelationCheckContext,
) -> None:
    expected_role = {
        ValueRole.INPUT: PlanBufferRole.INPUT,
        ValueRole.OUTPUT: PlanBufferRole.OUTPUT,
    }.get(source.role)
    if expected_role is None or target.role is not expected_role:
        raise ValueError("training boundary buffer must preserve its distributed value role")
    if target.storage_class is not AbstractStorageClass.DEVICE_LOCAL:
        raise ValueError("training boundary buffers must remain abstract device-local storage")


def _verify_training_plan_task(
    source: DistributedTask,
    target: PlanTask,
    _context: RelationCheckContext,
) -> None:
    semantic = source.semantic
    if not isinstance(semantic, TransformerTrainingDistributedTaskSemantic):
        raise TypeError("plan relation source must retain a typed PrimitiveInvocation")
    invocation = semantic.invocation
    expected_work = WorkloadFacts(
        operations=invocation.work.operations,
        read_bytes=invocation.work.read_bytes,
        write_bytes=invocation.work.write_bytes,
        message_bytes=invocation.work.message_bytes,
    )
    if target.workload != expected_work:
        raise ValueError("portable task must conserve exact distributed workload facts")
    if target.operation != source.operation or target.logical_ranks != source.ranks or target.effects != source.effects:
        raise ValueError("portable task must preserve operation, logical ranks, and effects")
    if target.semantic != training_task_semantic(invocation):
        raise ValueError("portable task semantic projection differs from its invocation")
    if invocation.engine is EngineKind.COLLECTIVE:
        if not isinstance(target.body, CollectiveTask):
            raise ValueError("collective invocation must produce CollectiveTask")
    elif not isinstance(target.body, ComputeTask):
        raise ValueError("compute invocation must produce ComputeTask")


TRAINING_DISTRIBUTION_RULES = (
    relation(
        "transformer-distribute",
        "Map a model boundary value onto the logical tensor-parallel mesh",
        source=ModelValue,
        target=DistributedValue,
        verifier=_verify_training_distributed_value,
        introduces=("owners", "logical sharding", "mesh scope"),
        forbids=("physical device", "target implementation", "predicted time"),
    ),
    relation(
        "transformer-decompose",
        "Expand one semantic decoder-training operation into an ordered local/collective task DAG",
        source=ModelOperation,
        target=DistributedTask,
        verifier=_verify_training_distributed_task,
        introduces=("logical ranks", "task kind", "collective spec", "distributed dependencies"),
        forbids=("physical device", "queue", "kernel", "predicted time"),
    ),
)

TRAINING_PLANNING_RULES = (
    relation(
        "transformer-plan-buffer",
        "Materialize a distributed boundary value as an abstract plan buffer",
        source=DistributedValue,
        target=PlanBuffer,
        verifier=_verify_training_plan_buffer,
        introduces=("exact size", "storage class", "alignment", "producer/consumers"),
        forbids=("memory address", "physical memory region", "predicted time"),
    ),
    relation(
        "transformer-plan-work",
        "Materialize target-neutral exact work and capability requirements",
        source=DistributedTask,
        target=PlanTask,
        verifier=_verify_training_plan_task,
        introduces=("exact WorkloadFacts", "resource requirements", "implementation capabilities", "concurrency group"),
        forbids=("kernel ID", "physical queue", "empirical duration", "wall-clock timestamp"),
    ),
)


_TRAINING_STAGE = {
    TrainingPhase.FORWARD: 0,
    TrainingPhase.RECOMPUTE: 1,
    TrainingPhase.RECOMMUNICATION: 1,
    TrainingPhase.ACTIVATION_GRADIENT: 2,
    TrainingPhase.WEIGHT_GRADIENT: 2,
    TrainingPhase.OPTIMIZER: 3,
}


def _training_dependencies(
    invocations: tuple[PrimitiveInvocation, ...],
    task_ids: tuple[NodeId, ...],
) -> tuple[tuple[tuple[NodeId, ...], ...], int]:
    """Build a conservative phase-ordered DAG and identify the block output producer."""

    stages = tuple(_TRAINING_STAGE[item.phase] for item in invocations)
    if not stages or stages[0] != 0 or any(current < previous for previous, current in zip(stages, stages[1:])):
        raise ValueError("training invocations must be ordered by forward/recompute/backward/optimizer stage")
    forward_indices = tuple(index for index, item in enumerate(invocations) if item.phase is TrainingPhase.FORWARD)
    if not forward_indices:
        raise ValueError("training lowering requires at least one forward invocation")
    dependencies = tuple((task_ids[index - 1],) if index else () for index in range(len(task_ids)))
    return dependencies, forward_indices[-1]


def normalize_training_distribution(ir: ModelIR, session: SynthesisSession) -> DistributedTaskIR:
    """Purely derive logical distributed tasks for Transformer training."""
    model, workload, mapping = _semantic_specs(ir, session)
    invocations, block_memory = derive_transformer_block(model, workload, mapping)
    tp, _pp, _dp, _virtual_stages, communication, _recompute = _parallelism(mapping)
    ranks = tuple(range(tp))
    mesh = LogicalMesh("local-tensor-parallel-group", (MeshAxis("tp", tp),))
    source_input = ir.inputs[0]
    source_output = ir.outputs[0]
    input_id = ValueId.derive(ir.digest, "transformer-distributed", "input")
    output_id = ValueId.derive(ir.digest, "transformer-distributed", "output")
    tensor_type = TensorType(
        (workload.microbatch_size, model.sequence_length, model.hidden_size),
        workload.datatype,
    )
    match communication:
        case TensorParallelCommunication.REDUCE_SCATTER_ALL_GATHER:
            sharding = ShardingSpec(((), ("tp",), ()))
        case TensorParallelCommunication.ALL_REDUCE:
            sharding = ShardingSpec.replicated(tensor_type.rank, ("tp",))

    task_ids = tuple(
        NodeId.derive(ir.digest, "transformer-distributed", index, invocation.name)
        for index, invocation in enumerate(invocations)
    )
    dependencies, output_producer_index = _training_dependencies(invocations, task_ids)
    tasks = []
    for index, (task_id, invocation) in enumerate(zip(task_ids, invocations)):
        body: Collective | LocalCompute
        match invocation:
            case PrimitiveInvocation(engine=EngineKind.COLLECTIVE, collective=kind) if kind is not None:
                reduction = (
                    ReductionKind.SUM if kind in {CollectiveKind.ALL_REDUCE, CollectiveKind.REDUCE_SCATTER} else None
                )
                body = Collective(make_collective_spec(kind, ranks, invocation.work.message_bytes, reduction=reduction))
                operation = OperationName("collective", kind.value)
            case PrimitiveInvocation(engine=EngineKind.MATRIX | EngineKind.VECTOR, collective=None):
                body = LocalCompute()
                operation = OperationName("transformer", f"{invocation.primitive}_{invocation.phase.value}")
            case _:
                raise TypeError("unsupported Transformer training invocation")
        tasks.append(
            DistributedTask(
                id=task_id,
                body=body,
                operation=operation,
                ranks=ranks,
                inputs=(input_id,) if index == 0 else (),
                outputs=(output_id,) if index == output_producer_index else (),
                dependencies=dependencies[index],
                lineage=Lineage.lowered("transformer-decompose", (ir.operations[0].id,)),
                semantic=TransformerTrainingDistributedTaskSemantic(invocation),
            )
        )

    distributed = DistributedTaskIR(
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
        semantic=TransformerTrainingProgramSemantic(
            model,
            workload,
            mapping,
            "one-local-tensor-parallel-block",
            block_memory,
        ),
    )
    return distributed


def _plan_resources(invocation: PrimitiveInvocation) -> tuple[ResourceRequirement, ...]:
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
            )
        )
    return tuple(resources)


def normalize_training_plan(ir: DistributedTaskIR, session: SynthesisSession) -> PortablePlanIR:
    """Purely derive target-neutral exact work for Transformer training."""
    program = ir.semantic
    if not isinstance(program, TransformerTrainingProgramSemantic):
        raise TypeError("distributed Transformer IR is missing typed semantic facts")
    model = program.model
    workload = program.workload
    mapping = program.mapping
    tp, _pp, _dp, _virtual_stages, communication, _recompute = _parallelism(mapping)
    block_memory = program.block_memory
    strategy = session.bindings.strategy
    if strategy is None:
        raise ValueError("portable planning requires a strategy binding")
    if not isinstance(strategy.semantic, TransformerTrainingStrategySemantic):
        raise TypeError("strategy binding is missing typed Transformer training semantics")
    if strategy.semantic.mapping != mapping:
        raise ValueError("strategy binding does not match the distributed program semantics")

    input_id = BufferId.derive(ir.digest, "transformer-portable", "input")
    output_id = BufferId.derive(ir.digest, "transformer-portable", "output")
    boundary_elements = workload.microbatch_size * model.sequence_length * model.hidden_size
    if communication is TensorParallelCommunication.REDUCE_SCATTER_ALL_GATHER:
        boundary_elements //= tp
    boundary_bytes = boundary_elements * workload.bytes_per_element

    task_ids = tuple(
        NodeId.derive(ir.digest, "transformer-portable", index, task.id) for index, task in enumerate(ir.tasks)
    )
    task_semantics = tuple(task.semantic for task in ir.tasks)
    if any(not isinstance(item, TransformerTrainingDistributedTaskSemantic) for item in task_semantics):
        raise TypeError("distributed task is missing a typed PrimitiveInvocation")
    typed_semantics = cast(tuple[TransformerTrainingDistributedTaskSemantic, ...], task_semantics)
    typed_invocations = tuple(item.invocation for item in typed_semantics)
    dependencies, output_producer_index = _training_dependencies(typed_invocations, task_ids)
    tasks = []
    for index, (source_task, task_id, invocation) in enumerate(zip(ir.tasks, task_ids, typed_invocations)):
        alternatives: tuple[str, ...]
        match invocation:
            case PrimitiveInvocation(engine=EngineKind.MATRIX):
                capability, alternatives = "matrix-multiply", ("tensor-core", "matrix-engine")
            case PrimitiveInvocation(engine=EngineKind.VECTOR):
                capability, alternatives = "vector-elementwise", ("vector-engine",)
            case PrimitiveInvocation(engine=EngineKind.COLLECTIVE, collective=kind) if kind is not None:
                capability, alternatives = kind.value, ("collective-library", "network-engine")
            case _:
                raise TypeError("unsupported Transformer training invocation")
        tasks.append(
            PlanTask(
                id=task_id,
                body=CollectiveTask() if invocation.engine is EngineKind.COLLECTIVE else ComputeTask(),
                operation=source_task.operation,
                dependencies=dependencies[index],
                inputs=(input_id,) if index == 0 else (),
                outputs=(output_id,) if index == output_producer_index else (),
                logical_ranks=source_task.ranks,
                workload=WorkloadFacts(
                    operations=invocation.work.operations,
                    read_bytes=invocation.work.read_bytes,
                    write_bytes=invocation.work.write_bytes,
                    message_bytes=invocation.work.message_bytes,
                ),
                lineage=Lineage.lowered("transformer-plan-work", (source_task.id,)),
                resources=_plan_resources(invocation),
                implementations=(ImplementationRequirement(capability, alternatives=alternatives),),
                concurrency_group=("network" if invocation.engine is EngineKind.COLLECTIVE else "compute"),
                semantic=training_task_semantic(invocation),
            )
        )

    return PortablePlanIR(
        name=f"{model.name}-local-tp-block-plan",
        source_distributed_digest=ir.digest,
        strategy_fingerprint=strategy.fingerprint,
        planner_revision="transformer-work-analysis",
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
                producer=task_ids[output_producer_index],
                alignment_bytes=16,
            ),
        ),
        inputs=(input_id,),
        outputs=(output_id,),
        objectives=(
            PlanObjective(ObjectiveKind.LATENCY, ObjectiveDirection.MINIMIZE),
            PlanObjective(ObjectiveKind.PEAK_MEMORY, ObjectiveDirection.MINIMIZE),
        ),
        semantic=TransformerTrainingPlanSemantic(
            model,
            workload,
            mapping,
            program.scope,
            block_memory,
        ),
    )

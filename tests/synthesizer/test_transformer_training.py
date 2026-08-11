from __future__ import annotations

from dataclasses import replace

import pytest

from blueprinting.mapping import (
    DataParallel,
    PipelineParallel,
    RecomputePolicy,
    SingleStage,
    TensorParallel,
    TensorParallelCommunication,
    TransformerTrainingMappingSpec,
    TransformerTrainingParallelism,
)
from blueprinting.synthesizer.dialects.transformer import (
    TrainingPhase,
    TransformerTrainingDistributedTaskSemantic,
    TransformerTrainingPlanTaskSemantic,
    derive_transformer_block,
)
from blueprinting.synthesizer.errors import PassContractError
from blueprinting.synthesizer.frontend import build_transformer_model_ir, synthesis_session_for
from blueprinting.synthesizer.ids import Lineage, NodeId
from blueprinting.synthesizer.passes import (
    DeterminismPolicy,
    FunctionPass,
    PassManager,
    PassPipeline,
    TransitionVerifier,
)
from blueprinting.synthesizer.stages.distributed.ir import DistributedTaskIR
from blueprinting.synthesizer.stages.distributed.passes import DistributeTransformerTrainingPass
from blueprinting.synthesizer.stages.portable_plan.ir import PortablePlanIR
from blueprinting.synthesizer.stages.portable_plan.passes import PlanTransformerTrainingPass
from blueprinting.workload import TransformerModelSpec, TransformerTrainingWorkloadSpec


def _model(*, sequence_length: int = 32) -> TransformerModelSpec:
    return TransformerModelSpec(
        name="fixture-training",
        hidden_size=64,
        feedforward_size=256,
        sequence_length=sequence_length,
        attention_heads=8,
        attention_head_size=8,
        block_count=4,
    )


def _workload() -> TransformerTrainingWorkloadSpec:
    return TransformerTrainingWorkloadSpec(global_batch_size=8, microbatch_size=1, datatype="float16")


def _mapping(
    *,
    tensor_parallel: int = 2,
    communication: TensorParallelCommunication = TensorParallelCommunication.ALL_REDUCE,
) -> TransformerTrainingMappingSpec:
    return TransformerTrainingMappingSpec(
        TransformerTrainingParallelism(
            TensorParallel(tensor_parallel, communication),
            PipelineParallel(1, SingleStage()),
            DataParallel(1),
            RecomputePolicy.ATTENTION,
        )
    )


def _derive():
    model = _model()
    workload = _workload()
    mapping = _mapping()
    source = build_transformer_model_ir(model)
    session = synthesis_session_for(model, workload, mapping)
    result = PassManager(determinism=DeterminismPolicy.VERIFY).require_run(
        PassPipeline.of(DistributeTransformerTrainingPass(), PlanTransformerTrainingPass()),
        source,
        session=session,
    )
    return source, result, session


def test_training_output_is_produced_by_forward_not_optimizer() -> None:
    _, result, _ = _derive()
    distributed = result.checkpoints[0].ir
    plan = result.ir

    assert isinstance(distributed, DistributedTaskIR)
    assert isinstance(plan, PortablePlanIR)
    distributed_producer = next(task for task in distributed.tasks if distributed.outputs[0] in task.outputs)
    plan_output = next(buffer for buffer in plan.buffers if buffer.id == plan.outputs[0])
    plan_producer = next(task for task in plan.tasks if task.id == plan_output.producer)

    assert isinstance(distributed_producer.semantic, TransformerTrainingDistributedTaskSemantic)
    assert distributed_producer.semantic.invocation.phase is TrainingPhase.FORWARD
    assert isinstance(plan_producer.semantic, TransformerTrainingPlanTaskSemantic)
    assert plan_producer.semantic.phase is TrainingPhase.FORWARD
    assert plan.outputs[0] in plan_producer.outputs
    assert isinstance(plan.tasks[-1].semantic, TransformerTrainingPlanTaskSemantic)
    assert plan.tasks[-1].semantic.phase is TrainingPhase.OPTIMIZER
    assert plan.outputs[0] not in plan.tasks[-1].outputs


def test_training_phases_are_conservatively_ordered_and_round_trip_with_lineage() -> None:
    source, result, _ = _derive()
    distributed = result.checkpoints[0].ir
    plan = result.ir
    stage = {
        TrainingPhase.FORWARD.value: 0,
        TrainingPhase.RECOMPUTE.value: 1,
        TrainingPhase.RECOMMUNICATION.value: 1,
        TrainingPhase.ACTIVATION_GRADIENT.value: 2,
        TrainingPhase.WEIGHT_GRADIENT.value: 2,
        TrainingPhase.OPTIMIZER.value: 3,
    }
    stages = tuple(stage[task.semantic.phase.value] for task in plan.tasks)

    assert stages == tuple(sorted(stages))
    assert all(task.dependencies == (plan.tasks[index - 1].id,) for index, task in enumerate(plan.tasks[1:], 1))
    assert DistributedTaskIR.from_json(distributed.to_json()).or_raise() == distributed
    assert PortablePlanIR.from_json(plan.to_json()).or_raise() == plan
    assert source.digest in distributed.header.parent_digests
    assert distributed.digest in plan.header.parent_digests
    assert all(task.lineage.sources for task in distributed.tasks)
    assert all(task.lineage.sources for task in plan.tasks)


def test_sequence_parallel_requires_divisible_sequence_dimension() -> None:
    with pytest.raises(ValueError, match="sequence_length must be divisible by tensor_parallel"):
        synthesis_session_for(
            _model(sequence_length=30),
            _workload(),
            _mapping(tensor_parallel=4, communication=TensorParallelCommunication.REDUCE_SCATTER_ALL_GATHER),
        )


def test_tp1_communication_policies_have_identical_work_and_memory() -> None:
    model = _model()
    workload = _workload()
    all_reduce = _mapping(tensor_parallel=1, communication=TensorParallelCommunication.ALL_REDUCE)
    sequence_parallel = _mapping(
        tensor_parallel=1,
        communication=TensorParallelCommunication.REDUCE_SCATTER_ALL_GATHER,
    )

    assert derive_transformer_block(model, workload, all_reduce) == derive_transformer_block(
        model,
        workload,
        sequence_parallel,
    )


def test_strategy_binding_has_one_typed_source_for_mapping_facts() -> None:
    model = _model()
    workload = _workload()
    mapping = _mapping()
    source = build_transformer_model_ir(model)
    session = synthesis_session_for(model, workload, mapping)
    assert session.bindings.strategy is not None
    assert session.bindings.strategy.semantic.mapping == mapping
    assert not hasattr(session.bindings.strategy, "recompute_policy")
    with pytest.raises(TypeError, match="recompute_policy"):
        replace(session.bindings.strategy, recompute_policy=RecomputePolicy.NONE.value)

    distributed = PassManager().require_run(
        PassPipeline.of(DistributeTransformerTrainingPass()),
        source,
        session=session,
    )
    assert distributed.ir.verify().is_ok


def test_transition_gate_rejects_undeclared_cross_boundary_lineage_before_commit() -> None:
    model = _model()
    workload = _workload()
    mapping = _mapping()
    source = build_transformer_model_ir(model)
    session = synthesis_session_for(model, workload, mapping)
    distributed = (
        PassManager()
        .require_run(
            PassPipeline.of(DistributeTransformerTrainingPass()),
            source,
            session=session,
        )
        .ir
    )
    broken_task = replace(
        distributed.tasks[0],
        lineage=Lineage.lowered("undeclared-transform", (NodeId.derive("missing-source"),)),
    )
    broken = replace(distributed, tasks=(broken_task,) + distributed.tasks[1:])
    derivation_pass = FunctionPass(DistributeTransformerTrainingPass.contract, lambda _ir, _context: broken)

    with pytest.raises(PassContractError, match="undeclared lineage transform"):
        PassManager().require_run(PassPipeline.of(derivation_pass), source, session=session)


def test_transition_claim_rejects_distributed_shape_mutation() -> None:
    source, result, session = _derive()
    distributed = result.checkpoints[0].ir
    value = distributed.values[0]
    broken = replace(
        distributed,
        values=(replace(value, type=replace(value.type, shape=(999, *value.type.shape[1:]))), *distributed.values[1:]),
    )

    with pytest.raises(PassContractError, match="canonical normal form"):
        TransitionVerifier.verify(source, broken, DistributeTransformerTrainingPass.contract, session)


def test_transition_claim_rejects_plan_buffer_size_mutation() -> None:
    _, result, session = _derive()
    distributed = result.checkpoints[0].ir
    plan = result.ir
    buffer = plan.buffers[0]
    broken = replace(plan, buffers=(replace(buffer, size_bytes=buffer.size_bytes + 2), *plan.buffers[1:]))

    with pytest.raises(PassContractError, match="canonical normal form"):
        TransitionVerifier.verify(distributed, broken, PlanTransformerTrainingPass.contract, session)


def test_transition_claim_rejects_same_cardinality_dependency_rewire() -> None:
    _, result, session = _derive()
    distributed = result.checkpoints[0].ir
    plan = result.ir
    task = plan.tasks[2]
    assert len(task.dependencies) == 1 and task.dependencies != (plan.tasks[0].id,)
    broken = replace(plan, tasks=(*plan.tasks[:2], replace(task, dependencies=(plan.tasks[0].id,)), *plan.tasks[3:]))
    broken.require_valid()

    with pytest.raises(PassContractError, match="canonical normal form"):
        TransitionVerifier.verify(distributed, broken, PlanTransformerTrainingPass.contract, session)


def test_normal_form_rejects_semantic_payload_substitution() -> None:
    source, result, session = _derive()
    distributed = result.checkpoints[0].ir
    plan = result.ir

    broken_distributed = replace(
        distributed,
        tasks=(replace(distributed.tasks[0], semantic=distributed.tasks[1].semantic), *distributed.tasks[1:]),
    )
    broken_distributed.require_valid()
    with pytest.raises(PassContractError, match="canonical normal form"):
        TransitionVerifier.verify(
            source,
            broken_distributed,
            DistributeTransformerTrainingPass.contract,
            session,
        )

    broken_plan = replace(
        plan,
        tasks=(replace(plan.tasks[0], semantic=plan.tasks[1].semantic), *plan.tasks[1:]),
    )
    broken_plan.require_valid()
    with pytest.raises(PassContractError, match="canonical normal form"):
        TransitionVerifier.verify(
            distributed,
            broken_plan,
            PlanTransformerTrainingPass.contract,
            session,
        )

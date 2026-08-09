from __future__ import annotations

from dataclasses import replace

import pytest

from blueprinting.mapping import (
    RecomputePolicy,
    TensorParallelCommunication,
    TransformerTrainingMappingSpec,
)
from blueprinting.synthesizer.dialects.transformer import TrainingPhase, derive_transformer_block
from blueprinting.synthesizer.errors import PassExecutionError
from blueprinting.synthesizer.frontend import build_transformer_model_ir, synthesis_session_for
from blueprinting.synthesizer.ir import DistributedTaskIR, PortablePlanIR
from blueprinting.synthesizer.lowering import DistributeTransformerTrainingPass, PlanTransformerTrainingPass
from blueprinting.synthesizer.passes import PassManager, PassPipeline
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
        tensor_parallel=tensor_parallel,
        pipeline_parallel=1,
        data_parallel=1,
        recompute=RecomputePolicy.ATTENTION,
        pipeline_interleaving=1,
        optimizer_sharding=False,
        tensor_parallel_communication=communication,
    )


def _derive():
    model = _model()
    workload = _workload()
    mapping = _mapping()
    source = build_transformer_model_ir(model)
    result = PassManager().run(
        PassPipeline.of(DistributeTransformerTrainingPass(), PlanTransformerTrainingPass()),
        source,
        session=synthesis_session_for(model, workload, mapping),
    )
    return source, result


def test_training_output_is_produced_by_forward_not_optimizer() -> None:
    _, result = _derive()
    distributed = result.checkpoints[0].ir
    plan = result.ir

    assert isinstance(distributed, DistributedTaskIR)
    assert isinstance(plan, PortablePlanIR)
    distributed_producer = next(task for task in distributed.tasks if distributed.outputs[0] in task.outputs)
    plan_output = next(buffer for buffer in plan.buffers if buffer.id == plan.outputs[0])
    plan_producer = next(task for task in plan.tasks if task.id == plan_output.producer)

    assert distributed_producer.attributes["invocation"].phase is TrainingPhase.FORWARD
    assert plan_producer.workload.attributes["phase"] == TrainingPhase.FORWARD.value
    assert plan.outputs[0] in plan_producer.outputs
    assert plan.tasks[-1].workload.attributes["phase"] == TrainingPhase.OPTIMIZER.value
    assert plan.outputs[0] not in plan.tasks[-1].outputs


def test_training_phases_are_conservatively_ordered_and_round_trip_with_lineage() -> None:
    source, result = _derive()
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
    stages = tuple(stage[task.workload.attributes["phase"]] for task in plan.tasks)

    assert stages == tuple(sorted(stages))
    assert all(task.dependencies == (plan.tasks[index - 1].id,) for index, task in enumerate(plan.tasks[1:], 1))
    assert DistributedTaskIR.from_json(distributed.to_json()) == distributed
    assert PortablePlanIR.from_json(plan.to_json()) == plan
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


def test_lowering_rejects_strategy_policy_that_disagrees_with_typed_mapping() -> None:
    model = _model()
    workload = _workload()
    mapping = _mapping()
    source = build_transformer_model_ir(model)
    session = synthesis_session_for(model, workload, mapping)
    assert session.bindings.strategy is not None
    mismatched_strategy = replace(session.bindings.strategy, recompute_policy=RecomputePolicy.NONE.value)
    session = replace(session, bindings=session.bindings.with_binding(mismatched_strategy))

    with pytest.raises(PassExecutionError, match="strategy binding is inconsistent"):
        PassManager().run(
            PassPipeline.of(DistributeTransformerTrainingPass()),
            source,
            session=session,
        )

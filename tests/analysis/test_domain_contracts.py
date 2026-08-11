from __future__ import annotations

import pytest

from blueprinting.mapping import (
    DataParallel,
    ForwardOnly,
    InterleavedOneForwardOneBackward,
    NetworkTierBinding,
    PipelineParallel,
    ReplicaParallel,
    TensorParallel,
    TensorParallelCommunication,
    TransformerInferenceMappingSpec,
    TransformerInferenceParallelism,
    TransformerTrainingMappingSpec,
    TransformerTrainingParallelism,
)
from blueprinting.workload import TransformerModelSpec, TransformerTrainingWorkloadSpec


def test_legacy_aliases_may_match_but_cannot_define_conflicting_truth() -> None:
    training = {
        "tensor_parallel": 2,
        "tensor_par": 2,
        "pipeline_parallel": 1,
        "pipeline_par": 1,
        "data_parallel": 2,
        "data_par": 2,
        "recompute": "none",
        "activation_recompute": "none",
        "pipeline_interleaving": 1,
        "optimizer_sharding": False,
        "tensor_parallel_communication": "ar",
        "tensor_par_comm_type": "ar",
    }

    assert TransformerTrainingMappingSpec.from_mapping(training).world_size == 4
    with pytest.raises(ValueError, match="tensor_parallel conflicts"):
        TransformerTrainingMappingSpec.from_mapping({**training, "tensor_par": 4})

    assert (
        TransformerInferenceMappingSpec.from_mapping(
            {"tensor_parallel": 2, "pipeline_parallel": 1, "replicas": 2, "data_par": 2}
        ).world_size
        == 4
    )
    with pytest.raises(ValueError, match="replicas conflicts"):
        TransformerInferenceMappingSpec.from_mapping(
            {"tensor_parallel": 2, "pipeline_parallel": 1, "replicas": 2, "data_par": 4}
        )


def test_workload_and_network_aliases_reject_conflicts() -> None:
    assert (
        TransformerTrainingWorkloadSpec.from_mapping(
            {"global_batch_size": 8, "batch_size": 8, "microbatch_size": 1, "datatype": "float16"}
        ).global_batch_size
        == 8
    )
    with pytest.raises(ValueError, match="global_batch_size conflicts"):
        TransformerTrainingWorkloadSpec.from_mapping(
            {"global_batch_size": 8, "batch_size": 16, "microbatch_size": 1, "datatype": "float16"}
        )

    assert NetworkTierBinding.from_mapping({"tensor_parallel_network": 1, "tensor_par_net": 1}).tensor_parallel == 1
    with pytest.raises(ValueError, match="tensor_parallel_network conflicts"):
        NetworkTierBinding.from_mapping({"tensor_parallel_network": 1, "tensor_par_net": 0})


def test_megatron_parallel_axes_are_exposed_as_pattern_matchable_values() -> None:
    mapping = TransformerTrainingMappingSpec.from_mapping(
        {
            "tensor_parallel": 4,
            "pipeline_parallel": 8,
            "data_parallel": 2,
            "recompute": "full",
            "pipeline_interleaving": 3,
            "optimizer_sharding": True,
            "tensor_parallel_communication": "rs_ag",
        }
    )

    match mapping.parallelism:
        case TransformerTrainingParallelism(
            TensorParallel(4, TensorParallelCommunication.REDUCE_SCATTER_ALL_GATHER),
            PipelineParallel(8, InterleavedOneForwardOneBackward(3)),
            DataParallel(2, True),
            recompute,
        ):
            assert recompute.value == "full"
        case unexpected:  # pragma: no cover - diagnostic if the ADT shape regresses
            pytest.fail(f"unexpected training strategy: {unexpected!r}")

    inference = TransformerInferenceMappingSpec(
        TransformerInferenceParallelism(
            TensorParallel(4, TensorParallelCommunication.ALL_REDUCE),
            PipelineParallel(2, ForwardOnly()),
            ReplicaParallel(3),
        )
    )
    match inference.parallelism:
        case TransformerInferenceParallelism(
            TensorParallel(4),
            PipelineParallel(2, ForwardOnly()),
            ReplicaParallel(3),
        ):
            pass
        case unexpected:  # pragma: no cover
            pytest.fail(f"unexpected inference strategy: {unexpected!r}")


def test_pipeline_strategy_rejects_unrepresentable_static_partitions() -> None:
    mapping = TransformerTrainingMappingSpec.from_mapping(
        {
            "tensor_parallel": 1,
            "pipeline_parallel": 4,
            "data_parallel": 1,
            "recompute": "none",
            "pipeline_interleaving": 2,
            "optimizer_sharding": False,
            "tensor_parallel_communication": "ar",
        }
    )
    model = TransformerModelSpec(
        "pipeline-fixture",
        hidden_size=64,
        feedforward_size=256,
        sequence_length=128,
        attention_heads=8,
        attention_head_size=8,
        block_count=12,
    )

    with pytest.raises(ValueError, match="interleaving must divide"):
        mapping.validate_model(model)

from __future__ import annotations

import pytest

from blueprinting.mapping import (
    NetworkTierBinding,
    TransformerInferenceMappingSpec,
    TransformerTrainingMappingSpec,
)
from blueprinting.workload import TransformerTrainingWorkloadSpec


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

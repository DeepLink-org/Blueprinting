"""Target-neutral Transformer strategy contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from blueprinting.schema.codec import enum_type, record_type
from blueprinting.workload import TransformerModelSpec, TransformerTrainingWorkloadSpec


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


_MISSING = object()


def _field(data: Mapping[str, Any], canonical: str, legacy: str, default: Any = _MISSING) -> Any:
    if canonical in data and legacy in data and data[canonical] != data[legacy]:
        raise ValueError(f"{canonical} conflicts with legacy alias {legacy}")
    if canonical in data:
        return data[canonical]
    if legacy in data:
        return data[legacy]
    if default is _MISSING:
        raise KeyError(canonical)
    return default


@enum_type("compiler.transformer.recompute_policy")
class RecomputePolicy(Enum):
    NONE = "none"
    ATTENTION = "attn_only"
    FULL = "full"


@enum_type("compiler.transformer.tp_communication")
class TensorParallelCommunication(Enum):
    ALL_REDUCE = "ar"
    REDUCE_SCATTER_ALL_GATHER = "rs_ag"


@record_type("blueprinting.mapping.transformer-training.v1")
@dataclass(frozen=True)
class TransformerTrainingMappingSpec:
    """Target-neutral parallel and recomputation strategy for training."""

    tensor_parallel: int
    pipeline_parallel: int
    data_parallel: int
    recompute: RecomputePolicy
    pipeline_interleaving: int
    optimizer_sharding: bool
    tensor_parallel_communication: TensorParallelCommunication
    fused_activation: bool = False
    sequence_parallel_all_gather_redo: bool = False

    def __post_init__(self) -> None:
        for name in ("tensor_parallel", "pipeline_parallel", "data_parallel", "pipeline_interleaving"):
            _positive_integer(getattr(self, name), name)
        if not isinstance(self.recompute, RecomputePolicy):
            raise TypeError("recompute must be a RecomputePolicy")
        if not isinstance(self.tensor_parallel_communication, TensorParallelCommunication):
            raise TypeError("tensor_parallel_communication must be TensorParallelCommunication")
        if self.optimizer_sharding and self.data_parallel == 1:
            raise ValueError("optimizer sharding requires data_parallel > 1")

    @property
    def world_size(self) -> int:
        return self.tensor_parallel * self.pipeline_parallel * self.data_parallel

    def validate_workload(self, workload: TransformerTrainingWorkloadSpec) -> None:
        if not isinstance(workload, TransformerTrainingWorkloadSpec):
            raise TypeError("workload must be TransformerTrainingWorkloadSpec")
        if workload.global_batch_size % self.data_parallel:
            raise ValueError("global_batch_size must be divisible by data_parallel")
        if self.local_batch_size(workload) % workload.microbatch_size:
            raise ValueError("local batch size must be divisible by microbatch_size")

    def validate_model(self, model: TransformerModelSpec) -> None:
        if not isinstance(model, TransformerModelSpec):
            raise TypeError("model must be TransformerModelSpec")
        divisibility = {
            "hidden_size": model.hidden_size,
            "feedforward_size": model.feedforward_size,
            "attention_heads": model.attention_heads,
        }
        if self.tensor_parallel_communication is TensorParallelCommunication.REDUCE_SCATTER_ALL_GATHER:
            divisibility["sequence_length"] = model.sequence_length
        for name, value in divisibility.items():
            if value % self.tensor_parallel:
                raise ValueError(f"{name} must be divisible by tensor_parallel")

    def local_batch_size(self, workload: TransformerTrainingWorkloadSpec) -> int:
        return workload.global_batch_size // self.data_parallel

    def microbatch_count(self, workload: TransformerTrainingWorkloadSpec) -> int:
        self.validate_workload(workload)
        return self.local_batch_size(workload) // workload.microbatch_size

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> TransformerTrainingMappingSpec:
        tensor_parallel = _field(data, "tensor_parallel", "tensor_par")
        pipeline_parallel = _field(data, "pipeline_parallel", "pipeline_par")
        data_parallel = _field(data, "data_parallel", "data_par")
        expected_world_size = tensor_parallel * pipeline_parallel * data_parallel
        if data.get("num_procs", expected_world_size) != expected_world_size:
            raise ValueError("num_procs must equal tensor_parallel * pipeline_parallel * data_parallel")
        if not data.get("training", True):
            raise ValueError("this mapping describes training only")
        if data.get("attention_type", "multihead") != "multihead":
            raise ValueError("the current Transformer dialect supports multihead attention only")
        if data.get("tensor_par_overlap", "none") != "none" or data.get("data_par_overlap", False):
            raise ValueError("overlap requires concrete scheduling and is outside this mapping contract")
        if any(data.get(name, False) for name in ("weight_offload", "activations_offload", "optimizer_offload")):
            raise ValueError("offload strategies are outside this mapping contract")
        return cls(
            tensor_parallel=tensor_parallel,
            pipeline_parallel=pipeline_parallel,
            data_parallel=data_parallel,
            recompute=RecomputePolicy(_field(data, "recompute", "activation_recompute")),
            pipeline_interleaving=data["pipeline_interleaving"],
            optimizer_sharding=data["optimizer_sharding"],
            tensor_parallel_communication=TensorParallelCommunication(
                _field(data, "tensor_parallel_communication", "tensor_par_comm_type")
            ),
            fused_activation=data.get("fused_activation", False),
            sequence_parallel_all_gather_redo=data.get("seq_par_ag_redo", False),
        )


@record_type("blueprinting.mapping.transformer-inference.v1")
@dataclass(frozen=True)
class TransformerInferenceMappingSpec:
    """Target-neutral logical mapping for an inference replica."""

    tensor_parallel: int
    pipeline_parallel: int
    replicas: int

    def __post_init__(self) -> None:
        for name in ("tensor_parallel", "pipeline_parallel", "replicas"):
            _positive_integer(getattr(self, name), name)

    @property
    def world_size(self) -> int:
        return self.tensor_parallel * self.pipeline_parallel * self.replicas

    def validate_model(self, model: TransformerModelSpec) -> None:
        divisibility = {
            "hidden_size": model.hidden_size,
            "feedforward_size": model.feedforward_size,
            "attention_heads": model.attention_heads,
        }
        for name, value in divisibility.items():
            if value % self.tensor_parallel:
                raise ValueError(f"{name} must be divisible by tensor_parallel")
        if self.pipeline_parallel > model.block_count:
            raise ValueError("pipeline_parallel cannot exceed block_count")
        if model.block_count % self.pipeline_parallel:
            raise ValueError("pipeline_parallel must divide block_count for static inference planning")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> TransformerInferenceMappingSpec:
        replicas = _field(data, "replicas", "data_par", 1)
        tensor_parallel = _field(data, "tensor_parallel", "tensor_par")
        pipeline_parallel = _field(data, "pipeline_parallel", "pipeline_par")
        expected_world_size = tensor_parallel * pipeline_parallel * replicas
        if data.get("num_procs", expected_world_size) != expected_world_size:
            raise ValueError("num_procs must equal tensor_parallel * pipeline_parallel * replicas")
        return cls(
            tensor_parallel=tensor_parallel,
            pipeline_parallel=pipeline_parallel,
            replicas=replicas,
        )

"""Target-neutral Transformer strategy contracts."""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from typing import Any

from typing_extensions import assert_never

from blueprinting.schema.authoring import (
    AtLeastTwoInt,
    PositiveInt,
    adt,
    enum,
    record,
    seal_adt,
    variant,
)
from blueprinting.workload import TransformerModelSpec, TransformerTrainingWorkloadSpec

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


@enum("blueprinting.mapping.recompute-policy")
class RecomputePolicy(Enum):
    NONE = "none"
    ATTENTION = "attn_only"
    FULL = "full"


@enum("blueprinting.mapping.tensor-parallel-communication")
class TensorParallelCommunication(Enum):
    ALL_REDUCE = "ar"
    REDUCE_SCATTER_ALL_GATHER = "rs_ag"


@adt(wire="blueprinting.mapping.pipeline-schedule")
class PipelineSchedule:
    """Closed family of pipeline execution schedules."""


@variant("single-stage")
class SingleStage(PipelineSchedule):
    """No inter-stage pipeline because the pipeline degree is one."""


@variant("one-f-one-b")
class OneForwardOneBackward(PipelineSchedule):
    """Non-interleaved 1F1B training schedule."""


@variant("interleaved-one-f-one-b")
class InterleavedOneForwardOneBackward(PipelineSchedule):
    """Megatron-style interleaved 1F1B with virtual pipeline stages."""

    virtual_stages: AtLeastTwoInt


@variant("forward-only")
class ForwardOnly(PipelineSchedule):
    """Static inference pipeline with no backward wave."""


PipelineScheduleVariant = SingleStage | OneForwardOneBackward | InterleavedOneForwardOneBackward | ForwardOnly
seal_adt(PipelineSchedule, PipelineScheduleVariant)


@record("blueprinting.mapping.tensor-parallel")
class TensorParallel:
    degree: PositiveInt
    communication: TensorParallelCommunication

    @property
    def sequence_parallel(self) -> bool:
        return self.communication is TensorParallelCommunication.REDUCE_SCATTER_ALL_GATHER


@record("blueprinting.mapping.pipeline-parallel")
class PipelineParallel:
    degree: PositiveInt
    schedule: PipelineScheduleVariant

    def __post_init__(self) -> None:
        if self.degree == 1 and not isinstance(self.schedule, SingleStage):
            raise ValueError("pipeline degree one requires SingleStage")
        if self.degree > 1 and isinstance(self.schedule, SingleStage):
            raise ValueError("pipeline degree greater than one requires a pipeline schedule")


@record("blueprinting.mapping.data-parallel")
class DataParallel:
    degree: PositiveInt
    optimizer_sharding: bool = False

    def __post_init__(self) -> None:
        if self.optimizer_sharding and self.degree == 1:
            raise ValueError("optimizer sharding requires data parallel degree greater than one")


@record("blueprinting.mapping.replica-parallel")
class ReplicaParallel:
    degree: PositiveInt


@record("blueprinting.mapping.transformer-training-strategy")
class TransformerTrainingParallelism:
    tensor: TensorParallel
    pipeline: PipelineParallel
    data: DataParallel
    recompute: RecomputePolicy

    def __post_init__(self) -> None:
        if isinstance(self.pipeline.schedule, ForwardOnly):
            raise ValueError("forward-only schedule is invalid for training")


@record("blueprinting.mapping.transformer-inference-strategy")
class TransformerInferenceParallelism:
    tensor: TensorParallel
    pipeline: PipelineParallel
    replicas: ReplicaParallel

    def __post_init__(self) -> None:
        if not isinstance(self.pipeline.schedule, (SingleStage, ForwardOnly)):
            raise ValueError("inference requires a single-stage or forward-only schedule")


def _pipeline_interleaving(schedule: PipelineScheduleVariant) -> int:
    match schedule:
        case InterleavedOneForwardOneBackward(virtual_stages):
            return virtual_stages
        case SingleStage() | OneForwardOneBackward() | ForwardOnly():
            return 1
    assert_never(schedule)


def _training_schedule(pipeline_degree: int, virtual_stages: int) -> PipelineScheduleVariant:
    if pipeline_degree == 1:
        if virtual_stages != 1:
            raise ValueError("pipeline interleaving requires pipeline_parallel > 1")
        return SingleStage()
    if virtual_stages == 1:
        return OneForwardOneBackward()
    return InterleavedOneForwardOneBackward(virtual_stages)


@record("blueprinting.mapping.transformer-training")
class TransformerTrainingMappingSpec:
    """Target-neutral parallel and recomputation strategy for training."""

    parallelism: TransformerTrainingParallelism
    fused_activation: bool = False
    sequence_parallel_all_gather_redo: bool = False

    @property
    def world_size(self) -> int:
        return self.tensor_parallel * self.pipeline_parallel * self.data_parallel

    @property
    def tensor_parallel(self) -> int:
        return self.parallelism.tensor.degree

    @property
    def pipeline_parallel(self) -> int:
        return self.parallelism.pipeline.degree

    @property
    def data_parallel(self) -> int:
        return self.parallelism.data.degree

    @property
    def recompute(self) -> RecomputePolicy:
        return self.parallelism.recompute

    @property
    def pipeline_interleaving(self) -> int:
        return _pipeline_interleaving(self.parallelism.pipeline.schedule)

    @property
    def optimizer_sharding(self) -> bool:
        return self.parallelism.data.optimizer_sharding

    @property
    def tensor_parallel_communication(self) -> TensorParallelCommunication:
        return self.parallelism.tensor.communication

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
        if self.pipeline_parallel > model.block_count:
            raise ValueError("pipeline_parallel cannot exceed block_count")
        if model.block_count % self.pipeline_parallel:
            raise ValueError("pipeline_parallel must divide block_count for static planning")
        blocks_per_stage = model.block_count // self.pipeline_parallel
        if self.pipeline_interleaving > blocks_per_stage:
            raise ValueError("pipeline_interleaving cannot exceed blocks per pipeline stage")
        if blocks_per_stage % self.pipeline_interleaving:
            raise ValueError("pipeline_interleaving must divide blocks per pipeline stage")

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
        parallelism = TransformerTrainingParallelism(
            TensorParallel(
                tensor_parallel,
                TensorParallelCommunication(_field(data, "tensor_parallel_communication", "tensor_par_comm_type")),
            ),
            PipelineParallel(
                pipeline_parallel,
                _training_schedule(pipeline_parallel, data["pipeline_interleaving"]),
            ),
            DataParallel(data_parallel, data["optimizer_sharding"]),
            RecomputePolicy(_field(data, "recompute", "activation_recompute")),
        )
        return cls(
            parallelism=parallelism,
            fused_activation=data.get("fused_activation", False),
            sequence_parallel_all_gather_redo=data.get("seq_par_ag_redo", False),
        )


@record("blueprinting.mapping.transformer-inference")
class TransformerInferenceMappingSpec:
    """Target-neutral logical mapping for an inference replica."""

    parallelism: TransformerInferenceParallelism

    @property
    def world_size(self) -> int:
        return self.tensor_parallel * self.pipeline_parallel * self.replicas

    @property
    def tensor_parallel(self) -> int:
        return self.parallelism.tensor.degree

    @property
    def pipeline_parallel(self) -> int:
        return self.parallelism.pipeline.degree

    @property
    def replicas(self) -> int:
        return self.parallelism.replicas.degree

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
        schedule: PipelineScheduleVariant = SingleStage() if pipeline_parallel == 1 else ForwardOnly()
        return cls(
            TransformerInferenceParallelism(
                TensorParallel(tensor_parallel, TensorParallelCommunication.ALL_REDUCE),
                PipelineParallel(pipeline_parallel, schedule),
                ReplicaParallel(replicas),
            )
        )

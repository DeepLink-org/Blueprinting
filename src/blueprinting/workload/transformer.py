"""Target-neutral decoder-only Transformer workload contracts.

These immutable descriptions own model semantics and logical mapping intent.
They do not construct canonical IR, bind a hardware target, or estimate time;
the synthesizer frontend and analysis packages own those responsibilities.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from blueprinting.synthesizer.codec import enum_type, record_type


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@enum_type("compiler.transformer.recompute_policy")
class RecomputePolicy(Enum):
    NONE = "none"
    ATTENTION = "attn_only"
    FULL = "full"


@enum_type("compiler.transformer.tp_communication")
class TensorParallelCommunication(Enum):
    ALL_REDUCE = "ar"
    REDUCE_SCATTER_ALL_GATHER = "rs_ag"


@record_type("compiler.transformer.model_spec.v1")
@dataclass(frozen=True)
class TransformerModelSpec:
    """Target-independent decoder-only Transformer dimensions."""

    name: str
    hidden_size: int
    feedforward_size: int
    sequence_length: int
    attention_heads: int
    attention_head_size: int
    block_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Transformer model name must not be empty")
        for field_name in (
            "hidden_size",
            "feedforward_size",
            "sequence_length",
            "attention_heads",
            "attention_head_size",
            "block_count",
        ):
            _positive_integer(getattr(self, field_name), field_name)
        if self.attention_heads * self.attention_head_size != self.hidden_size:
            raise ValueError("attention_heads * attention_head_size must equal hidden_size")

    @classmethod
    def from_mapping(cls, name: str, data: Mapping[str, Any]) -> TransformerModelSpec:
        return cls(
            name=name,
            hidden_size=data["hidden"],
            feedforward_size=data["feedforward"],
            sequence_length=data["seq_size"],
            attention_heads=data["attn_heads"],
            attention_head_size=data["attn_size"],
            block_count=data["num_blocks"],
        )


@record_type("compiler.transformer.execution_spec.v1")
@dataclass(frozen=True)
class TransformerExecutionSpec:
    """Structurally relevant training strategy used by the comparison experiment.

    Every field changes structure, multiplicity, storage, or communication.
    There are intentionally no efficiency or correction-factor fields here.
    """

    world_size: int
    tensor_parallel: int
    pipeline_parallel: int
    data_parallel: int
    global_batch_size: int
    microbatch_size: int
    datatype: str
    recompute: RecomputePolicy
    pipeline_interleaving: int
    optimizer_sharding: bool
    tensor_parallel_communication: TensorParallelCommunication
    tensor_parallel_network: int
    pipeline_parallel_network: int
    data_parallel_network: int
    fused_activation: bool = False
    sequence_parallel_all_gather_redo: bool = False
    data_parallel_overlap: bool = False
    training: bool = True
    attention_type: str = "multihead"
    tensor_parallel_overlap: str = "none"
    weight_offload: bool = False
    activation_offload: bool = False
    optimizer_offload: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "world_size",
            "tensor_parallel",
            "pipeline_parallel",
            "data_parallel",
            "global_batch_size",
            "microbatch_size",
            "pipeline_interleaving",
        ):
            _positive_integer(getattr(self, field_name), field_name)
        for field_name in (
            "tensor_parallel_network",
            "pipeline_parallel_network",
            "data_parallel_network",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        if self.world_size != self.tensor_parallel * self.pipeline_parallel * self.data_parallel:
            raise ValueError("world_size must equal tensor_parallel * pipeline_parallel * data_parallel")
        if self.global_batch_size % self.data_parallel:
            raise ValueError("global_batch_size must be divisible by data_parallel")
        if self.local_batch_size % self.microbatch_size:
            raise ValueError("local batch size must be divisible by microbatch_size")
        if not isinstance(self.recompute, RecomputePolicy):
            raise TypeError("recompute must be a RecomputePolicy")
        if not isinstance(self.tensor_parallel_communication, TensorParallelCommunication):
            raise TypeError("tensor_parallel_communication must be TensorParallelCommunication")
        if self.datatype not in {"float16", "bfloat16", "float32", "float8"}:
            raise ValueError(f"unsupported datatype: {self.datatype!r}")
        if self.attention_type != "multihead":
            raise ValueError("the calibrated experiment currently supports multihead attention only")
        if self.tensor_parallel_overlap != "none":
            raise ValueError("overlapped TP requires concrete scheduling and is outside this experiment")
        if self.data_parallel_overlap:
            raise ValueError("overlapped DP requires concrete scheduling and is outside this experiment")
        if self.weight_offload or self.activation_offload or self.optimizer_offload:
            raise ValueError("offload strategies are outside this experiment")
        if not self.training:
            raise ValueError("this execution spec describes training only")
        if self.optimizer_sharding and self.data_parallel == 1:
            raise ValueError("optimizer sharding requires data_parallel > 1")

    @property
    def local_batch_size(self) -> int:
        return self.global_batch_size // self.data_parallel

    @property
    def microbatch_count(self) -> int:
        return self.local_batch_size // self.microbatch_size

    @property
    def bytes_per_element(self) -> int:
        return {"float8": 1, "float16": 2, "bfloat16": 2, "float32": 4}[self.datatype]

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> TransformerExecutionSpec:
        return cls(
            world_size=data["num_procs"],
            tensor_parallel=data["tensor_par"],
            pipeline_parallel=data["pipeline_par"],
            data_parallel=data["data_par"],
            global_batch_size=data["batch_size"],
            microbatch_size=data["microbatch_size"],
            datatype=data["datatype"],
            recompute=RecomputePolicy(data["activation_recompute"]),
            pipeline_interleaving=data["pipeline_interleaving"],
            optimizer_sharding=data["optimizer_sharding"],
            tensor_parallel_communication=TensorParallelCommunication(data["tensor_par_comm_type"]),
            tensor_parallel_network=data["tensor_par_net"],
            pipeline_parallel_network=data["pipeline_par_net"],
            data_parallel_network=data["data_par_net"],
            fused_activation=data.get("fused_activation", False),
            sequence_parallel_all_gather_redo=data.get("seq_par_ag_redo", False),
            data_parallel_overlap=data.get("data_par_overlap", False),
            training=data.get("training", True),
            attention_type=data.get("attention_type", "multihead"),
            tensor_parallel_overlap=data.get("tensor_par_overlap", "none"),
            weight_offload=data.get("weight_offload", False),
            activation_offload=data.get("activations_offload", False),
            optimizer_offload=data.get("optimizer_offload", False),
        )

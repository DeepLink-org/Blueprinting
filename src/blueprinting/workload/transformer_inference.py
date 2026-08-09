"""Target-neutral workload facts for decoder-only Transformer inference.

Inference keeps three concerns separate:

* :class:`TransformerModelSpec` describes model structure;
* :class:`TransformerInferenceExecutionSpec` describes a logical mapping;
* :class:`TransformerInferenceRequestSpec` describes one request cohort.

The contracts describe logical mapping and one request cohort. Canonical IR
construction and phase binding are owned by the synthesizer frontend.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from blueprinting.synthesizer.codec import record_type

from .transformer import TransformerModelSpec

_SUPPORTED_DATATYPES = frozenset({"float8", "float16", "bfloat16", "float32"})


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@record_type("compiler.transformer.inference_execution_spec.v1")
@dataclass(frozen=True)
class TransformerInferenceExecutionSpec:
    """Target-neutral logical mapping for an inference replica."""

    world_size: int
    tensor_parallel: int
    pipeline_parallel: int
    replicas: int
    datatype: str
    tensor_parallel_network: int
    pipeline_parallel_network: int

    def __post_init__(self) -> None:
        for field_name in ("world_size", "tensor_parallel", "pipeline_parallel", "replicas"):
            _positive_integer(getattr(self, field_name), field_name)
        for field_name in ("tensor_parallel_network", "pipeline_parallel_network"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        if self.world_size != self.tensor_parallel * self.pipeline_parallel * self.replicas:
            raise ValueError("world_size must equal tensor_parallel * pipeline_parallel * replicas")
        if self.datatype not in _SUPPORTED_DATATYPES:
            raise ValueError(f"unsupported datatype: {self.datatype!r}")

    @property
    def bytes_per_element(self) -> int:
        return {"float8": 1, "float16": 2, "bfloat16": 2, "float32": 4}[self.datatype]

    def validate_model(self, model: TransformerModelSpec) -> None:
        """Reject mappings whose local tensor shapes are not integral."""

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
    def from_mapping(cls, data: Mapping[str, Any]) -> TransformerInferenceExecutionSpec:
        replicas = data.get("replicas", data.get("data_par", 1))
        tensor_parallel = data["tensor_par"]
        pipeline_parallel = data["pipeline_par"]
        return cls(
            world_size=data.get("num_procs", tensor_parallel * pipeline_parallel * replicas),
            tensor_parallel=tensor_parallel,
            pipeline_parallel=pipeline_parallel,
            replicas=replicas,
            datatype=data["datatype"],
            tensor_parallel_network=data.get("tensor_par_net", 0),
            pipeline_parallel_network=data.get("pipeline_par_net", 0),
        )


@record_type("compiler.transformer.inference_request_spec.v1")
@dataclass(frozen=True)
class TransformerInferenceRequestSpec:
    """A homogeneous request cohort before online scheduling is applied."""

    batch_size: int
    prompt_tokens: int
    generated_tokens: int

    def __post_init__(self) -> None:
        for field_name in ("batch_size", "prompt_tokens", "generated_tokens"):
            _positive_integer(getattr(self, field_name), field_name)

    @property
    def decode_iterations(self) -> int:
        """Iterations after prefill; prefill itself emits the first token."""

        return self.generated_tokens - 1

    @property
    def final_context_tokens(self) -> int:
        """Largest KV context consumed while producing this cohort."""

        return self.prompt_tokens + self.decode_iterations

    def decode_contexts(self) -> range:
        return range(self.prompt_tokens + 1, self.final_context_tokens + 1)

    def validate_model(self, model: TransformerModelSpec) -> None:
        if self.final_context_tokens > model.sequence_length:
            raise ValueError("prompt_tokens + generated_tokens - 1 cannot exceed the model sequence_length")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> TransformerInferenceRequestSpec:
        return cls(
            batch_size=data["batch_size"],
            prompt_tokens=data["prompt_tokens"],
            generated_tokens=data["generated_tokens"],
        )

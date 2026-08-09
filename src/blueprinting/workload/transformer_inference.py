"""Target-neutral workload facts for decoder-only Transformer inference.

Inference keeps three concerns separate:

* :class:`TransformerModelSpec` describes model structure;
* :class:`TransformerInferenceRequestSpec` describes one request cohort.

The contract describes one request cohort and its numerical representation.
Logical mapping is owned by :mod:`blueprinting.mapping`; canonical IR
construction and phase binding are owned by the synthesizer frontend.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from blueprinting.schema.codec import record_type

from .transformer import TransformerModelSpec


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@record_type("compiler.transformer.inference_request_spec.v1")
@dataclass(frozen=True)
class TransformerInferenceRequestSpec:
    """A homogeneous request cohort before online scheduling is applied."""

    batch_size: int
    prompt_tokens: int
    generated_tokens: int
    datatype: str = "float16"

    def __post_init__(self) -> None:
        for field_name in ("batch_size", "prompt_tokens", "generated_tokens"):
            _positive_integer(getattr(self, field_name), field_name)
        if self.datatype not in {"float8", "float16", "bfloat16", "float32"}:
            raise ValueError(f"unsupported datatype: {self.datatype!r}")

    @property
    def bytes_per_element(self) -> int:
        return {"float8": 1, "float16": 2, "bfloat16": 2, "float32": 4}[self.datatype]

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
            datatype=data.get("datatype", "float16"),
        )

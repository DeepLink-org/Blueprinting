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
from typing import Any

from blueprinting.schema.authoring import PositiveInt, record

from .transformer import TransformerDataType, TransformerModelSpec, transformer_element_bytes


@record("blueprinting.workload.transformer-inference-request")
class TransformerInferenceRequestSpec:
    """A homogeneous request cohort before online scheduling is applied."""

    batch_size: PositiveInt
    prompt_tokens: PositiveInt
    generated_tokens: PositiveInt
    datatype: TransformerDataType = "float16"

    @property
    def bytes_per_element(self) -> int:
        return transformer_element_bytes(self.datatype)

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

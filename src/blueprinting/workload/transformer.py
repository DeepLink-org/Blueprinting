"""Target-neutral decoder-only Transformer workload contracts.

These immutable descriptions own model semantics and scenario facts.
They do not construct canonical IR, bind a hardware target, or estimate time;
the synthesizer frontend and analysis packages own those responsibilities.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from blueprinting.schema.codec import record_type


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


@record_type("blueprinting.workload.transformer-model")
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


@record_type("blueprinting.workload.transformer-training")
@dataclass(frozen=True)
class TransformerTrainingWorkloadSpec:
    """Training scenario facts independent of parallel mapping and hardware."""

    global_batch_size: int
    microbatch_size: int
    datatype: str

    def __post_init__(self) -> None:
        for field_name in ("global_batch_size", "microbatch_size"):
            _positive_integer(getattr(self, field_name), field_name)
        if self.datatype not in {"float16", "bfloat16", "float32", "float8"}:
            raise ValueError(f"unsupported datatype: {self.datatype!r}")
        if self.microbatch_size > self.global_batch_size:
            raise ValueError("microbatch_size cannot exceed global_batch_size")

    @property
    def bytes_per_element(self) -> int:
        return {"float8": 1, "float16": 2, "bfloat16": 2, "float32": 4}[self.datatype]

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> TransformerTrainingWorkloadSpec:
        if "global_batch_size" in data and "batch_size" in data and data["global_batch_size"] != data["batch_size"]:
            raise ValueError("global_batch_size conflicts with legacy alias batch_size")
        return cls(
            global_batch_size=(data["global_batch_size"] if "global_batch_size" in data else data["batch_size"]),
            microbatch_size=data["microbatch_size"],
            datatype=data["datatype"],
        )

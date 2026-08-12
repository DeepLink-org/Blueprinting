"""Target-neutral decoder-only Transformer workload contracts.

These immutable descriptions own model semantics and scenario facts.
They do not construct canonical IR, bind a hardware target, or estimate time;
the synthesizer frontend and analysis packages own those responsibilities.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final, Literal, TypeAlias, cast, get_args

from blueprinting.schema.authoring import NonEmptyText, PositiveInt, record

TransformerDataType: TypeAlias = Literal["float8", "float16", "bfloat16", "float32"]
TRANSFORMER_DATA_TYPES: Final = cast(tuple[TransformerDataType, ...], get_args(TransformerDataType))

_TRANSFORMER_ELEMENT_BYTES: Final[dict[TransformerDataType, int]] = {
    "float8": 1,
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
}

if set(_TRANSFORMER_ELEMENT_BYTES) != set(TRANSFORMER_DATA_TYPES):  # pragma: no cover - import-time contract
    raise RuntimeError("Transformer datatype widths must cover the complete datatype domain")


def require_transformer_data_type(value: object) -> TransformerDataType:
    """Narrow an untyped boundary value to the closed Transformer datatype domain."""

    if type(value) is not str or value not in TRANSFORMER_DATA_TYPES:
        raise ValueError(f"unsupported Transformer datatype: {value!r}")
    return value


def transformer_element_bytes(datatype: TransformerDataType) -> int:
    """Return the canonical storage width for one Transformer element."""

    return _TRANSFORMER_ELEMENT_BYTES[require_transformer_data_type(datatype)]


@record("blueprinting.workload.transformer-model")
class TransformerModelSpec:
    """Target-independent decoder-only Transformer dimensions."""

    name: NonEmptyText
    hidden_size: PositiveInt
    feedforward_size: PositiveInt
    sequence_length: PositiveInt
    attention_heads: PositiveInt
    attention_head_size: PositiveInt
    block_count: PositiveInt

    def __post_init__(self) -> None:
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


@record("blueprinting.workload.transformer-training")
class TransformerTrainingWorkloadSpec:
    """Training scenario facts independent of parallel mapping and hardware."""

    global_batch_size: PositiveInt
    microbatch_size: PositiveInt
    datatype: TransformerDataType

    def __post_init__(self) -> None:
        if self.microbatch_size > self.global_batch_size:
            raise ValueError("microbatch_size cannot exceed global_batch_size")

    @property
    def bytes_per_element(self) -> int:
        return transformer_element_bytes(self.datatype)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> TransformerTrainingWorkloadSpec:
        if "global_batch_size" in data and "batch_size" in data and data["global_batch_size"] != data["batch_size"]:
            raise ValueError("global_batch_size conflicts with legacy alias batch_size")
        return cls(
            global_batch_size=(data["global_batch_size"] if "global_batch_size" in data else data["batch_size"]),
            microbatch_size=data["microbatch_size"],
            datatype=data["datatype"],
        )

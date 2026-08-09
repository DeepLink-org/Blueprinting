"""Normalized latency evidence contracts with an explicit oracle boundary.

Cost providers are admissible inputs to Blueprinting's estimator.  Baselines
are read-only comparison oracles and therefore expose a different method name;
they cannot be passed accidentally as cost providers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..bindings import InferencePhase


@dataclass(frozen=True)
class InferenceEvidenceQuery:
    """A component query with every shape fact needed for reproducibility."""

    phase: InferencePhase
    primitive: str
    source_layer: str
    model_name: str
    hardware_name: str
    model_sequence_length: int
    hidden_size: int
    feedforward_size: int
    attention_heads: int
    batch_size: int
    query_tokens: int
    context_tokens: int
    tensor_parallel: int
    datatype: str

    def __post_init__(self) -> None:
        if not isinstance(self.phase, InferencePhase):
            raise TypeError("phase must be InferencePhase")
        for field_name in ("primitive", "source_layer", "model_name", "hardware_name", "datatype"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field_name} must not be empty")
        for field_name in (
            "model_sequence_length",
            "hidden_size",
            "feedforward_size",
            "attention_heads",
            "batch_size",
            "query_tokens",
            "context_tokens",
            "tensor_parallel",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if self.query_tokens > self.context_tokens:
            raise ValueError("query_tokens cannot exceed context_tokens")
        if self.context_tokens > self.model_sequence_length:
            raise ValueError("context_tokens cannot exceed model_sequence_length")
        if self.phase is InferencePhase.DECODE and self.query_tokens != 1:
            raise ValueError("decode evidence queries require query_tokens == 1")


@dataclass(frozen=True)
class InferenceEvidenceResult:
    """Measured or simulated component latency and its provenance."""

    seconds: float
    provider: str
    revision: str
    match: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.seconds, bool)
            or not isinstance(self.seconds, (int, float))
            or not math.isfinite(self.seconds)
            or self.seconds < 0
        ):
            raise ValueError("evidence seconds must be finite and non-negative")
        for field_name in ("provider", "revision", "match"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field_name} must not be empty")


@runtime_checkable
class InferenceCostProvider(Protocol):
    """Admissible Blueprinting cost source, such as its performance database."""

    @property
    def revision(self) -> str: ...

    def resolve(self, query: InferenceEvidenceQuery) -> InferenceEvidenceResult | None: ...


@runtime_checkable
class InferenceBaseline(Protocol):
    """External comparison oracle that must never participate in lowering or costing."""

    @property
    def revision(self) -> str: ...

    def lookup(self, query: InferenceEvidenceQuery) -> InferenceEvidenceResult | None: ...

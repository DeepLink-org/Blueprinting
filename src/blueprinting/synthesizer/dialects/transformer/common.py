"""Shared target-neutral work primitives for Transformer dialects."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from blueprinting.schema.codec import enum_type, record_type

# Codec tags retain their legacy namespace as stable serialized identities.


@enum_type("compiler.analysis.engine_kind")
class EngineKind(Enum):
    MATRIX = "matrix"
    VECTOR = "vector"
    COLLECTIVE = "collective"


@record_type("compiler.analysis.phase_work.v1")
@dataclass(frozen=True)
class PhaseWork:
    """Exact work for one invocation, before target binding."""

    operations: int = 0
    read_bytes: int = 0
    write_bytes: int = 0
    message_bytes: int = 0

    def __post_init__(self) -> None:
        for field_name in ("operations", "read_bytes", "write_bytes", "message_bytes"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")

    @property
    def memory_bytes(self) -> int:
        return self.read_bytes + self.write_bytes

    @property
    def is_empty(self) -> bool:
        return self.operations == 0 and self.memory_bytes == 0 and self.message_bytes == 0

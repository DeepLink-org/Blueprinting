"""Shared target-neutral work primitives for Transformer dialects."""

from __future__ import annotations

from enum import Enum

from blueprinting.schema.authoring import NonNegativeInt, enum, record

# Codec tags retain their legacy namespace as stable serialized identities.


@enum("blueprinting.analysis.transformer.engine-kind")
class EngineKind(Enum):
    MATRIX = "matrix"
    VECTOR = "vector"
    COLLECTIVE = "collective"


@record("blueprinting.analysis.transformer.phase-work")
class PhaseWork:
    """Exact work for one invocation, before target binding."""

    operations: NonNegativeInt = 0
    read_bytes: NonNegativeInt = 0
    write_bytes: NonNegativeInt = 0
    message_bytes: NonNegativeInt = 0

    @property
    def memory_bytes(self) -> int:
        return self.read_bytes + self.write_bytes

    @property
    def is_empty(self) -> bool:
        return self.operations == 0 and self.memory_bytes == 0 and self.message_bytes == 0

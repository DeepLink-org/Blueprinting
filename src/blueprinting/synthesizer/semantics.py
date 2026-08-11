"""Marker contracts for typed, dialect-owned synthesis semantics."""

from __future__ import annotations

from blueprinting.schema.authoring import record


class SemanticPayload:
    """Base marker for registered immutable semantic payloads."""


class ModelOperationSemantic(SemanticPayload):
    """Dialect semantics attached to a ModelIR operation."""


class DistributedTaskSemantic(SemanticPayload):
    """Dialect semantics attached to a DistributedTaskIR task."""


class PlanTaskSemantic(SemanticPayload):
    """Dialect semantics attached to a PortablePlanIR task."""


class ProgramSemantic(SemanticPayload):
    """Dialect semantics shared by one canonical program snapshot."""


class BindingSemantic(SemanticPayload):
    """Typed specialization attached to a synthesis binding."""


class BufferSemantic(SemanticPayload):
    """Typed semantic role attached to a portable buffer."""


@record("blueprinting.ir.semantic.empty")
class EmptySemantic(
    ModelOperationSemantic,
    DistributedTaskSemantic,
    PlanTaskSemantic,
    ProgramSemantic,
    BindingSemantic,
    BufferSemantic,
):
    """Explicit absence of dialect-specific semantics for generic fixtures."""

    namespace: str = "generic"

    def __post_init__(self) -> None:
        if not isinstance(self.namespace, str) or not self.namespace:
            raise ValueError("empty semantic namespace must not be empty")


EMPTY_SEMANTIC = EmptySemantic()


__all__ = [
    "BindingSemantic",
    "BufferSemantic",
    "DistributedTaskSemantic",
    "EMPTY_SEMANTIC",
    "EmptySemantic",
    "ModelOperationSemantic",
    "PlanTaskSemantic",
    "ProgramSemantic",
    "SemanticPayload",
]

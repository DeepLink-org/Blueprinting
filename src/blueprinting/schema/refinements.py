"""Reusable, domain-free refined value types for immutable contracts."""

from __future__ import annotations

from typing import Annotated, TypeAlias

from .deriving import ValueConstraint

NonEmptyText: TypeAlias = Annotated[str, ValueConstraint.NON_EMPTY]
NonBlankText: TypeAlias = Annotated[str, ValueConstraint.NON_BLANK]
StableName: TypeAlias = Annotated[str, ValueConstraint.STABLE_NAME]
ContentDigest: TypeAlias = Annotated[str, ValueConstraint.CONTENT_DIGEST]
SymbolName: TypeAlias = Annotated[str, ValueConstraint.SYMBOL_NAME]
CanonicalLowerText: TypeAlias = Annotated[str, ValueConstraint.CANONICAL_LOWER_TEXT]

NonNegativeInt: TypeAlias = Annotated[int, ValueConstraint.NON_NEGATIVE]
PositiveInt: TypeAlias = Annotated[int, ValueConstraint.POSITIVE]
AtLeastTwoInt: TypeAlias = Annotated[int, ValueConstraint.AT_LEAST_TWO]
FiniteFloat: TypeAlias = Annotated[float, ValueConstraint.FINITE]
NonNegativeFiniteFloat: TypeAlias = Annotated[
    float,
    ValueConstraint.FINITE,
    ValueConstraint.NON_NEGATIVE,
]
PositiveFiniteFloat: TypeAlias = Annotated[
    float,
    ValueConstraint.FINITE,
    ValueConstraint.POSITIVE,
]
NonNegativeFiniteNumber: TypeAlias = Annotated[
    int | float,
    ValueConstraint.FINITE,
    ValueConstraint.NON_NEGATIVE,
]
PositiveFiniteNumber: TypeAlias = Annotated[
    int | float,
    ValueConstraint.FINITE,
    ValueConstraint.POSITIVE,
]
UnitIntervalNumber: TypeAlias = Annotated[
    int | float,
    ValueConstraint.FINITE,
    ValueConstraint.NON_NEGATIVE,
    ValueConstraint.AT_MOST_ONE,
]
PositiveUnitIntervalNumber: TypeAlias = Annotated[
    int | float,
    ValueConstraint.FINITE,
    ValueConstraint.POSITIVE,
    ValueConstraint.AT_MOST_ONE,
]

__all__ = [
    "AtLeastTwoInt",
    "CanonicalLowerText",
    "ContentDigest",
    "FiniteFloat",
    "NonBlankText",
    "NonEmptyText",
    "NonNegativeFiniteFloat",
    "NonNegativeFiniteNumber",
    "NonNegativeInt",
    "PositiveFiniteFloat",
    "PositiveFiniteNumber",
    "PositiveInt",
    "PositiveUnitIntervalNumber",
    "StableName",
    "SymbolName",
    "UnitIntervalNumber",
]

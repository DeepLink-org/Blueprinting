"""Immutable compute and memory descriptions for one accelerator chip."""

from __future__ import annotations

from typing import Annotated, TypeAlias

from blueprinting.schema.authoring import (
    NonNegativeInt,
    PositiveFiniteNumber,
    PositiveInt,
    PositiveUnitIntervalNumber,
    ValueConstraint,
    record,
)


def _non_negative_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


@record("blueprinting.system.efficiency-point")
class EfficiencyPoint:
    """Measured or simulated efficiency above one work-size threshold."""

    threshold: NonNegativeInt
    efficiency: PositiveUnitIntervalNumber


EfficiencyPoints: TypeAlias = Annotated[
    tuple[EfficiencyPoint, ...],
    ValueConstraint.NON_EMPTY,
]


@record("blueprinting.system.efficiency-curve")
class EfficiencyCurve:
    """Piecewise-constant utilization evidence indexed by exact work size."""

    points: EfficiencyPoints

    def __post_init__(self) -> None:
        thresholds = tuple(point.threshold for point in self.points)
        if thresholds != tuple(sorted(thresholds, reverse=True)) or len(set(thresholds)) != len(thresholds):
            raise ValueError("efficiency thresholds must be unique and descending")
        if thresholds[-1] != 0:
            raise ValueError("efficiency curve must cover a zero threshold")

    def lookup(self, work: int) -> float:
        if isinstance(work, bool) or not isinstance(work, int) or work < 0:
            raise ValueError("curve lookup work must be a non-negative integer")
        for point in self.points:
            if work >= point.threshold:
                return point.efficiency
        raise AssertionError("zero-threshold curve failed to cover work")


@record("blueprinting.system.processor-profile")
class ProcessorProfile:
    """One chip compute engine and its size-dependent utilization evidence."""

    peak_operations_per_second: PositiveFiniteNumber
    efficiency: EfficiencyCurve

    def throughput(self, operations: int, *, apply_efficiency: bool = True) -> float:
        _non_negative_integer(operations, "operations")
        if not isinstance(apply_efficiency, bool):
            raise TypeError("apply_efficiency must be bool")
        efficiency = self.efficiency.lookup(operations) if apply_efficiency else 1.0
        return self.peak_operations_per_second * efficiency


@record("blueprinting.system.memory-profile")
class MemoryProfile:
    """One chip-visible memory tier and its transfer-efficiency evidence."""

    capacity_bytes: PositiveInt
    peak_bytes_per_second: PositiveFiniteNumber
    efficiency: EfficiencyCurve

    def throughput(self, transferred_bytes: int, *, apply_efficiency: bool = True) -> float:
        _non_negative_integer(transferred_bytes, "transferred_bytes")
        if not isinstance(apply_efficiency, bool):
            raise TypeError("apply_efficiency must be bool")
        efficiency = self.efficiency.lookup(transferred_bytes) if apply_efficiency else 1.0
        return self.peak_bytes_per_second * efficiency

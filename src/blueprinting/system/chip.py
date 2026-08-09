"""Immutable compute and memory descriptions for one accelerator chip."""

from __future__ import annotations

import math
from dataclasses import dataclass

from blueprinting.synthesizer.codec import record_type


def _positive_rate(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")


def _non_negative_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


@record_type("compiler.analysis.efficiency_point.v1")
@dataclass(frozen=True)
class EfficiencyPoint:
    """Measured or simulated efficiency above one work-size threshold."""

    threshold: int
    efficiency: float

    def __post_init__(self) -> None:
        if isinstance(self.threshold, bool) or not isinstance(self.threshold, int) or self.threshold < 0:
            raise ValueError("efficiency threshold must be a non-negative integer")
        if (
            isinstance(self.efficiency, bool)
            or not isinstance(self.efficiency, (int, float))
            or not math.isfinite(self.efficiency)
            or not 0 < self.efficiency <= 1
        ):
            raise ValueError("efficiency must be finite and in (0, 1]")


@record_type("compiler.analysis.efficiency_curve.v1")
@dataclass(frozen=True)
class EfficiencyCurve:
    """Piecewise-constant utilization evidence indexed by exact work size."""

    points: tuple[EfficiencyPoint, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "points", tuple(self.points))
        if not self.points or any(not isinstance(point, EfficiencyPoint) for point in self.points):
            raise ValueError("an efficiency curve requires typed points")
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


@record_type("compiler.analysis.processor_profile.v1")
@dataclass(frozen=True)
class ProcessorProfile:
    """One chip compute engine and its size-dependent utilization evidence."""

    peak_operations_per_second: float
    efficiency: EfficiencyCurve

    def __post_init__(self) -> None:
        _positive_rate(self.peak_operations_per_second, "peak_operations_per_second")
        if not isinstance(self.efficiency, EfficiencyCurve):
            raise TypeError("efficiency must be EfficiencyCurve")

    def throughput(self, operations: int, *, apply_efficiency: bool = True) -> float:
        _non_negative_integer(operations, "operations")
        if not isinstance(apply_efficiency, bool):
            raise TypeError("apply_efficiency must be bool")
        efficiency = self.efficiency.lookup(operations) if apply_efficiency else 1.0
        return self.peak_operations_per_second * efficiency


@record_type("compiler.analysis.memory_profile.v1")
@dataclass(frozen=True)
class MemoryProfile:
    """One chip-visible memory tier and its transfer-efficiency evidence."""

    capacity_bytes: int
    peak_bytes_per_second: float
    efficiency: EfficiencyCurve

    def __post_init__(self) -> None:
        if (
            isinstance(self.capacity_bytes, bool)
            or not isinstance(self.capacity_bytes, int)
            or self.capacity_bytes <= 0
        ):
            raise ValueError("capacity_bytes must be a positive integer")
        _positive_rate(self.peak_bytes_per_second, "peak_bytes_per_second")
        if not isinstance(self.efficiency, EfficiencyCurve):
            raise TypeError("efficiency must be EfficiencyCurve")

    def throughput(self, transferred_bytes: int, *, apply_efficiency: bool = True) -> float:
        _non_negative_integer(transferred_bytes, "transferred_bytes")
        if not isinstance(apply_efficiency, bool):
            raise TypeError("apply_efficiency must be bool")
        efficiency = self.efficiency.lookup(transferred_bytes) if apply_efficiency else 1.0
        return self.peak_bytes_per_second * efficiency

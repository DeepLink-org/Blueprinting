"""Immutable interconnect descriptions shared by system analysis providers."""

from __future__ import annotations

import math
from dataclasses import dataclass

from blueprinting.schema.codec import record_type
from blueprinting.schema.frozen import FrozenDict


@record_type("compiler.analysis.network_operation.v1")
@dataclass(frozen=True)
class NetworkOperationProfile:
    """Explicit byte-volume rule for one point-to-point or collective operation."""

    volume_multiplier: float
    participant_offset: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.volume_multiplier, bool)
            or not isinstance(self.volume_multiplier, (int, float))
            or not math.isfinite(self.volume_multiplier)
            or self.volume_multiplier <= 0
        ):
            raise ValueError("volume_multiplier must be a finite positive number")
        if isinstance(self.participant_offset, bool) or not isinstance(self.participant_offset, int):
            raise TypeError("participant_offset must be an integer")


@record_type("compiler.analysis.network_profile.v1")
@dataclass(frozen=True)
class NetworkProfile:
    """One interconnect tier with bandwidth, latency, capacity, and volume rules."""

    peak_bytes_per_second: float
    efficiency: float
    latency_seconds: float
    participant_capacity: int
    operations: FrozenDict

    def __post_init__(self) -> None:
        if (
            isinstance(self.peak_bytes_per_second, bool)
            or not isinstance(self.peak_bytes_per_second, (int, float))
            or not math.isfinite(self.peak_bytes_per_second)
            or self.peak_bytes_per_second <= 0
        ):
            raise ValueError("peak_bytes_per_second must be a finite positive number")
        if (
            isinstance(self.efficiency, bool)
            or not isinstance(self.efficiency, (int, float))
            or not math.isfinite(self.efficiency)
            or not 0 < self.efficiency <= 1
        ):
            raise ValueError("efficiency must be finite and in (0, 1]")
        if (
            isinstance(self.latency_seconds, bool)
            or not isinstance(self.latency_seconds, (int, float))
            or not math.isfinite(self.latency_seconds)
            or self.latency_seconds < 0
        ):
            raise ValueError("latency_seconds must be a finite non-negative number")
        if (
            isinstance(self.participant_capacity, bool)
            or not isinstance(self.participant_capacity, int)
            or self.participant_capacity <= 0
        ):
            raise ValueError("participant_capacity must be a positive integer")
        operations = FrozenDict(self.operations)
        if any(not isinstance(item, NetworkOperationProfile) for item in operations.values()):
            raise TypeError("operations must contain NetworkOperationProfile values")
        object.__setattr__(self, "operations", operations)

    def transferred_bytes(self, operation: str, message_bytes: int, participants: int) -> float:
        profile = self.operations.get(operation)
        if not isinstance(profile, NetworkOperationProfile):
            raise ValueError(f"network does not define operation {operation!r}")
        if isinstance(message_bytes, bool) or not isinstance(message_bytes, int) or message_bytes < 0:
            raise ValueError("message_bytes must be a non-negative integer")
        if isinstance(participants, bool) or not isinstance(participants, int) or participants < 1:
            raise ValueError("participants must be a positive integer")
        if participants > self.participant_capacity:
            raise ValueError("participants exceed interconnect capacity")
        scaled = message_bytes * profile.volume_multiplier
        return scaled + scaled / participants * profile.participant_offset

    def time(
        self,
        operation: str,
        message_bytes: int,
        participants: int,
        *,
        apply_efficiency: bool = True,
    ) -> float:
        if not isinstance(apply_efficiency, bool):
            raise TypeError("apply_efficiency must be bool")
        transferred_bytes = self.transferred_bytes(operation, message_bytes, participants)
        if participants < 2:
            return 0.0
        efficiency = self.efficiency if apply_efficiency else 1.0
        return self.latency_seconds + transferred_bytes / (self.peak_bytes_per_second * efficiency)

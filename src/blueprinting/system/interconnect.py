"""Immutable interconnect descriptions shared by system analysis providers."""

from __future__ import annotations

from blueprinting.schema.authoring import (
    NonNegativeFiniteNumber,
    PositiveFiniteNumber,
    PositiveInt,
    PositiveUnitIntervalNumber,
    record,
)
from blueprinting.schema.frozen import FrozenDict


@record("blueprinting.system.network-operation")
class NetworkOperationProfile:
    """Explicit byte-volume rule for one point-to-point or collective operation."""

    volume_multiplier: PositiveFiniteNumber
    participant_offset: int


@record("blueprinting.system.network-profile")
class NetworkProfile:
    """One interconnect tier with bandwidth, latency, capacity, and volume rules."""

    peak_bytes_per_second: PositiveFiniteNumber
    efficiency: PositiveUnitIntervalNumber
    latency_seconds: NonNegativeFiniteNumber
    participant_capacity: PositiveInt
    operations: FrozenDict[NetworkOperationProfile]

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

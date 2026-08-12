from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprinting.schema.codec import canonical_dumps, canonical_loads
from blueprinting.system import SystemProfile

ROOT = Path(__file__).resolve().parents[2]


def _profile() -> SystemProfile:
    data = json.loads((ROOT / "data" / "systems" / "a100_80g.json").read_text(encoding="utf-8"))
    return SystemProfile.from_mapping("a100_80g", data, datatype="float16")


def test_system_profile_owns_chip_memory_and_interconnect_contracts() -> None:
    profile = _profile()

    assert profile.matrix.peak_operations_per_second == 312e12
    assert profile.memory.capacity_bytes == 80 * 1024**3
    assert profile.networks
    assert profile.networks[0].operations


def test_analysis_policy_is_explicit_at_the_system_profile_boundary() -> None:
    profile = _profile()
    operations = 100_000_000

    peak = profile.matrix.throughput(operations, apply_efficiency=False)
    evidence_backed = profile.matrix.throughput(operations, apply_efficiency=True)

    assert peak == profile.matrix.peak_operations_per_second
    assert evidence_backed < peak

    with pytest.raises(ValueError, match="operations must be a non-negative integer"):
        profile.matrix.throughput(-1, apply_efficiency=False)


def test_system_profile_round_trip_preserves_legacy_wire_identity() -> None:
    profile = _profile()
    payload = canonical_dumps(profile)
    restored = canonical_loads(payload)

    assert restored == profile
    assert '"$type":"blueprinting.system.profile"' in payload


def test_interconnect_rejects_participant_counts_beyond_its_capacity() -> None:
    network = _profile().networks[0]

    with pytest.raises(ValueError, match="exceed interconnect capacity"):
        network.time("all_reduce", 1024, network.participant_capacity + 1)

    with pytest.raises(ValueError, match="message_bytes must be a non-negative integer"):
        network.time("all_reduce", -1, 1)

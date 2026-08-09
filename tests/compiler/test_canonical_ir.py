from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError, dataclass, replace

import pytest

from blueprinting.compiler import FrozenDict, NodeId, SerializationError
from blueprinting.compiler.codec import canonical_dumps, canonical_loads, record_type
from blueprinting.compiler.ir import (
    ConcretePlanIR,
    DistributedTaskIR,
    MachineIR,
    ModelIR,
    PortablePlanIR,
)


@pytest.mark.parametrize(
    ("fixture_name", "ir_type"),
    (
        ("model_ir", ModelIR),
        ("distributed_ir", DistributedTaskIR),
        ("portable_ir", PortablePlanIR),
        ("concrete_ir", ConcretePlanIR),
        ("machine_ir", MachineIR),
    ),
)
def test_all_canonical_irs_round_trip(request: pytest.FixtureRequest, fixture_name: str, ir_type: type) -> None:
    ir = request.getfixturevalue(fixture_name)

    serialized = ir.to_json()

    assert ir.verify().ok
    assert ir_type.from_json(serialized) == ir
    assert ir_type.from_json(serialized).digest == ir.digest
    assert ir.to_json() == serialized
    assert '"content_digest"' in serialized


def test_model_snapshot_has_golden_text_digest(model_ir: ModelIR) -> None:
    textual_digest = hashlib.sha256(model_ir.to_json().encode("utf-8")).hexdigest()

    assert textual_digest == "25974dbb92796fa575ff258df15ca59988b8f86160e05c2efd3fced828932d5e"


def test_snapshot_digest_detects_payload_tampering(model_ir: ModelIR) -> None:
    serialized = model_ir.to_json()
    tampered = serialized.replace("fixture-model", "tampered-model", 1)

    with pytest.raises(SerializationError, match="digest mismatch"):
        ModelIR.from_json(tampered)


def test_snapshot_schema_is_checked(model_ir: ModelIR) -> None:
    with pytest.raises(SerializationError, match="is not blueprinting.portable-plan"):
        PortablePlanIR.from_json(model_ir.to_json())


def test_snapshot_envelope_metadata_cannot_diverge_from_payload(model_ir: ModelIR) -> None:
    serialized = model_ir.to_json()
    prefix, separator, suffix = serialized.rpartition('"producer_version":"0.1.0"')
    tampered = prefix + separator.replace("0.1.0", "9.9.9") + suffix

    with pytest.raises(SerializationError, match="envelope metadata"):
        ModelIR.from_json(tampered)


def test_codec_is_closed_world() -> None:
    with pytest.raises(SerializationError, match="unknown canonical record tag"):
        canonical_loads('{"$type":"python.eval","fields":{}}')


def test_codec_rejects_duplicate_keys_nonfinite_values_and_nonstring_map_keys() -> None:
    with pytest.raises(SerializationError, match="duplicate canonical JSON key"):
        canonical_loads('{"$tuple":[],"$tuple":[]}')
    with pytest.raises(SerializationError, match="does not permit NaN"):
        canonical_loads("NaN")
    with pytest.raises(SerializationError, match="mapping keys must be strings"):
        canonical_dumps({1: "invalid"})


def test_codec_registration_requires_frozen_records() -> None:
    @dataclass
    class MutableRecord:
        value: int

    with pytest.raises(TypeError, match="must be frozen"):
        record_type("tests.mutable-record")(MutableRecord)


def test_ir_snapshot_is_deeply_immutable(model_ir: ModelIR) -> None:
    source = {"nested": {"labels": ["initial"]}}
    snapshot = replace(model_ir, attributes=source)
    digest = snapshot.digest

    source["nested"]["labels"].append("mutated")

    assert snapshot.attributes["nested"]["labels"] == ("initial",)
    assert snapshot.digest == digest
    with pytest.raises(FrozenInstanceError):
        snapshot.name = "mutated"  # type: ignore[misc]
    with pytest.raises(TypeError):
        snapshot.attributes["new"] = "value"  # type: ignore[index]


def test_typed_ids_are_deterministic_and_namespace_separated() -> None:
    assert NodeId.derive("fixture", 1) == NodeId.derive("fixture", 1)
    assert str(NodeId.derive("fixture", 1)).startswith("node:")
    assert NodeId.derive("fixture", 1) != NodeId.derive("fixture", 2)
    assert FrozenDict({"id": NodeId.derive("fixture", 1)}) == FrozenDict({"id": NodeId.derive("fixture", 1)})

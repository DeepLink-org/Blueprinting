from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError, dataclass, replace

import pytest

from blueprinting.schema import FrozenDict, SerializationError, canonical_dumps, canonical_loads
from blueprinting.schema.codec import record_type
from blueprinting.synthesizer import BindingAxis, NodeId, Symbol
from blueprinting.synthesizer.stages.common import IRHeader, IRSnapshot
from blueprinting.synthesizer.stages.concrete_plan.ir import ConcretePlanIR
from blueprinting.synthesizer.stages.distributed.ir import DistributedTaskIR
from blueprinting.synthesizer.stages.machine.ir import MachineIR
from blueprinting.synthesizer.stages.model.ir import ModelIR
from blueprinting.synthesizer.stages.portable_plan.ir import PortablePlanIR, require_concrete_quantity


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

    assert ir.verify().is_ok
    assert ir_type.from_json(serialized).or_raise() == ir
    assert ir_type.from_json(serialized).or_raise().digest == ir.digest
    assert ir.to_json() == serialized
    assert '"content_digest"' in serialized


def test_model_snapshot_has_golden_text_digest(model_ir: ModelIR) -> None:
    textual_digest = hashlib.sha256(model_ir.to_json().encode("utf-8")).hexdigest()

    assert textual_digest == "b9773940c665731f0e94179b41aa5a90a47991b801477bae4b275e0a268fd41b"


def test_same_version_snapshot_without_typed_semantics_epoch_is_rejected(model_ir: ModelIR) -> None:
    old_header = IRHeader(
        model_ir.SCHEMA_NAME,
        model_ir.SCHEMA_VERSION,
        producer_version=model_ir.header.producer_version,
        feature_set=frozenset(),
    )
    old_model = replace(model_ir, header=old_header)
    payload = canonical_dumps(
        IRSnapshot(
            schema_name=old_model.SCHEMA_NAME,
            schema_version=old_model.SCHEMA_VERSION,
            producer_version=old_header.producer_version,
            feature_set=frozenset(),
            content_digest=old_model.digest,
            payload=old_model,
        )
    )

    with pytest.raises(SerializationError, match="typed-semantics"):
        ModelIR.require_from_json(payload)


def test_snapshot_digest_detects_payload_tampering(model_ir: ModelIR) -> None:
    serialized = model_ir.to_json()
    tampered = serialized.replace("fixture-model", "tampered-model", 1)

    with pytest.raises(SerializationError, match="digest mismatch"):
        ModelIR.require_from_json(tampered)


def test_snapshot_schema_is_checked(model_ir: ModelIR) -> None:
    with pytest.raises(SerializationError, match="is not blueprinting.portable-plan"):
        PortablePlanIR.require_from_json(model_ir.to_json())


def test_snapshot_envelope_metadata_cannot_diverge_from_payload(model_ir: ModelIR) -> None:
    serialized = model_ir.to_json()
    prefix, separator, suffix = serialized.rpartition('"producer_version":"0.0.0"')
    tampered = prefix + separator.replace("0.0.0", "9.9.9") + suffix

    with pytest.raises(SerializationError, match="envelope metadata"):
        ModelIR.require_from_json(tampered)


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


def test_concrete_workload_quantity_gate_rejects_symbolic_float_and_negative_values() -> None:
    assert require_concrete_quantity(0, "operations") == 0

    with pytest.raises(TypeError, match="concrete integer"):
        require_concrete_quantity(1.0, "operations")
    with pytest.raises(TypeError, match="concrete integer"):
        require_concrete_quantity(Symbol("batch", BindingAxis.WORKLOAD), "operations")
    with pytest.raises(ValueError, match="non-negative"):
        require_concrete_quantity(-1, "operations")

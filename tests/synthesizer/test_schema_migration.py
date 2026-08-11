from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import pytest

from blueprinting.schema import (
    canonical_dump_raw,
    canonical_dumps,
    canonical_parse,
    content_digest,
)
from blueprinting.schema.codec import record_type
from blueprinting.schema.errors import SerializationError
from blueprinting.synthesizer.schema_migration import (
    DEFAULT_SCHEMA_MIGRATIONS,
    SchemaMigration,
    SchemaMigrationRegistry,
)
from blueprinting.synthesizer.stages.common import IRSnapshot, SchemaVersion
from blueprinting.synthesizer.stages.concrete_plan.ir import ConcretePlanIR
from blueprinting.synthesizer.stages.distributed.ir import DistributedTaskIR
from blueprinting.synthesizer.stages.machine.ir import MachineIR
from blueprinting.synthesizer.stages.model.ir import ModelIR
from blueprinting.synthesizer.stages.portable_plan.ir import PortablePlanIR

SCHEMA = "blueprinting.test-migration"
V0 = SchemaVersion(0, 0, 0)
V01 = SchemaVersion(0, 1, 0)
V02 = SchemaVersion(0, 2, 0)


@record_type("blueprinting.test.migration-payload")
@dataclass(frozen=True)
class _Payload:
    revision: int


def _snapshot(version: SchemaVersion, revision: int) -> str:
    payload = _Payload(revision)
    return canonical_dumps(
        IRSnapshot(
            schema_name=SCHEMA,
            schema_version=version,
            producer_version="test",
            feature_set=frozenset(),
            content_digest=content_digest(payload, f"ir:{SCHEMA}"),
            payload=payload,
        )
    )


def _upgrade(to_version: SchemaVersion) -> Callable[[Any], Any]:
    def transform(raw: Any) -> Any:
        from blueprinting.schema import canonical_decode

        snapshot = canonical_decode(raw)
        payload = replace(snapshot.payload, revision=snapshot.payload.revision + 1)
        upgraded = replace(
            snapshot,
            schema_version=to_version,
            content_digest=content_digest(payload, f"ir:{SCHEMA}"),
            payload=payload,
        )
        return canonical_parse(canonical_dumps(upgraded))

    return transform


def _registry() -> SchemaMigrationRegistry:
    registry = SchemaMigrationRegistry()
    registry.register(SchemaMigration(SCHEMA, V0, V01, "test.0.0-to-0.1", _upgrade(V01)))
    registry.register(SchemaMigration(SCHEMA, V01, V02, "test.0.1-to-0.2", _upgrade(V02)))
    return registry


def test_current_canonical_schema_epoch_is_zero_and_has_no_history() -> None:
    assert {
        ModelIR.SCHEMA_VERSION,
        DistributedTaskIR.SCHEMA_VERSION,
        PortablePlanIR.SCHEMA_VERSION,
        ConcretePlanIR.SCHEMA_VERSION,
        MachineIR.SCHEMA_VERSION,
    } == {V0}
    for ir_type in (ModelIR, DistributedTaskIR, PortablePlanIR, ConcretePlanIR, MachineIR):
        with pytest.raises(SerializationError, match="no migration path"):
            DEFAULT_SCHEMA_MIGRATIONS.path(ir_type.SCHEMA_NAME, V0, V01)


def test_migration_mechanism_is_explicit_deterministic_and_digest_checked() -> None:
    result = _registry().migrate_json(_snapshot(V0, 1), schema_name=SCHEMA, target_version=V02)

    assert result.source_version == V0
    assert result.target_version == V02
    assert result.migration_ids == ("test.0.0-to-0.1", "test.0.1-to-0.2")
    assert result.source_digest != result.target_digest


def test_current_version_load_is_an_idempotent_noop() -> None:
    original = _snapshot(V02, 3)
    result = _registry().migrate_json(original, schema_name=SCHEMA, target_version=V02)

    assert result.payload == original
    assert result.migration_ids == ()
    assert result.source_digest == result.target_digest


def test_registry_rejects_unknown_ambiguous_duplicate_and_backward_edges() -> None:
    registry = _registry()
    with pytest.raises(SerializationError, match="no migration path"):
        registry.migrate_json(_snapshot(V0, 1), schema_name=SCHEMA, target_version=SchemaVersion(1, 0, 0))
    registry.register(SchemaMigration(SCHEMA, V0, V02, "test.direct", _upgrade(V02)))
    with pytest.raises(SerializationError, match="ambiguous migration path"):
        registry.migrate_json(_snapshot(V0, 1), schema_name=SCHEMA, target_version=V02)
    with pytest.raises(ValueError, match="duplicate schema migration edge"):
        registry.register(SchemaMigration(SCHEMA, V0, V01, "test.duplicate", _upgrade(V01)))
    with pytest.raises(ValueError, match="advance the version"):
        SchemaMigration(SCHEMA, V02, V0, "test.backward", _upgrade(V0))


def test_migration_rejects_tampered_source_digest() -> None:
    raw = canonical_parse(_snapshot(V0, 1))
    raw["fields"]["payload"]["fields"]["revision"] = 99

    with pytest.raises(SerializationError, match="digest mismatch"):
        _registry().migrate_json(canonical_dump_raw(raw), schema_name=SCHEMA, target_version=V01)

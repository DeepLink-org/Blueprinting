"""Explicit, deterministic migration registry for canonical IR snapshots."""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from blueprinting.schema.codec import (
    canonical_decode,
    canonical_dump_raw,
    canonical_parse,
    raw_content_digest,
)
from blueprinting.schema.errors import SerializationError

from .stages.common import SchemaVersion

RawMigration = Callable[[Any], Any]


@dataclass(frozen=True)
class SchemaMigration:
    schema_name: str
    from_version: SchemaVersion
    to_version: SchemaVersion
    migration_id: str
    transform: RawMigration

    def __post_init__(self) -> None:
        if not self.schema_name or not self.migration_id:
            raise ValueError("migration schema name and ID must not be empty")
        if self.to_version <= self.from_version:
            raise ValueError("schema migrations must advance the version")
        if not callable(self.transform):
            raise TypeError("schema migration transform must be callable")


@dataclass(frozen=True)
class MigrationResult:
    payload: str
    schema_name: str
    source_version: SchemaVersion
    target_version: SchemaVersion
    source_digest: str
    target_digest: str
    migration_ids: tuple[str, ...]


@dataclass(frozen=True)
class _RawSnapshot:
    schema_name: str
    schema_version: SchemaVersion
    content_digest: str
    payload: Any


def _raw_snapshot(raw: Any) -> _RawSnapshot:
    if not isinstance(raw, dict) or raw.get("$type") != "blueprinting.ir.snapshot" or set(raw) != {"$type", "fields"}:
        raise SerializationError("migration payload must be a canonical IR snapshot")
    fields = raw.get("fields")
    if not isinstance(fields, dict):
        raise SerializationError("migration snapshot fields must be an object")
    required = {"schema_name", "schema_version", "content_digest", "payload"}
    if not required.issubset(fields):
        raise SerializationError("migration snapshot is missing envelope fields")
    schema_name = fields["schema_name"]
    digest = fields["content_digest"]
    version = canonical_decode(fields["schema_version"])
    if not isinstance(schema_name, str) or not isinstance(digest, str) or not isinstance(version, SchemaVersion):
        raise SerializationError("migration snapshot has invalid envelope metadata")
    return _RawSnapshot(schema_name, version, digest, fields["payload"])


class SchemaMigrationRegistry:
    """Acyclic registry with unique-path migration resolution."""

    def __init__(self) -> None:
        self._steps: dict[tuple[str, SchemaVersion, SchemaVersion], SchemaMigration] = {}

    def register(self, migration: SchemaMigration) -> None:
        if not isinstance(migration, SchemaMigration):
            raise TypeError("migration must be SchemaMigration")
        key = (migration.schema_name, migration.from_version, migration.to_version)
        if key in self._steps:
            raise ValueError(f"duplicate schema migration edge: {key!r}")
        if any(
            item.schema_name == migration.schema_name and item.migration_id == migration.migration_id
            for item in self._steps.values()
        ):
            raise ValueError(f"duplicate schema migration ID: {migration.migration_id!r}")
        if self._reachable(migration.schema_name, migration.to_version, migration.from_version):
            raise ValueError("schema migration would introduce a cycle")
        self._steps[key] = migration

    def _outgoing(self, schema_name: str, version: SchemaVersion) -> tuple[SchemaMigration, ...]:
        return tuple(
            sorted(
                (
                    item
                    for item in self._steps.values()
                    if item.schema_name == schema_name and item.from_version == version
                ),
                key=lambda item: (item.to_version, item.migration_id),
            )
        )

    def _reachable(self, schema_name: str, source: SchemaVersion, target: SchemaVersion) -> bool:
        pending = [source]
        seen = set()
        while pending:
            current = pending.pop()
            if current == target:
                return True
            if current in seen:
                continue
            seen.add(current)
            pending.extend(item.to_version for item in self._outgoing(schema_name, current))
        return False

    def path(
        self,
        schema_name: str,
        source: SchemaVersion,
        target: SchemaVersion,
    ) -> tuple[SchemaMigration, ...]:
        if source == target:
            return ()
        paths: list[tuple[SchemaMigration, ...]] = []

        def visit(version: SchemaVersion, prefix: tuple[SchemaMigration, ...], seen: frozenset[SchemaVersion]) -> None:
            if len(paths) > 1:
                return
            for step in self._outgoing(schema_name, version):
                if step.to_version in seen:
                    continue
                candidate = prefix + (step,)
                if step.to_version == target:
                    paths.append(candidate)
                elif step.to_version < target:
                    visit(step.to_version, candidate, seen | {step.to_version})

        visit(source, (), frozenset({source}))
        if not paths:
            raise SerializationError(f"no migration path for {schema_name}@{source} -> {target}")
        if len(paths) != 1:
            raise SerializationError(f"ambiguous migration path for {schema_name}@{source} -> {target}")
        return paths[0]

    @staticmethod
    def _validated_snapshot(raw: Any, *, expected_schema: str, expected_version: SchemaVersion) -> _RawSnapshot:
        decoded = _raw_snapshot(raw)
        if decoded.schema_name != expected_schema or decoded.schema_version != expected_version:
            raise SerializationError(
                f"migration produced {decoded.schema_name}@{decoded.schema_version}, "
                f"expected {expected_schema}@{expected_version}"
            )
        expected_digest = raw_content_digest(decoded.payload, f"ir:{expected_schema}")
        if decoded.content_digest != expected_digest:
            raise SerializationError("canonical IR snapshot digest mismatch during migration")
        return decoded

    def migrate_json(
        self,
        payload: str,
        *,
        schema_name: str,
        target_version: SchemaVersion,
    ) -> MigrationResult:
        raw = canonical_parse(payload)
        decoded = _raw_snapshot(raw)
        if decoded.schema_name != schema_name:
            raise SerializationError(f"snapshot schema {decoded.schema_name!r} is not {schema_name!r}")
        source_version = decoded.schema_version
        source = self._validated_snapshot(raw, expected_schema=schema_name, expected_version=source_version)
        steps = self.path(schema_name, source_version, target_version)
        current_raw = raw
        current = source
        for step in steps:
            first = step.transform(copy.deepcopy(current_raw))
            second = step.transform(copy.deepcopy(current_raw))
            if canonical_dump_raw(first) != canonical_dump_raw(second):
                raise SerializationError(f"schema migration {step.migration_id!r} is nondeterministic")
            current_raw = first
            current = self._validated_snapshot(
                current_raw,
                expected_schema=schema_name,
                expected_version=step.to_version,
            )
        return MigrationResult(
            payload=canonical_dump_raw(current_raw),
            schema_name=schema_name,
            source_version=source_version,
            target_version=target_version,
            source_digest=source.content_digest,
            target_digest=current.content_digest,
            migration_ids=tuple(item.migration_id for item in steps),
        )


DEFAULT_SCHEMA_MIGRATIONS = SchemaMigrationRegistry()


__all__ = [
    "DEFAULT_SCHEMA_MIGRATIONS",
    "MigrationResult",
    "SchemaMigration",
    "SchemaMigrationRegistry",
]

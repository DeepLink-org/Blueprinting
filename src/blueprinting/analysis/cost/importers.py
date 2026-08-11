"""Strict tabular ingestion for simulator and profiler performance evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from blueprinting.schema.codec import content_digest
from blueprinting.schema.frozen import FrozenDict

from .database import EvidenceProvenance, PerformanceDatabase, PerformanceRecord
from .protocol import CostSubject, EstimateMethod


class LatencyUnit(Enum):
    SECONDS = "s"
    MILLISECONDS = "ms"
    MICROSECONDS = "us"
    NANOSECONDS = "ns"

    @property
    def seconds_multiplier(self) -> float:
        return {
            LatencyUnit.SECONDS: 1.0,
            LatencyUnit.MILLISECONDS: 1e-3,
            LatencyUnit.MICROSECONDS: 1e-6,
            LatencyUnit.NANOSECONDS: 1e-9,
        }[self]


_SCALAR_TYPES = frozenset({"string", "int", "float", "bool"})


@dataclass(frozen=True)
class TabularImportSpec:
    """Declarative mapping from one external table into normalized records."""

    name: str
    subject: CostSubject
    source: str
    source_revision: str
    method: EstimateMethod
    latency_column: str
    latency_unit: LatencyUnit
    operation: str | None = None
    operation_column: str | None = None
    hardware: str | None = None
    hardware_column: str | None = None
    datatype: str | None = None
    datatype_column: str | None = None
    selector_columns: FrozenDict = field(default_factory=FrozenDict)
    selector_types: FrozenDict = field(default_factory=FrozenDict)
    constant_selectors: FrozenDict = field(default_factory=FrozenDict)
    dimensions_json_column: str | None = None
    metadata_columns: tuple[str, ...] = ()
    record_id_column: str | None = None
    operation_aliases: FrozenDict = field(default_factory=FrozenDict)
    datatype_aliases: FrozenDict = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        for name in ("name", "source", "source_revision", "latency_column"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.subject, CostSubject):
            raise TypeError("subject must be CostSubject")
        if not isinstance(self.method, EstimateMethod):
            raise TypeError("method must be EstimateMethod")
        if not isinstance(self.latency_unit, LatencyUnit):
            raise TypeError("latency_unit must be LatencyUnit")
        for value_name, column_name in (
            ("operation", "operation_column"),
            ("hardware", "hardware_column"),
            ("datatype", "datatype_column"),
        ):
            value = getattr(self, value_name)
            column = getattr(self, column_name)
            if (value is None) == (column is None):
                raise ValueError(f"provide exactly one of {value_name} and {column_name}")
            for item, name in ((value, value_name), (column, column_name)):
                if item is not None and (not isinstance(item, str) or not item):
                    raise ValueError(f"{name} must be a non-empty string when present")
        for name in (
            "selector_columns",
            "selector_types",
            "constant_selectors",
            "operation_aliases",
            "datatype_aliases",
        ):
            object.__setattr__(self, name, FrozenDict(getattr(self, name)))
        object.__setattr__(self, "metadata_columns", tuple(self.metadata_columns))
        if any(not isinstance(item, str) or not item for item in self.metadata_columns):
            raise ValueError("metadata columns must be non-empty strings")
        for name in ("dimensions_json_column", "record_id_column"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{name} must be a non-empty string when present")
        if set(self.selector_columns) != set(self.selector_types):
            missing = sorted(set(self.selector_columns).symmetric_difference(self.selector_types))
            raise ValueError(f"selector columns and selector types must have identical keys: {missing}")
        invalid_types = {value for value in self.selector_types.values() if value not in _SCALAR_TYPES}
        if invalid_types:
            raise ValueError(f"unsupported selector types: {', '.join(sorted(invalid_types))}")
        collisions = set(self.selector_columns).intersection(self.constant_selectors)
        if collisions:
            raise ValueError(f"selector columns collide with constants: {', '.join(sorted(collisions))}")
        duplicate_identity = {"subject", "operation", "hardware", "datatype"}.intersection(
            set(self.selector_columns).union(self.constant_selectors)
        )
        if duplicate_identity:
            raise ValueError(f"selectors duplicate core identity: {', '.join(sorted(duplicate_identity))}")


def _python_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    return value


def read_tabular_rows(path: str | Path) -> tuple[dict[str, Any], ...]:
    """Read CSV, JSON/JSONL, or Parquet without changing source values."""

    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".csv":
        with source.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                raise ValueError(f"tabular evidence {source} has no CSV header")
            rows = tuple(dict(row) for row in reader)
    elif suffix in {".jsonl", ".ndjson"}:
        rows = tuple(json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip())
    elif suffix == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("JSON performance data must be an array of row objects")
        rows = tuple(payload)
    elif suffix == ".parquet":
        try:
            import pandas as pd

            frame = pd.read_parquet(source)
        except ImportError as error:
            raise ImportError(
                "Parquet import requires an engine such as pyarrow; install blueprinting[performance-data]"
            ) from error
        rows = tuple(frame.to_dict(orient="records"))
    else:
        raise ValueError(f"unsupported performance table format: {suffix or '<none>'}")
    if not rows:
        raise ValueError(f"tabular evidence {source} has no rows")
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("every performance table row must be an object")
    return tuple({str(key): _python_scalar(value) for key, value in row.items()} for row in rows)


def _required(row: dict[str, Any], column: str, row_number: int) -> Any:
    if column not in row:
        raise ValueError(f"row {row_number} is missing required column {column!r}")
    value = _python_scalar(row[column])
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(f"row {row_number} has an empty required value in {column!r}")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"row {row_number} has a non-finite value in {column!r}")
    return value


def _parse_scalar(value: Any, type_name: str, *, column: str, row_number: int) -> str | int | float | bool:
    value = _python_scalar(value)
    if type_name == "string":
        text = str(value)
        if not text:
            raise ValueError(f"row {row_number} has an empty string in {column!r}")
        return text
    if type_name == "bool":
        if isinstance(value, bool):
            return bool(value)
        normalized = str(value).strip().lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
            return False
        raise ValueError(f"row {row_number} has an invalid boolean in {column!r}")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"row {row_number} has a non-numeric value in {column!r}") from error
    if not math.isfinite(numeric):
        raise ValueError(f"row {row_number} has a non-finite value in {column!r}")
    if type_name == "float":
        return numeric
    if not numeric.is_integer():
        raise ValueError(f"row {row_number} has a non-integer value in {column!r}")
    return int(numeric)


def _identity(
    row: dict[str, Any],
    row_number: int,
    *,
    constant: str | None,
    column: str | None,
    aliases: FrozenDict | None = None,
) -> str:
    value = constant if constant is not None else _required(row, column or "", row_number)
    result = str(value).strip()
    if aliases is not None:
        result = aliases.get(result, result)
    if not isinstance(result, str) or not result:
        raise ValueError(f"row {row_number} resolves to an empty identity")
    return result


class TabularPerformanceImporter:
    """Import simulator/profiler tables using an explicit, revisioned schema."""

    IMPORTER_REVISION = "blueprinting-tabular-performance-v0"

    @classmethod
    def from_file(cls, path: str | Path, spec: TabularImportSpec) -> PerformanceDatabase:
        if not isinstance(spec, TabularImportSpec):
            raise TypeError("spec must be TabularImportSpec")
        source = Path(path)
        payload = source.read_bytes()
        data_digest = hashlib.sha256(payload).hexdigest()
        rows = read_tabular_rows(source)
        provenance = EvidenceProvenance(
            source=spec.source,
            source_revision=spec.source_revision,
            importer=cls.IMPORTER_REVISION,
            data_digest=data_digest,
            method=spec.method,
            metadata=FrozenDict({"file_name": source.name}),
        )
        records = []
        for row_number, row in enumerate(rows, start=1):
            latency = float(
                _parse_scalar(
                    _required(row, spec.latency_column, row_number),
                    "float",
                    column=spec.latency_column,
                    row_number=row_number,
                )
            )
            if latency < 0:
                raise ValueError(f"row {row_number} has negative latency")
            selector = spec.constant_selectors.to_dict()
            for name, column in spec.selector_columns.items():
                selector[name] = _parse_scalar(
                    _required(row, column, row_number),
                    spec.selector_types[name],
                    column=column,
                    row_number=row_number,
                )
            if spec.dimensions_json_column is not None:
                raw_dimensions = _required(row, spec.dimensions_json_column, row_number)
                dimensions = json.loads(raw_dimensions) if isinstance(raw_dimensions, str) else raw_dimensions
                if not isinstance(dimensions, dict):
                    raise ValueError(f"row {row_number} dimensions JSON must be an object")
                overlap = set(selector).intersection(dimensions)
                if overlap:
                    raise ValueError(f"row {row_number} dimensions collide with selectors: {sorted(overlap)}")
                selector.update(dimensions)
            metadata = {column: _required(row, column, row_number) for column in spec.metadata_columns}
            if spec.record_id_column is not None:
                record_id = str(_required(row, spec.record_id_column, row_number))
            else:
                record_id = content_digest(
                    FrozenDict(
                        {
                            "importer": cls.IMPORTER_REVISION,
                            "data_digest": data_digest,
                            "row": row_number,
                        }
                    ),
                    "performance-record-id",
                )
            records.append(
                PerformanceRecord(
                    record_id=record_id,
                    subject=spec.subject,
                    operation=_identity(
                        row,
                        row_number,
                        constant=spec.operation,
                        column=spec.operation_column,
                        aliases=spec.operation_aliases,
                    ),
                    hardware=_identity(
                        row,
                        row_number,
                        constant=spec.hardware,
                        column=spec.hardware_column,
                    ),
                    datatype=_identity(
                        row,
                        row_number,
                        constant=spec.datatype,
                        column=spec.datatype_column,
                        aliases=spec.datatype_aliases,
                    ),
                    seconds=latency * spec.latency_unit.seconds_multiplier,
                    selector=FrozenDict(selector),
                    provenance=provenance,
                    metadata=FrozenDict(metadata),
                )
            )
        return PerformanceDatabase(
            name=spec.name,
            records=tuple(records),
            metadata=FrozenDict(
                {
                    "source": spec.source,
                    "source_revision": spec.source_revision,
                    "data_digest": data_digest,
                    "importer": cls.IMPORTER_REVISION,
                    "file_name": source.name,
                }
            ),
        )


class SimulatorPerformanceImporter(TabularPerformanceImporter):
    """Named entry point for simulator outputs using ``TabularImportSpec``."""

    @classmethod
    def from_file(cls, path: str | Path, spec: TabularImportSpec) -> PerformanceDatabase:
        if spec.method is not EstimateMethod.SIMULATED:
            raise ValueError("simulator imports must declare method=EstimateMethod.SIMULATED")
        return super().from_file(path, spec)

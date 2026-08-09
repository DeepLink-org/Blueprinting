"""Immutable performance records and an exact-selector database provider."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from functools import cached_property

from blueprinting.schema.codec import canonical_dumps, canonical_loads, content_digest, record_type
from blueprinting.schema.frozen import FrozenDict

from .protocol import (
    CostEstimate,
    CostProvider,
    CostQuery,
    CostSubject,
    CostSupport,
    EstimateMatch,
    EstimateMethod,
    EstimateUncertainty,
    InvalidCostEvidenceError,
)

# Keep the legacy codec namespace as a stable serialized identity.


@record_type("compiler.analysis.cost.provenance.v1")
@dataclass(frozen=True)
class EvidenceProvenance:
    source: str
    source_revision: str
    importer: str
    data_digest: str
    method: EstimateMethod
    metadata: FrozenDict = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        for name in ("source", "source_revision", "importer", "data_digest"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.method, EstimateMethod):
            raise TypeError("method must be EstimateMethod")
        object.__setattr__(self, "metadata", FrozenDict(self.metadata))


@record_type("compiler.analysis.cost.performance_record.v1")
@dataclass(frozen=True)
class PerformanceRecord:
    record_id: str
    subject: CostSubject
    operation: str
    hardware: str
    datatype: str
    seconds: float
    selector: FrozenDict
    provenance: EvidenceProvenance
    metadata: FrozenDict = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        for name in ("record_id", "operation", "hardware", "datatype"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.subject, CostSubject):
            raise TypeError("subject must be CostSubject")
        if (
            isinstance(self.seconds, bool)
            or not isinstance(self.seconds, (int, float))
            or not math.isfinite(self.seconds)
            or self.seconds < 0
        ):
            raise ValueError("record seconds must be finite and non-negative")
        if not isinstance(self.provenance, EvidenceProvenance):
            raise TypeError("provenance must be EvidenceProvenance")
        object.__setattr__(self, "selector", FrozenDict(self.selector))
        object.__setattr__(self, "metadata", FrozenDict(self.metadata))
        duplicate_identity = {"subject", "operation", "hardware", "datatype"}.intersection(self.selector)
        if duplicate_identity:
            raise ValueError(f"record selector duplicates core identity: {', '.join(sorted(duplicate_identity))}")


@record_type("compiler.analysis.cost.performance_database.v1")
@dataclass(frozen=True)
class PerformanceDatabase:
    name: str
    records: tuple[PerformanceRecord, ...]
    metadata: FrozenDict = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("database name must be a non-empty string")
        object.__setattr__(self, "records", tuple(self.records))
        object.__setattr__(self, "metadata", FrozenDict(self.metadata))
        if not self.records or any(not isinstance(record, PerformanceRecord) for record in self.records):
            raise ValueError("performance database requires typed records")
        record_ids = tuple(record.record_id for record in self.records)
        if len(set(record_ids)) != len(record_ids):
            raise ValueError("performance database record IDs must be unique")

    @cached_property
    def revision(self) -> str:
        return content_digest(self, "performance-database")

    def to_json(self) -> str:
        return canonical_dumps(self)

    @classmethod
    def from_json(cls, payload: str) -> PerformanceDatabase:
        result = canonical_loads(payload)
        if not isinstance(result, cls):
            raise TypeError("payload does not contain a PerformanceDatabase")
        return result

    @classmethod
    def merge(cls, name: str, databases: tuple[PerformanceDatabase, ...]) -> PerformanceDatabase:
        databases = tuple(databases)
        if not databases:
            raise ValueError("at least one performance database is required")
        return cls(
            name=name,
            records=tuple(record for database in databases for record in database.records),
            metadata=FrozenDict(
                {
                    "merged_revisions": tuple(database.revision for database in databases),
                    "merged_names": tuple(database.name for database in databases),
                }
            ),
        )


class PerformanceDatabaseProvider(CostProvider):
    """Select the most-specific matching selector without interpolation."""

    def __init__(self, database: PerformanceDatabase) -> None:
        if not isinstance(database, PerformanceDatabase):
            raise TypeError("database must be PerformanceDatabase")
        self._database = database
        self._name = f"performance-db:{database.name}"
        self._revision = database.revision
        index = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for record in database.records:
            core = (record.subject, record.operation, record.hardware, record.datatype)
            selector_keys = tuple(record.selector)
            typed_values = tuple((type(record.selector[key]), record.selector[key]) for key in selector_keys)
            index[core][selector_keys][typed_values].append(record)
        self._index = {
            core: {
                selector_keys: {values: tuple(records) for values, records in value_index.items()}
                for selector_keys, value_index in selector_index.items()
            }
            for core, selector_index in index.items()
        }

    @property
    def database(self) -> PerformanceDatabase:
        return self._database

    @property
    def name(self) -> str:
        return self._name

    @property
    def revision(self) -> str:
        return self._revision

    def _selected_records(self, query: CostQuery) -> tuple[PerformanceRecord, ...] | None:
        core = (query.subject, query.operation, query.hardware, query.datatype)
        selector_index = self._index.get(core)
        if selector_index is None:
            return None
        context = query.match_context
        candidate_groups = []
        for selector_keys, value_index in selector_index.items():
            if any(key not in context for key in selector_keys):
                continue
            typed_values = tuple((type(context[key]), context[key]) for key in selector_keys)
            records = value_index.get(typed_values)
            if records is not None:
                candidate_groups.append(records)
        candidates = tuple(record for records in candidate_groups for record in records)
        if not candidates:
            return None
        specificity = max(len(record.selector) for record in candidates)
        candidates = tuple(record for record in candidates if len(record.selector) == specificity)
        groups: dict[tuple[object, ...], list[PerformanceRecord]] = defaultdict(list)
        for record in candidates:
            provenance = record.provenance
            key = (
                record.selector,
                provenance.source,
                provenance.source_revision,
                provenance.importer,
                provenance.method,
                provenance.data_digest,
            )
            groups[key].append(record)
        if len(groups) > 1:
            descriptions = sorted(
                f"{items[0].provenance.source}@{items[0].provenance.source_revision}:{dict(items[0].selector)}"
                for items in groups.values()
            )
            raise InvalidCostEvidenceError(
                "multiple equally specific evidence groups match the query: " + "; ".join(descriptions)
            )
        return tuple(next(iter(groups.values())))

    def supports(self, query: CostQuery) -> CostSupport:
        try:
            records = self._selected_records(query)
        except InvalidCostEvidenceError as error:
            return CostSupport.invalid(str(error))
        if records is None:
            return CostSupport.unavailable("no exact selector is covered by this database")
        return CostSupport.available(f"{len(records)} raw sample(s), selector specificity {len(records[0].selector)}")

    def estimate(self, query: CostQuery) -> CostEstimate:
        records = self._selected_records(query)
        if records is None:
            raise InvalidCostEvidenceError("database provider was asked to estimate an unsupported query")
        values = tuple(float(record.seconds) for record in records)
        seconds = statistics.median(values)
        deviation = statistics.pstdev(values) if len(values) > 1 else 0.0
        provenance = records[0].provenance
        selector = records[0].selector
        uncovered = tuple(sorted(set(query.match_context) - set(selector)))
        assumptions = ()
        if uncovered:
            assumptions = (
                "evidence matches an exact declared selector; dimensions not declared by the source are not "
                "claimed as controlled",
            )
        return CostEstimate(
            seconds=seconds,
            provider=self.name,
            provider_revision=self.revision,
            source_revision=provenance.source_revision,
            method=provenance.method,
            match=EstimateMatch.EXACT_SELECTOR,
            uncertainty=EstimateUncertainty(
                sample_count=len(values),
                standard_deviation_seconds=deviation,
                lower_bound_seconds=min(values),
                upper_bound_seconds=max(values),
            ),
            raw_record_ids=tuple(sorted(record.record_id for record in records)),
            validity_domain=selector,
            components=FrozenDict({"median_seconds": seconds}),
            assumptions=assumptions,
        )

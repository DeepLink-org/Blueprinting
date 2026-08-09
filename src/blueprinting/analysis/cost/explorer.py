"""Read-only derived views for inspecting and comparing performance evidence."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from blueprinting.schema.frozen import FrozenDict
from blueprinting.system import SystemProfile

from ..cost_model import CalibrationMode
from .database import PerformanceDatabase, PerformanceDatabaseProvider, PerformanceRecord
from .protocol import CostQuery, CostSubject
from .roofline import RooflineCostProvider


@dataclass(frozen=True)
class EvidenceCoverage:
    subject: str
    operation: str
    semantic_operation: str
    hardware: str
    datatype: str
    method: str
    record_count: int
    selector_axes: tuple[str, ...]


@dataclass(frozen=True)
class EvidenceDatabaseSummary:
    name: str
    revision: str
    source: str
    source_revision: str
    importer: str
    record_count: int
    hardware: tuple[str, ...]
    datatypes: tuple[str, ...]
    operations: tuple[str, ...]
    coverage: tuple[EvidenceCoverage, ...]


@dataclass(frozen=True)
class CostCurvePoint:
    x: int
    query_digest: str
    operations: int
    read_bytes: int
    write_bytes: int
    measured_seconds: float
    measured_lower_seconds: float | None
    measured_upper_seconds: float | None
    analytical_seconds: float
    measured_teraops_per_second: float
    analytical_teraops_per_second: float
    analytical_bottleneck: str
    relative_error_percent: float | None
    sample_count: int
    raw_record_ids: tuple[str, ...]


@dataclass(frozen=True)
class CostCurveReport:
    database_name: str
    database_revision: str
    analytical_provider: str
    analytical_revision: str
    operation: str
    semantic_operation: str
    hardware: str
    datatype: str
    axis: str
    fixed_selectors: FrozenDict
    points: tuple[CostCurvePoint, ...]
    limitations: tuple[str, ...]


def summarize_performance_database(database: PerformanceDatabase) -> EvidenceDatabaseSummary:
    """Build a stable catalog/coverage view without changing evidence records."""

    if not isinstance(database, PerformanceDatabase):
        raise TypeError("database must be PerformanceDatabase")
    groups: dict[tuple[str, ...], list[PerformanceRecord]] = defaultdict(list)
    for record in database.records:
        semantic = str(record.selector.get("semantic_operation", record.operation))
        key = (
            record.subject.value,
            record.operation,
            semantic,
            record.hardware,
            record.datatype,
            record.provenance.method.value,
        )
        groups[key].append(record)
    coverage = tuple(
        EvidenceCoverage(
            subject=key[0],
            operation=key[1],
            semantic_operation=key[2],
            hardware=key[3],
            datatype=key[4],
            method=key[5],
            record_count=len(records),
            selector_axes=tuple(sorted({name for record in records for name in record.selector})),
        )
        for key, records in sorted(groups.items())
    )
    provenances = tuple(record.provenance for record in database.records)
    return EvidenceDatabaseSummary(
        name=database.name,
        revision=database.revision,
        source=_single_value(tuple(item.source for item in provenances), "source"),
        source_revision=_single_value(tuple(item.source_revision for item in provenances), "source_revision"),
        importer=_single_value(tuple(item.importer for item in provenances), "importer"),
        record_count=len(database.records),
        hardware=tuple(sorted({record.hardware for record in database.records})),
        datatypes=tuple(sorted({record.datatype for record in database.records})),
        operations=tuple(sorted({record.operation for record in database.records})),
        coverage=coverage,
    )


def comparable_gemm_semantics(database: PerformanceDatabase) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                str(record.selector["semantic_operation"])
                for record in database.records
                if record.operation == "gemm" and "semantic_operation" in record.selector
            }
        )
    )


def build_gemm_comparison_curve(
    database: PerformanceDatabase,
    hardware: SystemProfile,
    semantic_operation: str,
    *,
    mode: CalibrationMode = CalibrationMode.SYSTEM_EVIDENCE,
) -> CostCurveReport:
    """Compare exact GEMM records with an analytical roofline on identical facts."""

    if not isinstance(database, PerformanceDatabase):
        raise TypeError("database must be PerformanceDatabase")
    if not isinstance(hardware, SystemProfile):
        raise TypeError("hardware must be SystemProfile")
    if not isinstance(mode, CalibrationMode):
        raise TypeError("mode must be CalibrationMode")
    records = tuple(
        record
        for record in database.records
        if record.subject is CostSubject.OPERATOR
        and record.operation == "gemm"
        and record.hardware == hardware.name
        and record.datatype == hardware.datatype
        and record.selector.get("semantic_operation") == semantic_operation
    )
    if not records:
        raise ValueError(f"database has no comparable GEMM records for {semantic_operation!r}")
    if any("num_tokens" not in record.selector for record in records):
        raise ValueError("comparable GEMM records require a num_tokens selector")
    fixed_selectors = _fixed_selectors(records, axis="num_tokens")
    database_provider = PerformanceDatabaseProvider(database)
    analytical_provider = RooflineCostProvider(hardware, mode=mode, processing_mode="roofline")
    points = []
    seen_tokens: set[int] = set()
    for record in sorted(records, key=lambda item: int(item.selector["num_tokens"])):
        tokens = int(record.selector["num_tokens"])
        if tokens in seen_tokens:
            continue
        seen_tokens.add(tokens)
        m, n, k = _gemm_shape(record.selector, semantic_operation)
        element_bytes = _datatype_bytes(record.datatype)
        operations = 2 * m * n * k
        read_bytes = (m * n + n * k) * element_bytes
        write_bytes = m * k * element_bytes
        query = CostQuery(
            subject=CostSubject.OPERATOR,
            operation="gemm",
            hardware=record.hardware,
            datatype=record.datatype,
            operations=operations,
            read_bytes=read_bytes,
            write_bytes=write_bytes,
            engine="matrix",
            hardware_revision=hardware.evidence_revision,
            dimensions=record.selector,
        )
        measured = database_provider.estimate(query)
        analytical = analytical_provider.estimate(query)
        measured_rate = operations / measured.seconds / 1e12 if measured.seconds else 0.0
        analytical_rate = operations / analytical.seconds / 1e12 if analytical.seconds else 0.0
        relative_error = None
        if measured.seconds:
            relative_error = (analytical.seconds - measured.seconds) / measured.seconds * 100
        points.append(
            CostCurvePoint(
                x=tokens,
                query_digest=query.digest,
                operations=operations,
                read_bytes=read_bytes,
                write_bytes=write_bytes,
                measured_seconds=measured.seconds,
                measured_lower_seconds=measured.uncertainty.lower_bound_seconds,
                measured_upper_seconds=measured.uncertainty.upper_bound_seconds,
                analytical_seconds=analytical.seconds,
                measured_teraops_per_second=measured_rate,
                analytical_teraops_per_second=analytical_rate,
                analytical_bottleneck=str(analytical.components["bottleneck"]),
                relative_error_percent=relative_error,
                sample_count=measured.uncertainty.sample_count,
                raw_record_ids=measured.raw_record_ids,
            )
        )
    return CostCurveReport(
        database_name=database.name,
        database_revision=database.revision,
        analytical_provider=analytical_provider.name,
        analytical_revision=analytical_provider.revision,
        operation="gemm",
        semantic_operation=semantic_operation,
        hardware=hardware.name,
        datatype=hardware.datatype,
        axis="num_tokens",
        fixed_selectors=FrozenDict(fixed_selectors),
        points=tuple(points),
        limitations=(
            "Measured points are exact imported selectors; connecting lines do not claim interpolation.",
            "The analytical series is a compute/memory roofline bound, not an event-level simulation.",
            "This PoC compares GEMM primitives only because their exact M/N/K workload can be reconstructed.",
        ),
    )


def _single_value(values: tuple[str, ...], name: str) -> str:
    unique = tuple(sorted(set(values)))
    if len(unique) != 1:
        raise ValueError(f"database contains multiple {name} values: {unique}")
    return unique[0]


def _fixed_selectors(records: tuple[PerformanceRecord, ...], *, axis: str) -> dict[str, object]:
    keys = set(records[0].selector) - {axis}
    result: dict[str, object] = {}
    for key in sorted(keys):
        values = {record.selector.get(key) for record in records}
        if len(values) != 1:
            raise ValueError(f"selector {key!r} varies within one comparison curve")
        result[key] = next(iter(values))
    return result


def _gemm_shape(selector: FrozenDict, semantic_operation: str) -> tuple[int, int, int]:
    tokens = int(selector["num_tokens"])
    hidden = int(selector["hidden_size"])
    feedforward = int(selector["feedforward_size"])
    shapes = {
        "attention_pre_projection": (tokens, hidden, 3 * hidden),
        "attention_post_projection": (tokens, hidden, hidden),
        "mlp_up_projection": (tokens, hidden, feedforward),
        "mlp_down_projection": (tokens, feedforward, hidden),
    }
    try:
        return shapes[semantic_operation]
    except KeyError as error:
        raise ValueError(f"unsupported comparable GEMM semantic: {semantic_operation!r}") from error


def _datatype_bytes(datatype: str) -> int:
    sizes = {"float8": 1, "float16": 2, "bfloat16": 2, "float32": 4}
    try:
        return sizes[datatype]
    except KeyError as error:
        raise ValueError(f"unsupported datatype size: {datatype!r}") from error


__all__ = [
    "CostCurvePoint",
    "CostCurveReport",
    "EvidenceCoverage",
    "EvidenceDatabaseSummary",
    "build_gemm_comparison_curve",
    "comparable_gemm_semantics",
    "summarize_performance_database",
]

from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprinting.analysis import (
    PerformanceDatabase,
    VidurProfileImporter,
    build_gemm_comparison_curve,
    comparable_gemm_semantics,
    summarize_performance_database,
)
from blueprinting.system import SystemProfile

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "data" / "validation" / "vidur" / "phi2_a100_tp1"
SOURCE_REVISION = "8383d2935bc62723a212090baa9f98ada206fc14"


def _database() -> PerformanceDatabase:
    return VidurProfileImporter.from_csv(
        attention_csv=PROFILE / "attention.csv",
        compute_csv=PROFILE / "mlp.csv",
        model_name="microsoft/phi-2",
        hardware_name="a100_80g",
        source_revision=SOURCE_REVISION,
        database_name="vidur-phi2-a100-poc",
    )


def _hardware() -> SystemProfile:
    payload = json.loads((ROOT / "data" / "systems" / "a100_80g.json").read_text(encoding="utf-8"))
    return SystemProfile.from_mapping("a100_80g", payload, datatype="float16")


def test_evidence_catalog_reports_typed_coverage_and_provenance() -> None:
    summary = summarize_performance_database(_database())

    assert summary.record_count == 20
    assert summary.hardware == ("a100_80g",)
    assert summary.datatypes == ("float16",)
    assert summary.source == "vidur-profile"
    assert summary.source_revision == SOURCE_REVISION
    assert {item.semantic_operation for item in summary.coverage} >= {
        "attention_core",
        "attention_pre_projection",
        "mlp_up_projection",
    }


def test_gemm_curve_compares_identical_facts_without_mutating_database() -> None:
    database = _database()
    before = database.to_json()

    report = build_gemm_comparison_curve(database, _hardware(), "mlp_up_projection")

    assert [point.x for point in report.points] == [1, 128]
    assert all(point.operations > 0 and point.read_bytes > 0 and point.write_bytes > 0 for point in report.points)
    assert all(point.measured_seconds > 0 and point.analytical_seconds > 0 for point in report.points)
    assert all(point.raw_record_ids and point.query_digest for point in report.points)
    assert len({point.query_digest for point in report.points}) == 2
    assert database.to_json() == before


def test_evidence_curve_is_stable_after_database_round_trip() -> None:
    database = _database()
    restored = PerformanceDatabase.from_json(database.to_json())

    original = build_gemm_comparison_curve(database, _hardware(), "attention_post_projection")
    round_trip = build_gemm_comparison_curve(restored, _hardware(), "attention_post_projection")

    assert restored.revision == database.revision
    assert round_trip == original


def test_evidence_curve_rejects_an_uncovered_semantic() -> None:
    database = _database()

    assert comparable_gemm_semantics(database) == (
        "attention_post_projection",
        "attention_pre_projection",
        "mlp_down_projection",
        "mlp_up_projection",
    )
    with pytest.raises(ValueError, match="no comparable GEMM records"):
        build_gemm_comparison_curve(database, _hardware(), "attention_core")

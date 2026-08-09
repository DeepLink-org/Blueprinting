from __future__ import annotations

from blueprinting.analysis import build_gemm_comparison_curve
from blueprinting.workbench.catalog import default_catalog
from blueprinting.workbench.evidence_lab import (
    coverage_chart_options,
    coverage_rows,
    curve_rows,
    latency_curve_options,
    load_vidur_poc_evidence,
    throughput_curve_options,
)


def test_evidence_lab_builds_catalog_and_comparison_presentations() -> None:
    data = load_vidur_poc_evidence(default_catalog())
    report = build_gemm_comparison_curve(data.database, data.hardware, "mlp_up_projection")

    coverage = coverage_rows(data.summary)
    latency = latency_curve_options(report)
    throughput = throughput_curve_options(report)
    rows = curve_rows(report)

    assert len(coverage) == 9
    assert {series["name"] for series in latency["series"]} == {"Vidur measured", "Analytical roofline"}
    assert {series["name"] for series in throughput["series"]} == {"Vidur effective", "Roofline effective"}
    assert [row["num_tokens"] for row in rows] == [1, 128]
    assert all(row["query_digest"] and row["record_ids"] for row in rows)
    assert coverage_chart_options(data.summary)["series"][0]["name"] == "Imported records"

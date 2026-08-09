from __future__ import annotations

from blueprinting.analysis import build_gemm_comparison_curve
from blueprinting.workbench.catalog import default_catalog
from blueprinting.workbench.evidence_lab import (
    coverage_chart_options,
    coverage_rows,
    curve_rows,
    latency_curve_options,
    load_vidur_poc_evidence,
    relative_error_chart_options,
    summarize_comparison,
    throughput_curve_options,
)


def test_evidence_lab_builds_catalog_and_comparison_presentations() -> None:
    data = load_vidur_poc_evidence(default_catalog())
    report = build_gemm_comparison_curve(data.database, data.hardware, "mlp_up_projection")

    coverage = coverage_rows(data.summary)
    latency = latency_curve_options(report)
    throughput = throughput_curve_options(report)
    relative_error = relative_error_chart_options(report)
    comparison = summarize_comparison(report)
    rows = curve_rows(report)

    assert len(coverage) == 9
    assert {series["name"] for series in latency["series"]} == {"Vidur measured", "Analytical roofline"}
    assert {series["name"] for series in throughput["series"]} == {"Vidur effective", "Roofline effective"}
    assert relative_error["series"][0]["name"] == "Roofline error"
    assert comparison.point_count == 2
    assert comparison.total_samples == 2
    assert comparison.token_min == 1
    assert comparison.token_max == 128
    assert comparison.bias == "Roofline 全部偏乐观"
    assert comparison.worst_error_tokens == 128
    assert [row["num_tokens"] for row in rows] == [1, 128]
    assert all(row["query_digest"] and row["record_ids"] for row in rows)
    assert coverage_chart_options(data.summary)["series"][0]["name"] == "Imported records"

"""Read-only performance-evidence catalog and comparison PoC for NiceGUI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nicegui import ui

from blueprinting.analysis import (
    CalibrationMode,
    CostCurveReport,
    EvidenceDatabaseSummary,
    PerformanceDatabase,
    VidurProfileImporter,
    build_gemm_comparison_curve,
    comparable_gemm_semantics,
    summarize_performance_database,
)
from blueprinting.system import SystemProfile

from .catalog import ConfigCatalog

_SEMANTIC_LABELS = {
    "attention_pre_projection": "Attention QKV projection",
    "attention_post_projection": "Attention output projection",
    "mlp_up_projection": "MLP up projection",
    "mlp_down_projection": "MLP down projection",
}


@dataclass(frozen=True)
class EvidenceLabData:
    database: PerformanceDatabase
    hardware: SystemProfile
    summary: EvidenceDatabaseSummary
    manifest: dict[str, Any]


@dataclass(frozen=True)
class ComparisonSummary:
    point_count: int
    total_samples: int
    token_min: int
    token_max: int
    peak_measured_tops: float
    mean_absolute_error_percent: float
    worst_error_percent: float
    worst_error_tokens: int
    bias: str
    bottlenecks: tuple[str, ...]


def load_vidur_poc_evidence(catalog: ConfigCatalog, root: Path | None = None) -> EvidenceLabData:
    """Load the explicitly pinned repository/package Vidur validation slice."""

    evidence_root = root or _default_vidur_root()
    manifest_path = evidence_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = manifest["source"]
    selection = manifest["selection"]
    blueprinting = manifest["blueprinting"]
    hardware_name = str(blueprinting["hardware"]["name"])
    datatype = str(selection["datatype"])
    database = VidurProfileImporter.from_csv(
        attention_csv=evidence_root / "attention.csv",
        compute_csv=evidence_root / "mlp.csv",
        model_name=str(selection["model"]),
        hardware_name=hardware_name,
        source_revision=str(source["revision"]),
        datatype=datatype,
        database_name="vidur-phi2-a100-poc",
    )
    hardware = SystemProfile.from_mapping(
        hardware_name,
        catalog.load("systems", f"{hardware_name}.json"),
        datatype=datatype,
    )
    return EvidenceLabData(database, hardware, summarize_performance_database(database), manifest)


def coverage_rows(summary: EvidenceDatabaseSummary) -> list[dict[str, Any]]:
    return [
        {
            "semantic_operation": item.semantic_operation,
            "operation": item.operation,
            "subject": item.subject,
            "records": item.record_count,
            "hardware": item.hardware,
            "datatype": item.datatype,
            "method": item.method,
            "axes": ", ".join(item.selector_axes),
        }
        for item in summary.coverage
    ]


def coverage_chart_options(summary: EvidenceDatabaseSummary) -> dict[str, Any]:
    rows = sorted(coverage_rows(summary), key=lambda item: (item["records"], item["semantic_operation"]))
    return {
        "backgroundColor": "transparent",
        "animationDuration": 250,
        "grid": {"left": 172, "right": 30, "top": 18, "bottom": 38},
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "xAxis": {"type": "value", "name": "records", "minInterval": 1},
        "yAxis": {
            "type": "category",
            "data": [item["semantic_operation"] for item in rows],
            "axisLabel": {"width": 158, "overflow": "truncate"},
        },
        "series": [
            {
                "name": "Imported records",
                "type": "bar",
                "barMaxWidth": 16,
                "itemStyle": {"color": "#2563eb", "borderRadius": [0, 4, 4, 0]},
                "data": [item["records"] for item in rows],
            }
        ],
    }


def latency_curve_options(report: CostCurveReport) -> dict[str, Any]:
    return {
        "backgroundColor": "transparent",
        "animationDuration": 250,
        "legend": {"top": 0, "data": ["Vidur measured", "Analytical roofline"]},
        "grid": {"left": 68, "right": 28, "top": 50, "bottom": 48},
        "tooltip": {"trigger": "axis"},
        "xAxis": {"type": "log", "name": "num_tokens", "min": 1},
        "yAxis": {"type": "value", "name": "latency (µs)", "min": 0},
        "series": [
            {
                "name": "Vidur measured",
                "type": "scatter",
                "symbolSize": 10,
                "itemStyle": {"color": "#2563eb"},
                "data": [[point.x, point.measured_seconds * 1e6] for point in report.points],
            },
            {
                "name": "Analytical roofline",
                "type": "line",
                "showSymbol": True,
                "lineStyle": {"color": "#d97706", "width": 2},
                "itemStyle": {"color": "#d97706"},
                "data": [[point.x, point.analytical_seconds * 1e6] for point in report.points],
            },
        ],
    }


def throughput_curve_options(report: CostCurveReport) -> dict[str, Any]:
    return {
        "backgroundColor": "transparent",
        "animationDuration": 250,
        "legend": {"top": 0, "data": ["Vidur effective", "Roofline effective"]},
        "grid": {"left": 68, "right": 28, "top": 50, "bottom": 48},
        "tooltip": {"trigger": "axis"},
        "xAxis": {"type": "log", "name": "num_tokens", "min": 1},
        "yAxis": {"type": "value", "name": "effective TOPS", "min": 0},
        "series": [
            {
                "name": "Vidur effective",
                "type": "scatter",
                "symbolSize": 10,
                "itemStyle": {"color": "#0891b2"},
                "data": [[point.x, point.measured_teraops_per_second] for point in report.points],
            },
            {
                "name": "Roofline effective",
                "type": "line",
                "showSymbol": True,
                "lineStyle": {"color": "#7c3aed", "width": 2},
                "itemStyle": {"color": "#7c3aed"},
                "data": [[point.x, point.analytical_teraops_per_second] for point in report.points],
            },
        ],
    }


def relative_error_chart_options(report: CostCurveReport) -> dict[str, Any]:
    return {
        "backgroundColor": "transparent",
        "animationDuration": 250,
        "grid": {"left": 58, "right": 18, "top": 24, "bottom": 44},
        "tooltip": {"trigger": "axis"},
        "xAxis": {"type": "category", "name": "tokens", "data": [point.x for point in report.points]},
        "yAxis": {"type": "value", "name": "error %"},
        "series": [
            {
                "name": "Roofline error",
                "type": "bar",
                "barMaxWidth": 34,
                "data": [
                    {
                        "value": point.relative_error_percent,
                        "itemStyle": {"color": "#d97706" if (point.relative_error_percent or 0) < 0 else "#0891b2"},
                    }
                    for point in report.points
                ],
                "markLine": {
                    "silent": True,
                    "symbol": "none",
                    "lineStyle": {"color": "#94a3b8", "type": "dashed"},
                    "data": [{"yAxis": 0}],
                },
            }
        ],
    }


def summarize_comparison(report: CostCurveReport) -> ComparisonSummary:
    if not report.points:
        raise ValueError("comparison report must contain at least one point")
    errors = tuple(point.relative_error_percent or 0.0 for point in report.points)
    worst = max(report.points, key=lambda point: abs(point.relative_error_percent or 0.0))
    if all(error < 0 for error in errors):
        bias = "Roofline 全部偏乐观"
    elif all(error > 0 for error in errors):
        bias = "Roofline 全部偏保守"
    else:
        bias = "Roofline 偏差方向不一致"
    return ComparisonSummary(
        point_count=len(report.points),
        total_samples=sum(point.sample_count for point in report.points),
        token_min=min(point.x for point in report.points),
        token_max=max(point.x for point in report.points),
        peak_measured_tops=max(point.measured_teraops_per_second for point in report.points),
        mean_absolute_error_percent=sum(abs(error) for error in errors) / len(errors),
        worst_error_percent=worst.relative_error_percent or 0.0,
        worst_error_tokens=worst.x,
        bias=bias,
        bottlenecks=tuple(sorted({point.analytical_bottleneck for point in report.points})),
    )


def curve_rows(report: CostCurveReport) -> list[dict[str, Any]]:
    return [
        {
            "num_tokens": point.x,
            "measured_us": round(point.measured_seconds * 1e6, 4),
            "roofline_us": round(point.analytical_seconds * 1e6, 4),
            "relative_error_percent": (
                round(point.relative_error_percent, 2) if point.relative_error_percent is not None else None
            ),
            "measured_tops": round(point.measured_teraops_per_second, 3),
            "roofline_tops": round(point.analytical_teraops_per_second, 3),
            "bottleneck": point.analytical_bottleneck,
            "samples": point.sample_count,
            "query_digest": point.query_digest,
            "record_ids": ", ".join(point.raw_record_ids),
        }
        for point in report.points
    ]


class EvidenceLabPanel:
    """Stateful NiceGUI panel around immutable evidence-derived views."""

    def __init__(self, catalog: ConfigCatalog) -> None:
        self.catalog = catalog
        self.data: EvidenceLabData | None = None
        self.semantic_operation = "attention_pre_projection"
        self.content: Any | None = None
        self.error: str | None = None

    def build(self) -> None:
        try:
            self.data = load_vidur_poc_evidence(self.catalog)
            semantics = comparable_gemm_semantics(self.data.database)
            if self.semantic_operation not in semantics:
                self.semantic_operation = semantics[0]
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            self.error = f"{type(error).__name__}: {error}"
        with ui.column().classes("w-full gap-3").mark("evidence-lab"):
            with (
                ui.element("section").classes("bp-evidence-surface"),
                ui.element("div").classes("bp-evidence-toolbar"),
            ):
                self._render_heading()
                if self.error is None and self.data is not None:
                    semantics = comparable_gemm_semantics(self.data.database)
                    with ui.element("div").classes("bp-evidence-toolbar-controls"):
                        ui.select(
                            {item: _SEMANTIC_LABELS.get(item, item) for item in semantics},
                            value=self.semantic_operation,
                            label="对比 Primitive",
                            on_change=self._semantic_changed,
                        ).props("outlined dense").classes("bp-evidence-selector").mark("evidence-semantic-operation")
                        ui.label("EXACT · NO INTERPOLATION").classes("bp-fidelity-tag bp-mono")
            if self.error is not None or self.data is None:
                with (
                    ui.element("section").classes("bp-evidence-surface"),
                    ui.element("div").classes("bp-evidence-section"),
                ):
                    ui.label("证据数据不可用").classes("bp-card-title")
                    ui.label(self.error or "unknown evidence loading error").classes("bp-card-copy bp-mono")
                return
            self.content = ui.column().classes("w-full gap-3")
            self._render_content()

    @staticmethod
    def _render_heading() -> None:
        with ui.column().classes("bp-evidence-heading gap-1"):
            ui.label("EVIDENCE LENS").classes("bp-kicker")
            ui.label("性能证据实验室").classes("bp-result-title")
            ui.label("同一 workload facts 下比较实测证据与解析模型。 ").classes("bp-card-copy")

    def _semantic_changed(self, event: Any) -> None:
        self.semantic_operation = str(event.value)
        self._render_content()

    def _render_content(self) -> None:
        if self.content is None or self.data is None:
            return
        self.content.clear()
        report = build_gemm_comparison_curve(
            self.data.database,
            self.data.hardware,
            self.semantic_operation,
            mode=CalibrationMode.SYSTEM_EVIDENCE,
        )
        summary = self.data.summary
        source = self.data.manifest["source"]
        comparison = summarize_comparison(report)
        fixed = report.fixed_selectors
        with self.content:
            with ui.element("section").classes("bp-evidence-surface"):
                with ui.element("div").classes("bp-evidence-section bp-evidence-summary-head"):
                    with ui.row().classes("items-center gap-2 no-wrap"):
                        ui.label(_SEMANTIC_LABELS.get(self.semantic_operation, self.semantic_operation)).classes(
                            "bp-card-title"
                        )
                        ui.label(f"{comparison.point_count} exact points").classes("bp-fidelity-tag bp-mono")
                    ui.label(
                        f"Phi-2 · {report.hardware} · {report.datatype} · "
                        f"M={comparison.token_min}–{comparison.token_max} tokens · "
                        f"H={fixed['hidden_size']} · FFN={fixed['feedforward_size']}"
                    ).classes("bp-card-copy bp-mono")
                with ui.element("div").classes("bp-evidence-metrics"):
                    for label, value, detail in (
                        ("EXACT POINTS", str(comparison.point_count), f"{comparison.total_samples} total samples"),
                        ("TOKEN DOMAIN", f"{comparison.token_min}–{comparison.token_max}", "discrete selectors"),
                        ("PEAK MEASURED", f"{comparison.peak_measured_tops:.1f} TOPS", "within imported points"),
                        ("MEAN |ERROR|", f"{comparison.mean_absolute_error_percent:.1f}%", "roofline vs measured"),
                        ("BOUND", " / ".join(comparison.bottlenecks), "analytical bottleneck"),
                    ):
                        with ui.element("div").classes("bp-evidence-metric"):
                            ui.label(label).classes("bp-summary-label")
                            ui.label(value).classes("bp-evidence-metric-value")
                            ui.label(detail).classes("bp-metric-detail")
                with ui.element("div").classes("bp-evidence-insight"):
                    ui.icon("warning_amber", size="19px")
                    ui.label(
                        f"{comparison.bias}；最大偏差 {comparison.worst_error_percent:+.1f}% "
                        f"出现在 {comparison.worst_error_tokens} tokens。当前点数不足以支持插值。"
                    ).classes("bp-card-copy")
            with ui.element("section").classes("bp-evidence-surface"):
                with ui.element("div").classes("bp-evidence-section bp-evidence-compact-head"):
                    ui.label("Measured vs analytical").classes("bp-card-title")
                    ui.label("散点是 Vidur exact records；连线只帮助阅读，不表示中间点已有证据。 ").classes(
                        "bp-card-copy"
                    )
                with ui.element("div").classes("bp-evidence-chart-grid"):
                    with ui.element("section").classes("bp-evidence-chart"):
                        ui.label("Latency · µs").classes("bp-section-title")
                        ui.echart(latency_curve_options(report), renderer="svg").classes(
                            "w-full bp-evidence-chart-canvas"
                        )
                    with ui.element("section").classes("bp-evidence-chart"):
                        ui.label("Effective throughput · TOPS").classes("bp-section-title")
                        ui.echart(throughput_curve_options(report), renderer="svg").classes(
                            "w-full bp-evidence-chart-canvas"
                        )
                    with ui.element("section").classes("bp-evidence-chart"):
                        ui.label("Roofline relative error").classes("bp-section-title")
                        ui.echart(relative_error_chart_options(report), renderer="svg").classes(
                            "w-full bp-evidence-chart-canvas"
                        )
                with ui.element("div").classes("bp-evidence-section bp-evidence-table-section"):
                    ui.aggrid(
                        {
                            "columnDefs": [
                                {"headerName": "Tokens", "field": "num_tokens", "pinned": "left", "width": 90},
                                {"headerName": "Measured µs", "field": "measured_us", "type": "numericColumn"},
                                {"headerName": "Roofline µs", "field": "roofline_us", "type": "numericColumn"},
                                {"headerName": "Error %", "field": "relative_error_percent", "type": "numericColumn"},
                                {"headerName": "Measured TOPS", "field": "measured_tops", "type": "numericColumn"},
                                {"headerName": "Bound", "field": "bottleneck"},
                                {"headerName": "Samples", "field": "samples", "type": "numericColumn"},
                                {"headerName": "Record IDs", "field": "record_ids", "flex": 1},
                                {"headerName": "Query digest", "field": "query_digest", "hide": True},
                            ],
                            "rowData": curve_rows(report),
                            "defaultColDef": {"sortable": True, "filter": True, "resizable": True},
                        },
                        theme="quartz",
                    ).classes("w-full").style("height: 186px").mark("evidence-query-inspector")
            with ui.element("section").classes("bp-evidence-surface"):
                with ui.element("div").classes("bp-evidence-section bp-evidence-compact-head"):
                    ui.label("Operation coverage").classes("bp-card-title")
                    ui.label("导入记录的离散覆盖；不暗示未测 selector 可插值。 ").classes("bp-card-copy")
                with ui.element("div").classes("bp-evidence-coverage-grid"):
                    with ui.element("section").classes("bp-evidence-chart"):
                        ui.echart(coverage_chart_options(summary), renderer="svg").classes(
                            "w-full bp-evidence-coverage-canvas"
                        )
                    with ui.element("section").classes("bp-evidence-coverage-table"):
                        ui.aggrid(
                            {
                                "columnDefs": [
                                    {"headerName": "Operation", "field": "semantic_operation", "pinned": "left"},
                                    {"headerName": "Family", "field": "operation"},
                                    {"headerName": "Records", "field": "records", "type": "numericColumn"},
                                    {"headerName": "Selectors", "field": "axes", "flex": 1},
                                ],
                                "rowData": coverage_rows(summary),
                                "defaultColDef": {"sortable": True, "filter": True, "resizable": True},
                            },
                            theme="quartz",
                        ).classes("w-full").style("height: 276px")
            with ui.element("section").classes("bp-evidence-surface bp-evidence-footer-grid"):
                with ui.element("div").classes("bp-evidence-section"):
                    ui.label("Evidence catalog").classes("bp-card-title")
                    with ui.element("div").classes("bp-evidence-catalog-grid"):
                        for label, value in (
                            ("Database", summary.name),
                            ("Records", str(summary.record_count)),
                            ("Hardware", ", ".join(summary.hardware)),
                            ("Datatype", ", ".join(summary.datatypes)),
                            ("Method", summary.coverage[0].method),
                            ("Source revision", summary.source_revision[:12]),
                        ):
                            with ui.column().classes("gap-0 min-w-0"):
                                ui.label(label).classes("bp-summary-label")
                                ui.label(value).classes("bp-card-copy bp-mono")
                    ui.link("Vidur source", source["repository"], new_tab=True).classes("bp-card-copy")
                    ui.label(f"License {source['license']} · DB {summary.revision[:16]}").classes(
                        "bp-card-copy bp-mono"
                    )
                with ui.element("div").classes("bp-evidence-section"):
                    ui.label("适用边界").classes("bp-card-title")
                    for limitation in report.limitations:
                        with ui.row().classes("items-start gap-2 no-wrap"):
                            ui.icon("info", size="17px", color="secondary")
                            ui.label(limitation).classes("bp-card-copy")


def _default_vidur_root() -> Path:
    package_root = Path(__file__).resolve().parents[1] / "presets" / "evidence" / "vidur" / "phi2_a100_tp1"
    if package_root.is_dir():
        return package_root
    repository_root = Path(__file__).resolve().parents[3]
    source_root = repository_root / "data" / "validation" / "vidur" / "phi2_a100_tp1"
    if source_root.is_dir():
        return source_root
    raise FileNotFoundError("the pinned Vidur Phi-2/A100 evidence slice is not installed")


__all__ = [
    "EvidenceLabData",
    "EvidenceLabPanel",
    "ComparisonSummary",
    "coverage_chart_options",
    "coverage_rows",
    "curve_rows",
    "latency_curve_options",
    "load_vidur_poc_evidence",
    "relative_error_chart_options",
    "summarize_comparison",
    "throughput_curve_options",
]

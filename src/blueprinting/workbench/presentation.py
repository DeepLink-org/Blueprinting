"""Framework-neutral presentation models for interactive Blueprinting clients.

The application service owns analysis semantics.  This module only converts its
immutable reports into display-oriented values, rows, and chart specifications.
Keeping these adapters free of NiceGUI makes them cheap to unit test and reusable
by future CLI, notebook, or agent surfaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from blueprinting.application import AnalysisOutcome, AnalysisReport, SweepReport


@dataclass(frozen=True)
class MetricView:
    """One compact headline fact with optional supporting context."""

    label: str
    value: str
    detail: str
    tone: str = "neutral"


def format_seconds(value: float) -> str:
    if value >= 1:
        return f"{value:.3f} s"
    if value >= 1e-3:
        return f"{value * 1e3:.3f} ms"
    if value >= 1e-6:
        return f"{value * 1e6:.3f} µs"
    return f"{value * 1e9:.3f} ns"


def format_bytes(value: int | float) -> str:
    number = float(value)
    units = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")
    for unit in units:
        if abs(number) < 1024 or unit == units[-1]:
            return f"{number:.2f} {unit}"
        number /= 1024
    raise AssertionError("byte formatter did not terminate")


def format_count(value: int | float) -> str:
    number = float(value)
    for scale, suffix in ((1e15, "P"), (1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "K")):
        if abs(number) >= scale:
            return f"{number / scale:.2f}{suffix}"
    return f"{number:.0f}"


def analysis_metrics(report: AnalysisReport) -> tuple[MetricView, ...]:
    utilization = report.memory["total"] / report.memory["capacity"]
    return (
        MetricView("迭代延迟", format_seconds(report.total_seconds), "端到端解析式估算", "primary"),
        MetricView("Token/s", format_count(report.total_tokens_per_second), "全局训练吞吐", "cyan"),
        MetricView(
            "Token/s/设备",
            format_count(report.tokens_per_second_per_device),
            f"{report.world_size:,} 个设备",
            "violet",
        ),
        MetricView("单设备内存", format_bytes(report.memory["total"]), f"容量使用率 {utilization:.1%}", "amber"),
        MetricView("主导项", report.bottleneck.replace("_", " "), "当前延迟分解最大项", "neutral"),
    )


def stage_rows(outcome: AnalysisOutcome) -> list[dict[str, Any]]:
    if outcome.report is None:
        return []
    return [
        {
            "stage": stage.stage,
            "label": stage.label,
            "pass": stage.pass_name,
            "schema": stage.schema,
            "nodes": stage.node_count,
            "values": stage.value_count,
            "lowering_ms": round(stage.duration_ns / 1e6, 4),
            "valid": stage.valid,
            "digest": stage.digest,
        }
        for stage in outcome.report.stages
    ]


def task_rows(outcome: AnalysisOutcome) -> list[dict[str, Any]]:
    if outcome.report is None:
        return []
    return [
        {
            "operation": task.operation,
            "phase": task.phase,
            "engine": task.engine,
            "kind": task.kind,
            "ops": task.operations,
            "read_bytes": task.read_bytes,
            "write_bytes": task.write_bytes,
            "message_bytes": task.message_bytes,
            "compute_s": task.compute_seconds,
            "memory_s": task.memory_seconds,
            "network_s": task.network_seconds,
            "total_s": task.total_seconds,
            "dependencies": len(task.dependencies),
            "concurrency_group": task.concurrency_group,
            "task_id": task.task_id,
        }
        for task in outcome.report.tasks
    ]


def sweep_rows(report: SweepReport) -> list[dict[str, Any]]:
    return [
        {
            "tp": case.tensor_parallel,
            "pp": case.pipeline_parallel,
            "dp": case.data_parallel,
            "world_size": case.world_size,
            "status": case.status,
            "feasible": case.feasible,
            "pareto": case.pareto,
            "latency_s": case.total_seconds,
            "memory_gib": case.memory_bytes / 1024**3 if case.memory_bytes is not None else None,
            "tokens_s_device": case.tokens_per_second_per_device,
            "bottleneck": case.bottleneck,
            "diagnostic": "; ".join(item.message for item in case.diagnostics),
            "request_digest": case.request_digest,
        }
        for case in report.cases
    ]


def latency_chart_options(report: AnalysisReport) -> dict[str, Any]:
    labels = [name.replace("_", " ") for name in report.latency]
    values = [float(value) for value in report.latency.values()]
    return {
        "backgroundColor": "transparent",
        "animationDuration": 500,
        "grid": {"left": 118, "right": 24, "top": 18, "bottom": 32},
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "xAxis": {
            "type": "value",
            "name": "seconds",
            "nameTextStyle": {"color": "#7f8ca8"},
            "axisLabel": {"color": "#7f8ca8"},
            "splitLine": {"lineStyle": {"color": "rgba(130, 153, 197, .10)"}},
        },
        "yAxis": {
            "type": "category",
            "data": labels,
            "inverse": True,
            "axisLabel": {"color": "#aebbd4"},
            "axisLine": {"show": False},
            "axisTick": {"show": False},
        },
        "series": [
            {
                "type": "bar",
                "data": values,
                "barMaxWidth": 16,
                "itemStyle": {
                    "borderRadius": [0, 8, 8, 0],
                    "color": {
                        "type": "linear",
                        "x": 0,
                        "y": 0,
                        "x2": 1,
                        "y2": 0,
                        "colorStops": [
                            {"offset": 0, "color": "#7357ff"},
                            {"offset": 1, "color": "#22d3ee"},
                        ],
                    },
                },
            }
        ],
    }


def memory_chart_options(report: AnalysisReport) -> dict[str, Any]:
    entries = tuple((name, value) for name, value in report.memory.items() if name not in {"total", "capacity"})
    return {
        "backgroundColor": "transparent",
        "animationDuration": 500,
        "grid": {"left": 118, "right": 24, "top": 18, "bottom": 32},
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "xAxis": {
            "type": "value",
            "name": "GiB",
            "nameTextStyle": {"color": "#7f8ca8"},
            "axisLabel": {"color": "#7f8ca8"},
            "splitLine": {"lineStyle": {"color": "rgba(130, 153, 197, .10)"}},
        },
        "yAxis": {
            "type": "category",
            "data": [name.replace("_", " ") for name, _ in entries],
            "inverse": True,
            "axisLabel": {"color": "#aebbd4"},
            "axisLine": {"show": False},
            "axisTick": {"show": False},
        },
        "series": [
            {
                "type": "bar",
                "data": [float(value) / 1024**3 for _, value in entries],
                "barMaxWidth": 16,
                "itemStyle": {"borderRadius": [0, 8, 8, 0], "color": "#f59e0b"},
            }
        ],
    }


def sweep_chart_options(report: SweepReport) -> dict[str, Any]:
    successful = [row for row in sweep_rows(report) if row["status"] == "success" and row["latency_s"] is not None]

    def points(pareto: bool) -> list[dict[str, Any]]:
        return [
            {
                "value": [row["memory_gib"], row["latency_s"]],
                "name": f"TP{row['tp']} / PP{row['pp']} / DP{row['dp']}",
            }
            for row in successful
            if bool(row["pareto"]) is pareto
        ]

    return {
        "backgroundColor": "transparent",
        "animationDuration": 600,
        "legend": {"top": 0, "textStyle": {"color": "#aebbd4"}},
        "grid": {"left": 64, "right": 26, "top": 48, "bottom": 48},
        "tooltip": {"trigger": "item"},
        "xAxis": {
            "type": "value",
            "name": "Memory / GiB",
            "nameLocation": "middle",
            "nameGap": 30,
            "nameTextStyle": {"color": "#7f8ca8"},
            "axisLabel": {"color": "#7f8ca8"},
            "splitLine": {"lineStyle": {"color": "rgba(130, 153, 197, .10)"}},
        },
        "yAxis": {
            "type": "value",
            "name": "Latency / s",
            "nameTextStyle": {"color": "#7f8ca8"},
            "axisLabel": {"color": "#7f8ca8"},
            "splitLine": {"lineStyle": {"color": "rgba(130, 153, 197, .10)"}},
        },
        "series": [
            {
                "name": "可行候选",
                "type": "scatter",
                "symbolSize": 11,
                "data": points(False),
                "itemStyle": {"color": "#53627d", "opacity": 0.72},
            },
            {
                "name": "Pareto 前沿",
                "type": "scatter",
                "symbolSize": 18,
                "data": points(True),
                "itemStyle": {"color": "#22d3ee", "shadowBlur": 16, "shadowColor": "rgba(34,211,238,.5)"},
            },
        ],
    }

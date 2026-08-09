"""Framework-neutral presentation models for interactive Blueprinting clients.

The application service owns analysis semantics.  This module only converts its
immutable reports into display-oriented values, rows, and chart specifications.
Keeping these adapters free of NiceGUI makes them cheap to unit test and reusable
by future CLI, notebook, or agent surfaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any

from blueprinting.application import AnalysisOutcome, AnalysisReport, SweepReport


@dataclass(frozen=True)
class MetricView:
    """One compact headline fact with optional supporting context."""

    label: str
    value: str
    detail: str
    tone: str = "neutral"


@dataclass(frozen=True)
class TimelineSummary:
    """Scope and size of a dependency-only task-time projection."""

    task_count: int
    dependency_count: int
    lane_count: int
    span_seconds: float


_TIME_CATEGORY_LABELS = {
    "forward": "前向计算",
    "backward": "反向计算",
    "optimizer": "优化器更新",
    "recompute": "激活重计算",
    "tensor_parallel": "张量并行通信",
    "pipeline_parallel": "流水并行通信",
    "data_parallel": "数据并行通信",
    "recommunication": "重通信",
    "pipeline_bubble": "流水空泡",
}
_PHASE_LABELS = {
    "forward": "前向",
    "recompute": "重计算",
    "activation_gradient": "激活梯度",
    "weight_gradient": "权重梯度",
    "optimizer": "优化器",
    "recommunication": "重通信",
}
_SYNTHETIC_TIME_TERMS = {
    "pipeline_parallel": "Pipeline point-to-point 解析项",
    "data_parallel": "Data-parallel collective 解析项",
    "pipeline_bubble": "1F1B / interleaving 空泡解析项",
}
_ENGINE_COLORS = {
    "matrix": "#2563eb",
    "vector": "#7c3aed",
    "collective": "#0891b2",
}


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
        MetricView("迭代延迟", format_seconds(report.total_seconds), "解析式训练迭代估算", "primary"),
        MetricView("全局吞吐", format_count(report.total_tokens_per_second), "Token/s", "cyan"),
        MetricView(
            "单设备吞吐",
            format_count(report.tokens_per_second_per_device),
            f"Token/s · {report.world_size:,} 个设备",
            "violet",
        ),
        MetricView("单设备内存", format_bytes(report.memory["total"]), f"容量使用率 {utilization:.1%}", "amber"),
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
    entries = [(name.replace("_", " "), float(value)) for name, value in report.latency.items()]
    largest = max((value for _, value in entries), default=0.0)
    data = [
        {
            "value": value,
            "itemStyle": {
                "color": "#2563eb" if value == largest else "#94a3b8",
                "borderRadius": [0, 4, 4, 0],
            },
        }
        for _, value in entries
    ]
    return {
        "backgroundColor": "transparent",
        "animationDuration": 350,
        "grid": {"left": 120, "right": 34, "top": 18, "bottom": 38},
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "xAxis": {
            "type": "value",
            "name": "seconds",
            "nameTextStyle": {"color": "#64748b"},
            "axisLabel": {"color": "#64748b"},
            "axisLine": {"lineStyle": {"color": "#cbd5e1"}},
            "splitLine": {"lineStyle": {"color": "#e2e8f0"}},
        },
        "yAxis": {
            "type": "category",
            "data": [label for label, _ in entries],
            "inverse": True,
            "axisLabel": {"color": "#475569"},
            "axisLine": {"show": False},
            "axisTick": {"show": False},
        },
        "series": [
            {
                "type": "bar",
                "data": data,
                "barMaxWidth": 14,
            }
        ],
    }


def memory_chart_options(report: AnalysisReport) -> dict[str, Any]:
    entries = tuple((name, value) for name, value in report.memory.items() if name not in {"total", "capacity"})
    colors = ("#2563eb", "#0891b2", "#64748b", "#7c3aed", "#d97706", "#be123c")
    capacity_gib = float(report.memory["capacity"]) / 1024**3
    series = []
    for index, (name, value) in enumerate(entries):
        item: dict[str, Any] = {
            "name": name.replace("_", " "),
            "type": "bar",
            "stack": "memory",
            "barMaxWidth": 34,
            "data": [float(value) / 1024**3],
            "itemStyle": {"color": colors[index % len(colors)]},
            "emphasis": {"focus": "series"},
        }
        if index == 0:
            item["markLine"] = {
                "silent": True,
                "symbol": "none",
                "label": {
                    "show": True,
                    "formatter": f"容量 {capacity_gib:.1f} GiB",
                    "color": "#b45309",
                    "position": "insideEndTop",
                },
                "lineStyle": {"color": "#b45309", "type": "dashed", "width": 1},
                "data": [{"xAxis": capacity_gib}],
            }
        series.append(item)
    return {
        "backgroundColor": "transparent",
        "animationDuration": 350,
        "legend": {
            "type": "scroll",
            "top": 0,
            "textStyle": {"color": "#475569", "fontSize": 11},
            "pageTextStyle": {"color": "#64748b"},
        },
        "grid": {"left": 24, "right": 28, "top": 72, "bottom": 42},
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "xAxis": {
            "type": "value",
            "name": "GiB",
            "nameTextStyle": {"color": "#64748b"},
            "axisLabel": {"color": "#64748b"},
            "axisLine": {"lineStyle": {"color": "#cbd5e1"}},
            "splitLine": {"lineStyle": {"color": "#e2e8f0"}},
        },
        "yAxis": {
            "type": "category",
            "data": ["每设备内存"],
            "axisLabel": {"show": False},
            "axisLine": {"show": False},
            "axisTick": {"show": False},
        },
        "series": series,
    }


def _time_category_for_task(phase: str, engine: str) -> str | None:
    if engine == "collective":
        if phase == "recommunication":
            return "recommunication"
        if phase in {"forward", "activation_gradient"}:
            return "tensor_parallel"
        return None
    if phase in {"activation_gradient", "weight_gradient"}:
        return "backward"
    if phase in {"forward", "recompute", "optimizer"}:
        return phase
    return None


def _scaled_time_children(report: AnalysisReport, category: str, seconds: float) -> list[dict[str, Any]]:
    matching = [
        task
        for task in report.tasks
        if _time_category_for_task(task.phase, task.engine) == category and task.total_seconds > 0
    ]
    raw_total = sum(float(task.total_seconds) for task in matching)
    if not matching or raw_total <= 0:
        return [
            {
                "name": _SYNTHETIC_TIME_TERMS.get(category, "未进一步分解的解析项"),
                "value": seconds,
            }
        ]

    groups: dict[str, dict[str, dict[str, float]]] = {}
    for task in matching:
        source_layer = task.source_layer or "other"
        domain = source_layer.split(".", 1)[0]
        phase = _PHASE_LABELS.get(task.phase, task.phase.replace("_", " "))
        group_name = f"{phase} · {domain}" if category in {"backward", "tensor_parallel"} else domain
        layer = groups.setdefault(group_name, {}).setdefault(source_layer, {})
        operation = task.operation.replace("transformer.", "").replace("collective.", "")
        layer[operation] = layer.get(operation, 0.0) + float(task.total_seconds)

    scale = seconds / raw_total
    children: list[dict[str, Any]] = []
    for group_name, layers in groups.items():
        layer_nodes = []
        for layer_name, operations in layers.items():
            operation_nodes = [
                {"name": operation.replace("_", " "), "value": raw_seconds * scale}
                for operation, raw_seconds in operations.items()
            ]
            layer_nodes.append(
                {
                    "name": layer_name,
                    "value": sum(float(node["value"]) for node in operation_nodes),
                    "children": operation_nodes,
                }
            )
        children.append(
            {
                "name": group_name,
                "value": sum(float(node["value"]) for node in layer_nodes),
                "children": layer_nodes,
            }
        )
    return children


def time_breakdown_tree(report: AnalysisReport) -> list[dict[str, Any]]:
    """Build an iteration-to-operation hierarchy while preserving headline totals."""

    nodes = []
    for category, value in report.latency.items():
        seconds = float(value)
        if seconds <= 0:
            continue
        nodes.append(
            {
                "name": _TIME_CATEGORY_LABELS.get(category, category.replace("_", " ")),
                "value": seconds,
                "category": category,
                "children": _scaled_time_children(report, category, seconds),
            }
        )
    return nodes


def time_breakdown_rows(report: AnalysisReport) -> list[dict[str, Any]]:
    """Flatten the hierarchy for precise, sortable inspection."""

    rows: list[dict[str, Any]] = []
    total = float(report.total_seconds)

    def append_nodes(nodes: list[dict[str, Any]], path: tuple[str, ...], parent_seconds: float) -> None:
        for node in nodes:
            seconds = float(node["value"])
            node_path = (*path, str(node["name"]))
            rows.append(
                {
                    "level": len(node_path),
                    "component": str(node["name"]),
                    "path": " / ".join(node_path),
                    "seconds": seconds,
                    "share_iteration": seconds / total if total else 0.0,
                    "share_parent": seconds / parent_seconds if parent_seconds else 0.0,
                    "category": str(node.get("category", "detail")),
                }
            )
            children = node.get("children", [])
            if children:
                append_nodes(children, node_path, seconds)

    append_nodes(time_breakdown_tree(report), (), total)
    return rows


def time_breakdown_chart_options(report: AnalysisReport) -> dict[str, Any]:
    total = float(report.total_seconds)
    return {
        "backgroundColor": "transparent",
        "animationDuration": 350,
        "tooltip": {
            ":formatter": (
                "function(params) {"
                "const value = Number(params.value || 0);"
                f"const total = {total!r};"
                "const share = total > 0 ? value / total * 100 : 0;"
                "return '<strong>' + params.name + '</strong><br/>' + "
                "value.toFixed(6) + ' s · ' + share.toFixed(2) + '% of iteration';"
                "}"
            )
        },
        "series": [
            {
                "type": "treemap",
                "name": "Iteration time",
                "data": time_breakdown_tree(report),
                "roam": False,
                "nodeClick": "zoomToNode",
                "leafDepth": 2,
                "visibleMin": 24,
                "squareRatio": 1.15,
                "breadcrumb": {
                    "show": True,
                    "bottom": 0,
                    "height": 22,
                    "itemStyle": {"color": "#f8fafc", "borderColor": "#cbd5e1"},
                    "emphasis": {"itemStyle": {"color": "#eff6ff"}},
                },
                "label": {"show": True, "color": "#ffffff", "fontSize": 11, "overflow": "truncate"},
                "upperLabel": {"show": True, "height": 24, "color": "#ffffff", "fontSize": 11},
                "levels": [
                    {"itemStyle": {"borderColor": "#ffffff", "borderWidth": 0, "gapWidth": 3}},
                    {
                        "color": ["#2563eb", "#0891b2", "#7c3aed", "#d97706", "#64748b", "#be123c"],
                        "colorMappingBy": "index",
                        "itemStyle": {"borderColor": "#ffffff", "borderWidth": 3, "gapWidth": 3},
                    },
                    {
                        "colorSaturation": [0.28, 0.58],
                        "itemStyle": {"borderColorSaturation": 0.68, "gapWidth": 2, "borderWidth": 1},
                    },
                    {
                        "colorSaturation": [0.18, 0.42],
                        "itemStyle": {"borderColorSaturation": 0.55, "gapWidth": 1},
                    },
                ],
            }
        ],
    }


def task_dependency_projection(report: AnalysisReport) -> list[dict[str, Any]]:
    """Project task start/end from dependencies only, without resource scheduling."""

    tasks = {task.task_id: task for task in report.tasks}
    starts: dict[str, float] = {}
    ends: dict[str, float] = {}
    visiting: set[str] = set()

    def project(task_id: str) -> float:
        if task_id in ends:
            return ends[task_id]
        task = tasks[task_id]
        if task_id in visiting:
            raise ValueError("portable task dependencies contain a cycle")
        visiting.add(task_id)
        start = max((project(item) for item in task.dependencies if item in tasks), default=0.0)
        starts[task_id] = start
        ends[task_id] = start + max(float(task.total_seconds), 0.0)
        visiting.remove(task_id)
        return ends[task_id]

    for task in report.tasks:
        project(task.task_id)

    return [
        {
            "task_id": task.task_id,
            "operation": task.operation,
            "phase": task.phase,
            "engine": task.engine,
            "source_layer": task.source_layer,
            "concurrency_group": task.concurrency_group,
            "start_seconds": starts[task.task_id],
            "end_seconds": ends[task.task_id],
            "duration_seconds": max(float(task.total_seconds), 0.0),
            "dependencies": len(task.dependencies),
        }
        for task in report.tasks
    ]


def timeline_summary(report: AnalysisReport) -> TimelineSummary:
    rows = task_dependency_projection(report)
    return TimelineSummary(
        task_count=len(rows),
        dependency_count=sum(len(task.dependencies) for task in report.tasks),
        lane_count=len({task.phase for task in report.tasks}),
        span_seconds=max((float(row["end_seconds"]) for row in rows), default=0.0),
    )


def dependency_timeline_chart_options(report: AnalysisReport) -> dict[str, Any]:
    """Render every portable task on a zoomable dependency-projected timeline."""

    projection = task_dependency_projection(report)
    phases = list(dict.fromkeys(str(row["phase"]) for row in projection))
    phase_indices = {phase: index for index, phase in enumerate(phases)}
    data = []
    for row in projection:
        start_ms = float(row["start_seconds"]) * 1e3
        end_ms = float(row["end_seconds"]) * 1e3
        duration_ms = float(row["duration_seconds"]) * 1e3
        data.append(
            {
                "name": str(row["operation"]).split(".")[-1].replace("_", " "),
                "value": [phase_indices[str(row["phase"])], start_ms, end_ms, duration_ms],
                "operation": row["operation"],
                "phase": _PHASE_LABELS.get(str(row["phase"]), str(row["phase"]).replace("_", " ")),
                "engine": row["engine"],
                "source_layer": row["source_layer"],
                "task_id": row["task_id"],
                "dependencies": row["dependencies"],
                "itemStyle": {
                    "color": _ENGINE_COLORS.get(str(row["engine"]), "#64748b"),
                    "opacity": 0.9,
                },
            }
        )
    return {
        "backgroundColor": "transparent",
        "animationDuration": 250,
        "grid": {"left": 112, "right": 26, "top": 20, "bottom": 72},
        "tooltip": {
            ":formatter": (
                "function(params) {"
                "const d = params.data; const v = d.value;"
                "return '<strong>' + d.operation + '</strong><br/>' + d.source_layer + "
                "'<br/>Phase · ' + d.phase + ' · Engine · ' + d.engine + "
                "'<br/>Start · ' + Number(v[1]).toFixed(4) + ' ms' + "
                "'<br/>Duration · ' + Number(v[3]).toFixed(4) + ' ms' + "
                "'<br/>Dependencies · ' + d.dependencies;"
                "}"
            )
        },
        "dataZoom": [
            {"type": "inside", "xAxisIndex": 0, "filterMode": "weakFilter"},
            {
                "type": "slider",
                "xAxisIndex": 0,
                "height": 18,
                "bottom": 18,
                "filterMode": "weakFilter",
                "borderColor": "#e2e8f0",
                "fillerColor": "rgba(37, 99, 235, .12)",
                "handleStyle": {"color": "#2563eb"},
            },
        ],
        "xAxis": {
            "type": "value",
            "name": "dependency-projected ms",
            "nameLocation": "middle",
            "nameGap": 48,
            "nameTextStyle": {"color": "#64748b"},
            "axisLabel": {"color": "#64748b"},
            "axisLine": {"lineStyle": {"color": "#cbd5e1"}},
            "splitLine": {"lineStyle": {"color": "#e2e8f0"}},
            "min": 0,
        },
        "yAxis": {
            "type": "category",
            "data": [_PHASE_LABELS.get(phase, phase.replace("_", " ")) for phase in phases],
            "inverse": True,
            "axisLabel": {"color": "#475569", "fontSize": 11},
            "axisLine": {"show": False},
            "axisTick": {"show": False},
        },
        "series": [
            {
                "type": "custom",
                "name": "Portable tasks",
                "clip": True,
                ":renderItem": (
                    "function(params, api) {"
                    "const lane = api.value(0);"
                    "const start = api.coord([api.value(1), lane]);"
                    "const end = api.coord([api.value(2), lane]);"
                    "const height = api.size([0, 1])[1] * 0.56;"
                    "const bounds = params.coordSys;"
                    "const left = Math.max(start[0], bounds.x);"
                    "const right = Math.min("
                    "Math.max(end[0], start[0] + 1), bounds.x + bounds.width"
                    ");"
                    "const top = Math.max(start[1] - height / 2, bounds.y);"
                    "const bottom = Math.min(start[1] + height / 2, bounds.y + bounds.height);"
                    "if (right <= left || bottom <= top) { return null; }"
                    "return {type: 'rect', shape: {"
                    "x: left, y: top, width: right - left, height: bottom - top"
                    "}, style: api.style()};"
                    "}"
                ),
                "encode": {"x": [1, 2], "y": 0},
                "data": data,
            }
        ],
    }


def sweep_chart_options(
    report: SweepReport,
    rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    successful = [
        row
        for row in (rows if rows is not None else sweep_rows(report))
        if row["status"] == "success" and row["latency_s"] is not None
    ]

    def points(*, feasible: bool, pareto: bool = False) -> list[dict[str, Any]]:
        return [
            {
                "value": [row["memory_gib"], row["latency_s"]],
                "name": f"TP{row['tp']} / PP{row['pp']} / DP{row['dp']}",
            }
            for row in successful
            if bool(row["feasible"]) is feasible and (not feasible or bool(row["pareto"]) is pareto)
        ]

    return {
        "backgroundColor": "transparent",
        "animationDuration": 400,
        "legend": {"top": 0, "textStyle": {"color": "#475569"}},
        "grid": {"left": 66, "right": 28, "top": 52, "bottom": 52},
        "tooltip": {"trigger": "item"},
        "xAxis": {
            "type": "value",
            "name": "Memory / GiB",
            "nameLocation": "middle",
            "nameGap": 30,
            "nameTextStyle": {"color": "#64748b"},
            "axisLabel": {"color": "#64748b"},
            "axisLine": {"lineStyle": {"color": "#cbd5e1"}},
            "splitLine": {"lineStyle": {"color": "#e2e8f0"}},
        },
        "yAxis": {
            "type": "value",
            "name": "Latency / s",
            "nameTextStyle": {"color": "#64748b"},
            "axisLabel": {"color": "#64748b"},
            "axisLine": {"lineStyle": {"color": "#cbd5e1"}},
            "splitLine": {"lineStyle": {"color": "#e2e8f0"}},
        },
        "series": [
            {
                "name": "可行候选",
                "type": "scatter",
                "symbolSize": 10,
                "data": points(feasible=True, pareto=False),
                "itemStyle": {"color": "#64748b", "opacity": 0.72},
            },
            {
                "name": "非支配候选",
                "type": "scatter",
                "symbolSize": 16,
                "data": points(feasible=True, pareto=True),
                "itemStyle": {"color": "#2563eb", "borderColor": "#dbeafe", "borderWidth": 1},
            },
            {
                "name": "容量不可行",
                "type": "scatter",
                "symbol": "emptyCircle",
                "symbolSize": 12,
                "data": points(feasible=False),
                "itemStyle": {"color": "#b91c1c", "opacity": 0.82},
            },
        ],
    }


def _histogram(values: list[float]) -> tuple[list[str], list[int]]:
    if not values:
        return [], []
    lower = min(values)
    upper = max(values)
    if lower == upper:
        return [f"{lower:.3g}"], [len(values)]

    bin_count = max(2, min(10, round(sqrt(len(values)))))
    width = (upper - lower) / bin_count
    counts = [0] * bin_count
    for value in values:
        index = min(int((value - lower) / width), bin_count - 1)
        counts[index] += 1
    labels = [f"{lower + index * width:.3g}–{lower + (index + 1) * width:.3g}" for index in range(bin_count)]
    return labels, counts


def sweep_distribution_chart_options(
    report: SweepReport,
    rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Render latency and memory histograms for the visible successful cases."""

    successful = [row for row in (rows if rows is not None else sweep_rows(report)) if row["status"] == "success"]
    latency_labels, latency_counts = _histogram(
        [float(row["latency_s"]) for row in successful if row["latency_s"] is not None]
    )
    memory_labels, memory_counts = _histogram(
        [float(row["memory_gib"]) for row in successful if row["memory_gib"] is not None]
    )
    common_axis = {
        "type": "category",
        "axisLabel": {"color": "#64748b", "fontSize": 10, "hideOverlap": True},
        "axisLine": {"lineStyle": {"color": "#cbd5e1"}},
        "axisTick": {"show": False},
    }
    common_value_axis = {
        "type": "value",
        "minInterval": 1,
        "axisLabel": {"color": "#64748b", "fontSize": 10},
        "axisLine": {"show": False},
        "splitLine": {"lineStyle": {"color": "#e2e8f0"}},
    }
    return {
        "backgroundColor": "transparent",
        "animationDuration": 350,
        "title": [
            {"text": "延迟 / seconds", "left": 2, "top": 0, "textStyle": {"color": "#475569", "fontSize": 11}},
            {"text": "单设备内存 / GiB", "left": 2, "top": "51%", "textStyle": {"color": "#475569", "fontSize": 11}},
        ],
        "grid": [
            {"left": 42, "right": 14, "top": 34, "height": "29%"},
            {"left": 42, "right": 14, "top": "62%", "height": "25%"},
        ],
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "xAxis": [
            {**common_axis, "gridIndex": 0, "data": latency_labels},
            {**common_axis, "gridIndex": 1, "data": memory_labels},
        ],
        "yAxis": [
            {**common_value_axis, "gridIndex": 0},
            {**common_value_axis, "gridIndex": 1},
        ],
        "series": [
            {
                "name": "Case 数",
                "type": "bar",
                "xAxisIndex": 0,
                "yAxisIndex": 0,
                "data": latency_counts,
                "barMaxWidth": 26,
                "itemStyle": {"color": "#2563eb", "borderRadius": [3, 3, 0, 0]},
            },
            {
                "name": "Case 数",
                "type": "bar",
                "xAxisIndex": 1,
                "yAxisIndex": 1,
                "data": memory_counts,
                "barMaxWidth": 26,
                "itemStyle": {"color": "#0891b2", "borderRadius": [3, 3, 0, 0]},
            },
        ],
    }

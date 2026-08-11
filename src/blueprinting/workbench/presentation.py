"""Framework-neutral presentation models for interactive Blueprinting clients.

The application service owns analysis semantics.  This module only converts its
immutable reports into display-oriented values, rows, and chart specifications.
Keeping these adapters free of NiceGUI makes them cheap to unit test and reusable
by future CLI, notebook, or agent surfaces.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import sqrt
from typing import Any

from blueprinting.application import (
    AnalysisOutcome,
    AnalysisReport,
    DerivedOverlay,
    IRBoundaryView,
    IRGraphEdge,
    IRGraphNode,
    IRGraphView,
    SweepReport,
)
from blueprinting.synthesizer.stages.concrete_plan.ir import ConcretePlanIR
from blueprinting.synthesizer.stages.distributed.ir import DistributedTaskIR
from blueprinting.synthesizer.stages.machine.ir import MachineIR
from blueprinting.synthesizer.stages.model.ir import ModelIR
from blueprinting.synthesizer.stages.portable_plan.ir import PortablePlanIR


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


@dataclass(frozen=True)
class IRStageNarrative:
    """Human-facing explanation of one canonical representation boundary."""

    question: str
    answer: str
    result: str
    excludes: str
    next_step: str


@dataclass(frozen=True)
class LoweringNarrative:
    """Readable summary of a typed-lineage boundary."""

    headline: str
    detail: str
    metrics: tuple[MetricView, ...]


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
_IR_KIND_COLORS = {
    "operation": "#2563eb",
    "task": "#2563eb",
    "command": "#7c3aed",
    "instruction": "#7c3aed",
    "value": "#0891b2",
    "buffer": "#0891b2",
    "device": "#b45309",
    "queue": "#d97706",
    "memory_region": "#ca8a04",
    "section": "#475569",
    "entry_point": "#15803d",
    "sync_token": "#dc2626",
    "compute": "#2563eb",
    "local_compute": "#2563eb",
    "collective": "#0891b2",
    "transfer": "#0f766e",
    "barrier": "#dc2626",
    "host": "#64748b",
    "group": "#64748b",
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


def _kind_summary(items: tuple[Any, ...], *, limit: int = 3) -> str:
    counts: dict[str, int] = {}
    for item in items:
        kind = getattr(item, "kind", None)
        label = getattr(kind, "value", str(kind))
        counts[label] = counts.get(label, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return " · ".join(f"{count:,} {label}" for label, count in ranked[:limit]) or "0"


def ir_stage_narrative(ir: Any) -> IRStageNarrative:
    """Explain ownership and the concrete contents of a canonical snapshot."""

    if isinstance(ir, ModelIR):
        return IRStageNarrative(
            "这个模型在语义上做什么？",
            "显式记录 tensor value、operation、数据流与副作用，作为后续推导不变的语义起点。",
            f"{len(ir.operations):,} 个 operation · {len(ir.values):,} 个 value · "
            f"{len(ir.inputs):,} 入 / {len(ir.outputs):,} 出",
            "不包含并行 placement、硬件吞吐或预测时间。",
            "下一步：把模型语义展开到逻辑 mesh 上的 local compute、collective 与依赖。",
        )
    if isinstance(ir, DistributedTaskIR):
        axes = " × ".join(f"{axis.name}={axis.size}" for axis in ir.mesh.axes)
        return IRStageNarrative(
            "模型工作如何分布到逻辑参与者？",
            "记录逻辑 mesh、sharding、collective 与分布式依赖；rank 仍是虚拟参与者。",
            f"{len(ir.tasks):,} 个 task（{_kind_summary(ir.tasks)}）· mesh {axes} = {ir.mesh.size:,} ranks",
            "不选择物理设备、queue、kernel 或目标实现。",
            "下一步：固化已选策略的精确 workload、抽象 buffer 与资源需求。",
        )
    if isinstance(ir, PortablePlanIR):
        return IRStageNarrative(
            "已选策略需要完成哪些精确工作？",
            "用 target-neutral task DAG 表达 workload facts、抽象 buffer、资源需求与目标。",
            f"{len(ir.tasks):,} 个 task（{_kind_summary(ir.tasks)}）· {len(ir.buffers):,} 个 buffer · "
            f"{len(ir.objectives):,} 个 objective",
            "不包含 kernel ID、物理 queue、经验 duration 或 wall-clock timestamp。",
            "下一步：绑定 target 与 deployment，完成实现选择、placement、ordering 与内存区域规划。",
        )
    if isinstance(ir, ConcretePlanIR):
        return IRStageNarrative(
            "这个 target 上合法的执行计划是什么？",
            "记录实现选择、物理 placement、ordering、buffer region、同步与 command DAG。",
            f"{len(ir.commands):,} 个 command · {len(ir.devices):,} 个 device · "
            f"{len(ir.queues):,} 个 queue · {len(ir.memory_regions):,} 个 memory region",
            "预测时间不是执行正确性的事实源；target-only 语义属于 typed extension。",
            "下一步：同一 concrete digest 可进入 timing/simulation，或由 target plugin 实现为 MachineIR。",
        )
    if isinstance(ir, MachineIR):
        return IRStageNarrative(
            "目标机器最终执行什么程序？",
            "由 target plugin 拥有指令 dialect、section、entry point 与 ABI。",
            f"{len(ir.instructions):,} 条 instruction · {len(ir.sections):,} 个 section · "
            f"{len(ir.entry_points):,} 个 entry point · {ir.program_format}",
            "不为了统一表面形式而把 target 语义上提到 portable 层。",
            "下一步：生成 target artifact 或 replay package，并沿 lineage 关联观测。",
        )
    raise TypeError(f"unsupported canonical IR type: {type(ir).__name__}")


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


_STRUCTURE_PHASE_ORDER = {
    "input": 0,
    "model": 10,
    "forward": 20,
    "recompute": 30,
    "activation_gradient": 40,
    "weight_gradient": 50,
    "recommunication": 60,
    "optimizer": 70,
    "output": 80,
    "topology": 10,
    "placement": 20,
    "memory": 30,
    "commands": 40,
    "synchronization": 50,
    "entry": 10,
    "sections": 20,
    "instructions": 30,
}
_STRUCTURE_PHASE_LABELS = {
    "input": "输入",
    "model": "模型语义",
    "forward": "前向",
    "recompute": "重计算",
    "activation_gradient": "激活梯度",
    "weight_gradient": "权重梯度",
    "recommunication": "重通信",
    "optimizer": "优化器",
    "output": "输出",
    "topology": "Target 拓扑",
    "placement": "物理放置",
    "memory": "内存计划",
    "commands": "Command DAG",
    "synchronization": "同步",
    "entry": "入口",
    "sections": "Sections",
    "instructions": "指令流",
}


def _structure_parts(node: IRGraphNode, stage: Any) -> tuple[str, str, str]:
    """Map one entity to a stable stage/subsystem/type structure coordinate."""

    prefix, _, suffix = (node.group or "unscoped").partition("/")
    entity_kind = node.ref.kind
    semantic_kind = dict(node.properties).get("kind", entity_kind)
    if stage.value == "model":
        if entity_kind == "value" and suffix in {"input", "output"}:
            return suffix, "tensor", entity_kind
        return "model", suffix or prefix, entity_kind
    if stage.value in {"distributed", "portable"}:
        if entity_kind in {"value", "buffer"}:
            phase = suffix if suffix in {"input", "output"} else "input"
            return phase, entity_kind, entity_kind
        subsystem = suffix.split(".", maxsplit=1)[0] if suffix else entity_kind
        return prefix or "unscoped", subsystem or entity_kind, semantic_kind
    if stage.value == "concrete":
        phase = {
            "device": "topology",
            "queue": "placement",
            "memory_region": "placement",
            "buffer": "memory",
            "command": "commands",
            "sync_token": "synchronization",
        }.get(entity_kind, "commands")
        return phase, entity_kind, semantic_kind
    if stage.value == "machine":
        phase = {
            "entry_point": "entry",
            "section": "sections",
            "instruction": "instructions",
        }.get(entity_kind, "instructions")
        return phase, suffix or entity_kind, entity_kind
    return prefix or "unscoped", suffix or entity_kind, entity_kind


def _bounded_ir_graph(graph: IRGraphView, max_nodes: int = 20) -> tuple[list[IRGraphNode], list[IRGraphEdge], bool]:
    """Collapse a dense graph by semantic stage, subsystem, and entity kind."""

    if len(graph.nodes) <= max_nodes:
        return list(graph.nodes), list(graph.edges), False
    groups: dict[tuple[str, str, str], list[IRGraphNode]] = {}
    for node in graph.nodes:
        groups.setdefault(_structure_parts(node, graph.stage), []).append(node)
    ranked = sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))
    if len(ranked) > max_nodes:
        kept = ranked[: max_nodes - 1]
        overflow = [node for _, members in ranked[max_nodes - 1 :] for node in members]
        ranked = kept + [(("other", "mixed", "group"), overflow)]
    digest = graph.snapshot_digest
    group_nodes = []
    node_to_group = {}
    for index, ((phase, subsystem, entity_kind), members) in enumerate(ranked):
        ref = members[0].ref.__class__(
            graph.stage,
            digest,
            "group",
            f"group:{index}:{phase}:{subsystem}:{entity_kind}",
        )
        noun = entity_kind.replace("_", " ")
        group_nodes.append(
            IRGraphNode(
                ref,
                f"{subsystem}\n{len(members):,} {noun}",
                f"{phase}/{subsystem}",
                (
                    ("phase", phase),
                    ("subsystem", subsystem),
                    ("entity_kind", entity_kind),
                    ("entity_count", str(len(members))),
                ),
            )
        )
        for member in members:
            node_to_group[member.ref.key] = ref
    edge_counts: dict[tuple[str, str, str], int] = {}
    refs = {node.ref.key: node.ref for node in group_nodes}
    for edge in graph.edges:
        source = node_to_group.get(edge.source.key)
        target = node_to_group.get(edge.target.key)
        if source is None or target is None or source == target:
            continue
        key = (source.key, target.key, edge.kind)
        edge_counts[key] = edge_counts.get(key, 0) + edge.count
        refs[source.key] = source
        refs[target.key] = target
    edges = [
        IRGraphEdge(refs[source], refs[target], kind, count=count)
        for (source, target, kind), count in sorted(edge_counts.items())
    ]
    return group_nodes, edges, True


def ir_graph_chart_options(
    graph: IRGraphView,
    overlay: DerivedOverlay | None = None,
    *,
    max_nodes: int = 20,
) -> dict[str, Any]:
    nodes, edges, aggregated = _bounded_ir_graph(graph, max_nodes)
    overlay_metrics = {item.entity_id: dict(item.metrics) for item in overlay.entities} if overlay is not None else {}
    max_cost = max((metrics.get("total_seconds", 0.0) for metrics in overlay_metrics.values()), default=0.0)
    node_parts: dict[str, tuple[str, str, str]] = {}
    for node in nodes:
        properties = dict(node.properties)
        node_parts[node.ref.key] = (
            properties.get("phase", _structure_parts(node, graph.stage)[0]),
            properties.get("subsystem", _structure_parts(node, graph.stage)[1]),
            properties.get("entity_kind", _structure_parts(node, graph.stage)[2]),
        )
    phases = sorted(
        {parts[0] for parts in node_parts.values()},
        key=lambda phase: (_STRUCTURE_PHASE_ORDER.get(phase, 45), phase),
    )
    lanes = sorted(
        {(parts[1], parts[2]) for parts in node_parts.values()},
        key=lambda lane: (lane[0] not in {"tensor", "value", "buffer"}, lane[0], lane[1]),
    )
    lane_y = {lane: 92 + index * 76 for index, lane in enumerate(lanes)}
    by_phase: dict[str, list[IRGraphNode]] = {phase: [] for phase in phases}
    for node in nodes:
        by_phase[node_parts[node.ref.key][0]].append(node)
    for members in by_phase.values():
        members.sort(key=lambda node: (node_parts[node.ref.key][1], node_parts[node.ref.key][2], node.label))
    phase_x = {phase: 190 + index * 210 for index, phase in enumerate(phases)}
    data = []
    for subsystem, entity_kind in lanes:
        data.append(
            {
                "id": f"lane:{subsystem}:{entity_kind}",
                "name": f"{subsystem}\n{_ENTITY_TYPE_LABELS.get(entity_kind, entity_kind.replace('_', ' '))}",
                "x": 0,
                "y": lane_y[(subsystem, entity_kind)],
                "symbol": "roundRect",
                "symbolSize": [132, 46],
                "itemStyle": {"color": "#f8fafc", "borderColor": "#cbd5e1", "borderWidth": 1},
                "label": {
                    "show": True,
                    "position": "inside",
                    "color": "#475569",
                    "fontSize": 10,
                    "lineHeight": 14,
                },
                "is_lane": True,
                "subsystem": subsystem,
                "entity_kind": entity_kind,
            }
        )
    for phase in phases:
        members = by_phase[phase]
        data.append(
            {
                "id": f"phase:{phase}",
                "name": _STRUCTURE_PHASE_LABELS.get(phase, phase.replace("_", " ")),
                "x": phase_x[phase],
                "y": 0,
                "symbol": "roundRect",
                "symbolSize": [154, 34],
                "itemStyle": {"color": "#e2e8f0", "borderColor": "#cbd5e1", "borderWidth": 1},
                "label": {"show": True, "color": "#334155", "fontSize": 11, "fontWeight": 650},
                "is_phase": True,
                "entity_count": len(members),
            }
        )
    for node in nodes:
        phase, subsystem, entity_kind = node_parts[node.ref.key]
        total_seconds = overlay_metrics.get(node.ref.entity_id, {}).get("total_seconds")
        color = _IR_KIND_COLORS.get(entity_kind, "#64748b")
        if total_seconds is not None and max_cost > 0:
            opacity = 0.35 + 0.65 * (total_seconds / max_cost) ** 0.5
            color = f"rgba(220, 38, 38, {opacity:.3f})"
        data.append(
            {
                "id": node.ref.key,
                "entity_id": "" if aggregated else node.ref.entity_id,
                "name": node.label,
                "kind": node.ref.kind,
                "subsystem": subsystem,
                "entity_kind": entity_kind,
                "group": node.group,
                "properties": "<br/>".join(f"{key} · {value}" for key, value in node.properties),
                "x": phase_x[phase],
                "y": lane_y[(subsystem, entity_kind)],
                "symbol": "roundRect",
                "symbolSize": [154, 52],
                "itemStyle": {"color": color, "borderColor": "#ffffff", "borderWidth": 1.5},
                "overlay_seconds": total_seconds,
                "label": {
                    "show": True,
                    "position": "inside",
                    "color": "#ffffff",
                    "fontSize": 10,
                    "lineHeight": 14,
                },
            }
        )
    return {
        "backgroundColor": "transparent",
        "animationDuration": 250,
        "tooltip": {
            ":formatter": (
                "function(params) { const d = params.data;"
                "if (!d) return '';"
                "if (d.is_phase) return '<strong>' + d.name + '</strong><br/>' + d.entity_count + ' structure group(s)';"
                "if (d.is_lane) return '<strong>' + d.name.replace('\\n', ' · ') + '</strong><br/>semantic lane';"
                "let result = '<strong>' + d.name.replace('\\n', ' · ') + '</strong><br/>' + d.entity_kind;"
                "if (d.entity_id) result += '<br/>' + d.entity_id;"
                "if (d.properties) result += '<br/>' + d.properties;"
                "if (d.overlay_seconds != null) result += '<br/>DERIVED total · ' + d.overlay_seconds + ' s';"
                "return result; }"
            )
        },
        "series": [
            {
                "type": "graph",
                "layout": "none",
                "roam": False,
                "draggable": False,
                "data": data,
                "links": [
                    {
                        "source": edge.source.key,
                        "target": edge.target.key,
                        "value": edge.count,
                        "kind": edge.kind,
                        "lineStyle": {"width": min(1 + edge.count**0.5, 5), "opacity": 0.1, "curveness": 0.04},
                    }
                    for edge in edges
                ],
                "edgeSymbol": ["none", "arrow"],
                "edgeSymbolSize": 7,
                "emphasis": {"focus": "adjacency", "lineStyle": {"width": 3, "opacity": 0.8}},
            }
        ],
        "graphic": [
            {
                "type": "text",
                "right": 10,
                "bottom": 8,
                "style": {
                    "text": (
                        f"Semantic structure projection · {len(graph.nodes):,} canonical entities"
                        if aggregated
                        else "Canonical entity structure"
                    ),
                    "fill": "#64748b",
                    "fontSize": 11,
                },
            }
        ],
    }


def boundary_rows(boundary: IRBoundaryView) -> list[dict[str, Any]]:
    return [
        {
            "target": relation.target.entity_id,
            "target_kind": relation.target.kind,
            "sources": ", ".join(item.entity_id for item in relation.sources),
            "source_count": len(relation.sources),
            "unresolved": ", ".join(relation.unresolved_sources),
            "lineage_kind": relation.lineage_kind,
            "transform": relation.transform,
            "explicit_mismatch": relation.explicit_source_mismatch,
        }
        for relation in boundary.relations
    ]


def _node_lookup(graph: IRGraphView | None) -> dict[str, IRGraphNode]:
    if graph is None:
        return {}
    result: dict[str, IRGraphNode] = {}
    for node in graph.nodes:
        result[node.ref.key] = node
        result[node.ref.entity_id] = node
    return result


def _graph_display_aliases(graph: IRGraphView | None) -> dict[str, str]:
    """Assign readable snapshot-local aliases without replacing stable IDs."""

    aliases: dict[str, str] = {}
    counters: dict[str, int] = {}
    if graph is None:
        return aliases
    for node in graph.nodes:
        identifier = node.ref.entity_id
        prefix, separator, _ = identifier.partition(":")
        if node.label != identifier or not separator or prefix not in {"value", "node", "buffer"}:
            continue
        counters[prefix] = counters.get(prefix, 0) + 1
        aliases[identifier] = f"{prefix}#{counters[prefix]}"
    return aliases


def _display_label(node: IRGraphNode, aliases: dict[str, str]) -> str:
    return aliases.get(node.ref.entity_id, node.label)


def _semantic_boundary_bucket(node: IRGraphNode) -> tuple[str, str, int]:
    phase, subsystem, entity_kind = _structure_parts(node, node.ref.stage)
    phase_label = _STRUCTURE_PHASE_LABELS.get(phase, phase.replace("_", " "))
    return (
        f"{phase_label} · {subsystem}",
        entity_kind.replace("_", " ").replace("-", " "),
        _STRUCTURE_PHASE_ORDER.get(phase, 45),
    )


_ENTITY_TYPE_LABELS = {
    "operation": "Operation",
    "value": "Value",
    "task": "Task",
    "local_compute": "Local compute",
    "local compute": "Local compute",
    "compute": "Compute",
    "collective": "Collective",
    "point_to_point": "Point-to-point",
    "point to point": "Point-to-point",
    "reshard": "Reshard",
    "buffer": "Buffer",
    "command": "Command",
    "device": "Device",
    "queue": "Queue",
    "memory_region": "Memory region",
    "memory region": "Memory region",
    "sync_token": "Sync token",
    "sync token": "Sync token",
    "instruction": "Instruction",
    "section": "Section",
    "entry_point": "Entry point",
    "entry point": "Entry point",
    "generated": "Generated",
}


def _entity_type_label(kind: str) -> str:
    return _ENTITY_TYPE_LABELS.get(kind, kind.replace("_", " ").replace("-", " ").title())


def _entity_tone(kind: str) -> str:
    normalized = kind.replace(" ", "_").replace("-", "_")
    if normalized in {"operation", "task", "local_compute", "compute"}:
        return "blue"
    if normalized in {"value", "buffer", "memory_region"}:
        return "green"
    if normalized in {"collective", "point_to_point", "reshard", "sync_token"}:
        return "violet"
    if normalized in {"command", "queue", "device"}:
        return "amber"
    if normalized in {"instruction", "section", "entry_point"}:
        return "cyan"
    return "slate"


def _decoded_property(node: IRGraphNode, name: str) -> Any:
    raw = dict(node.properties).get(name)
    if raw is None:
        return None
    try:
        value = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return raw
    if isinstance(value, dict) and set(value) == {"$tuple"}:
        return value["$tuple"]
    return value


def _unique_property(nodes: list[IRGraphNode], name: str) -> list[Any]:
    result = []
    for node in nodes:
        value = _decoded_property(node, name)
        if value is not None and value not in result:
            result.append(value)
    return result


def _sum_integer_property(nodes: list[IRGraphNode], name: str) -> int | None:
    values = [_decoded_property(node, name) for node in nodes]
    if not values or any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        return None
    return sum(int(value) for value in values)


def _format_parameter(value: Any) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, list):
        return "[" + ",".join(str(item) for item in value) + "]"
    return str(value)


def _parameter_category(name: str) -> str:
    """Classify expression parameters for stable semantic highlighting."""

    if name in {"shape", "layout", "storage", "sharding"}:
        return "structure"
    if name in {"dtype", "role", "kind", "implementation", "opcode"}:
        return "type"
    if name in {"ranks", "rank", "replicated", "collective", "participants", "queue", "ordered"}:
        return "topology"
    if name in {
        "ops",
        "read_B",
        "write_B",
        "message_B",
        "size_B",
        "capacity_bytes",
        "data_bytes",
        "offset",
        "alignment",
        "alignment_bytes",
        "count",
    }:
        return "workload"
    if name in {"relation", "cardinality"}:
        return "mapping"
    return "property"


def _semantic_expression_name(
    label: str,
    nodes: list[IRGraphNode],
    display_aliases: dict[str, str] | None = None,
) -> str:
    if len(nodes) == 1:
        return _display_label(nodes[0], display_aliases or {})
    if nodes:
        phase, subsystem, _ = _structure_parts(nodes[0], nodes[0].ref.stage)
        family = {"mlp": "MLP", "attention": "Attention"}.get(subsystem, subsystem)
        if family and phase not in {"model", "input", "output", "unscoped"}:
            return f"{family}.{phase}"
        if family:
            return family
    return label.replace(" · ", ".")


def _short_entity_expression_parts(
    label: str,
    kind: str,
    nodes: list[IRGraphNode],
    count: int,
    display_aliases: dict[str, str] | None = None,
) -> tuple[str, tuple[dict[str, str], ...]]:
    """Build structured, stage-correct tokens for one semantic expression."""

    parameters: list[tuple[str, str]] = []

    def add_parameter(name: str, value: Any) -> None:
        parameters.append((name, _format_parameter(value)))

    for name in ("shape", "dtype", "role", "storage", "sharding"):
        values = _unique_property(nodes, name)
        if len(values) == 1:
            add_parameter(name, values[0])
        elif values:
            add_parameter(name, values)

    if len(nodes) == 1:
        for name in (
            "phase",
            "source_layer",
            "queue",
            "implementation",
            "opcode",
            "operands",
            "rank",
            "ordered",
            "offset",
            "alignment",
            "alignment_bytes",
            "capacity_bytes",
            "data_bytes",
        ):
            values = _unique_property(nodes, name)
            if len(values) == 1:
                add_parameter(name, values[0])
        if nodes[0].ref.kind in {"command", "queue", "section"}:
            kinds = _unique_property(nodes, "kind")
            if len(kinds) == 1:
                add_parameter("kind", kinds[0])

    rank_counts = _unique_property(nodes, "rank_count") or _unique_property(nodes, "owner_count")
    if len(rank_counts) == 1:
        add_parameter("ranks", rank_counts[0])
    elif rank_counts:
        add_parameter("ranks", rank_counts)

    replicated_axes = _unique_property(nodes, "replicated_axes")
    if replicated_axes:
        add_parameter("replicated", replicated_axes[0])

    collective_kinds = _unique_property(nodes, "collective_kind")
    if collective_kinds:
        value = collective_kinds[0] if len(collective_kinds) == 1 else collective_kinds
        add_parameter("collective", value)
    participant_counts = _unique_property(nodes, "participants")
    if len(participant_counts) == 1:
        add_parameter("participants", participant_counts[0])

    for property_name, parameter_name in (
        ("operations", "ops"),
        ("read_bytes", "read_B"),
        ("write_bytes", "write_B"),
        ("message_bytes", "message_B"),
        ("size_bytes", "size_B"),
    ):
        total = _sum_integer_property(nodes, property_name)
        if total is not None and total != 0:
            add_parameter(parameter_name, total)

    if count > 1:
        add_parameter("count", count)
    if not nodes and count:
        add_parameter("count", count)
    name = _semantic_expression_name(label, nodes, display_aliases)
    return name, tuple({"name": key, "value": value, "category": _parameter_category(key)} for key, value in parameters)


def _short_entity_expression(
    label: str,
    kind: str,
    nodes: list[IRGraphNode],
    count: int,
    display_aliases: dict[str, str] | None = None,
) -> str:
    """Render the plain-text equivalent of a structured semantic expression."""

    name, parameters = _short_entity_expression_parts(label, kind, nodes, count, display_aliases)
    body = ", ".join(f"{item['name']}={item['value']}" for item in parameters)
    return f"{name}({body})" if parameters else name


def lowering_narrative(
    boundary: IRBoundaryView,
    source_graph: IRGraphView | None = None,
    target_graph: IRGraphView | None = None,
) -> LoweringNarrative:
    """Turn cardinality and lineage into a short lowering explanation."""

    source_nodes = _node_lookup(source_graph)
    target_nodes = _node_lookup(target_graph)
    source_aliases = _graph_display_aliases(source_graph)
    expansions: dict[str, set[str]] = {}
    kinds: dict[str, int] = {}
    transforms: dict[str, int] = {}
    for relation in boundary.relations:
        target_node = target_nodes.get(relation.target.key) or target_nodes.get(relation.target.entity_id)
        target_kind = target_node.ref.kind if target_node is not None else relation.target.kind
        kinds[target_kind] = kinds.get(target_kind, 0) + 1
        transform = relation.transform or boundary.pass_name
        transforms[transform] = transforms.get(transform, 0) + 1
        for source in relation.sources:
            expansions.setdefault(source.entity_id, set()).add(relation.target.entity_id)

    largest_source, largest_targets = max(expansions.items(), key=lambda item: len(item[1]), default=("", set()))
    source_node = source_nodes.get(largest_source)
    source_label = _display_label(source_node, source_aliases) if source_node is not None else largest_source
    kind_text = (
        " · ".join(f"{count:,} {kind}" for kind, count in sorted(kinds.items(), key=lambda item: (-item[1], item[0])))
        or "无目标实体"
    )
    primary_transform, transform_count = max(
        transforms.items(), key=lambda item: item[1], default=(boundary.pass_name, 0)
    )
    summary = boundary.summary
    if len(largest_targets) > 1:
        headline = f"{source_label} 被展开为 {len(largest_targets):,} 个目标实体"
        shape = "语义展开（1:N）"
    elif summary.many_to_one > summary.one_to_many:
        headline = "多个 source 语义被融合到较少的目标实体"
        shape = "语义融合（N:1）"
    else:
        headline = "lowering 主要保持 source 与 target 的一一对应"
        shape = "结构保持（1:1）"
    detail = (
        f"目标结构：{kind_text}。{summary.mapped_target_entities:,}/{summary.target_entities:,} 个目标实体"
        f"具有可解析 typed lineage；{summary.dangling_sources:,} 条 source 引用悬空。"
    )
    return LoweringNarrative(
        headline,
        detail,
        (
            MetricView("Lowering 形态", shape, f"最大展开 {len(largest_targets):,} 个 target", "primary"),
            MetricView("目标结构", kind_text, f"共 {summary.target_entities:,} 个 canonical entity", "cyan"),
            MetricView(
                "Lineage 完整性",
                f"{summary.mapped_target_entities:,}/{summary.target_entities:,}",
                f"dangling source · {summary.dangling_sources:,}",
                "violet" if summary.dangling_sources == 0 else "amber",
            ),
            MetricView("主要 transform", primary_transform, f"覆盖 {transform_count:,} 个 target", "neutral"),
        ),
    )


def semantic_boundary_rows(
    boundary: IRBoundaryView,
    source_graph: IRGraphView | None = None,
    target_graph: IRGraphView | None = None,
) -> list[dict[str, Any]]:
    """Aggregate typed lineage into readable source/transform/target table rows."""

    relations = boundary.relations
    source_nodes = _node_lookup(source_graph)
    target_nodes = _node_lookup(target_graph)
    source_aliases = _graph_display_aliases(source_graph)
    target_aliases = _graph_display_aliases(target_graph)
    distinct_sources = {item.entity_id for relation in relations for item in relation.sources}
    keep_source_entities = len(distinct_sources) <= 12
    grouped: dict[tuple[str, str, str, str, str, str], dict[str, Any]] = {}
    for relation in relations:
        target_node = target_nodes.get(relation.target.key) or target_nodes.get(relation.target.entity_id)
        if target_node is None:
            target_label, target_kind, target_order = relation.target.kind, relation.target.kind, 45
            target_example = relation.target.entity_id
        else:
            target_label, target_kind, target_order = _semantic_boundary_bucket(target_node)
            target_example = _display_label(target_node, target_aliases)
            if target_example == relation.target.entity_id:
                target_example = target_label
        source_refs = relation.sources or (None,)
        for source_ref in source_refs:
            source_node = (
                source_nodes.get(source_ref.key) or source_nodes.get(source_ref.entity_id)
                if source_ref is not None
                else None
            )
            if source_node is None:
                source_label = "generated / no source" if source_ref is None else source_ref.entity_id
                source_kind = "generated" if source_ref is None else source_ref.kind
                source_order = 45
            else:
                grouped_source_label, grouped_source_kind, source_order = _semantic_boundary_bucket(source_node)
                if keep_source_entities:
                    source_label = _display_label(source_node, source_aliases)
                    source_kind = source_node.ref.kind
                else:
                    source_label, source_kind = grouped_source_label, grouped_source_kind
            transform = relation.transform or boundary.pass_name
            key = (
                source_label,
                source_kind,
                transform,
                target_label,
                target_kind,
                relation.lineage_kind,
            )
            item = grouped.setdefault(
                key,
                {
                    "source_entities": set(),
                    "target_entities": set(),
                    "target_examples": set(),
                    "relation_count": 0,
                    "source_order": source_order,
                    "target_order": target_order,
                },
            )
            if source_ref is not None:
                item["source_entities"].add(source_ref.entity_id)
            item["target_entities"].add(relation.target.entity_id)
            item["target_examples"].add(target_example)
            item["relation_count"] += 1

    rows = []
    for (source, source_kind, transform, target, target_kind, lineage_kind), item in grouped.items():
        source_count = len(item["source_entities"])
        target_count = len(item["target_entities"])
        if source_count == 0:
            mapping_kind = "生成"
        elif source_count == 1 and target_count > 1:
            mapping_kind = "展开"
        elif source_count > 1 and target_count == 1:
            mapping_kind = "融合"
        elif source_count == target_count:
            mapping_kind = "保持"
        else:
            mapping_kind = "重组"
        examples = sorted(item["target_examples"])
        target_member_nodes = [
            target_nodes[identifier] for identifier in item["target_entities"] if identifier in target_nodes
        ]
        target_expression_name, target_expression_parameters = _short_entity_expression_parts(
            target,
            target_kind,
            target_member_nodes,
            target_count,
            target_aliases,
        )
        rows.append(
            {
                "source": source,
                "source_kind": source_kind,
                "transform": transform,
                "target": target,
                "target_kind": target_kind,
                "mapping": f"{source_count:,} → {target_count:,}",
                "mapping_kind": mapping_kind,
                "source_count": source_count,
                "target_count": target_count,
                "source_entity_ids": tuple(sorted(item["source_entities"])),
                "target_entity_ids": tuple(sorted(item["target_entities"])),
                "relation_count": item["relation_count"],
                "target_examples": " · ".join(examples[:3]) + (" · …" if len(examples) > 3 else ""),
                "target_expression": _short_entity_expression(
                    target,
                    target_kind,
                    target_member_nodes,
                    target_count,
                    target_aliases,
                ),
                "target_expression_name": target_expression_name,
                "target_expression_parameters": target_expression_parameters,
                "target_type": _entity_type_label(target_kind),
                "target_tone": _entity_tone(target_kind),
                "lineage_kind": lineage_kind,
                "source_order": item["source_order"],
                "target_order": item["target_order"],
            }
        )
    rows = sorted(
        rows,
        key=lambda row: (
            row["source_order"],
            row["source"],
            row["transform"],
            row["target_order"],
            row["target"],
            row["target_kind"],
        ),
    )
    totals: dict[tuple[str, str, str], dict[str, set[str]]] = {}
    for row in rows:
        group_key = (row["source"], row["source_kind"], row["transform"])
        total = totals.setdefault(group_key, {"source_entities": set(), "target_entities": set()})
        total["source_entities"].update(row["source_entity_ids"])
        total["target_entities"].update(row["target_entity_ids"])

    previous_source = ""
    previous_group = ""
    for row in rows:
        group_key = (row["source"], row["source_kind"], row["transform"])
        group_id = "\u241f".join(group_key)
        total = totals[group_key]
        source_count = len(total["source_entities"])
        target_count = len(total["target_entities"])
        if source_count == 0:
            mapping_kind = "生成"
        elif source_count == 1 and target_count > 1:
            mapping_kind = "展开"
        elif source_count > 1 and target_count == 1:
            mapping_kind = "融合"
        elif source_count == target_count:
            mapping_kind = "保持"
        else:
            mapping_kind = "重组"
        source_member_nodes = [
            source_nodes[identifier] for identifier in total["source_entities"] if identifier in source_nodes
        ]
        source_expression_name, source_expression_parameters = _short_entity_expression_parts(
            row["source"],
            row["source_kind"],
            source_member_nodes,
            source_count,
            source_aliases,
        )
        row.update(
            {
                "source_span_key": f"{group_id}\u241fsource",
                "transform_span_key": f"{group_id}\u241ftransform",
                "mapping_span_key": f"{group_id}\u241fmapping",
                "mapping_kind_span_key": f"{group_id}\u241frelation",
                "mapping": f"{source_count:,} → {target_count:,}",
                "mapping_kind": mapping_kind,
                "source_expression": _short_entity_expression(
                    row["source"],
                    row["source_kind"],
                    source_member_nodes,
                    source_count,
                    source_aliases,
                ),
                "source_expression_name": source_expression_name,
                "source_expression_parameters": source_expression_parameters,
                "source_type": _entity_type_label(row["source_kind"]),
                "source_tone": _entity_tone(row["source_kind"]),
                "source_group_start": row["source"] != previous_source,
                "transform_group_start": group_id != previous_group,
            }
        )
        previous_source = row["source"]
        previous_group = group_id
    return rows


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
        layer_nodes: list[dict[str, Any]] = []
        for layer_name, operations in layers.items():
            operation_nodes = [
                {"name": operation.replace("_", " "), "value": raw_seconds * scale}
                for operation, raw_seconds in operations.items()
            ]
            layer_nodes.append(
                {
                    "name": layer_name,
                    "value": sum(raw_seconds * scale for raw_seconds in operations.values()),
                    "children": operation_nodes,
                }
            )
        children.append(
            {
                "name": group_name,
                "value": sum(
                    sum(raw_seconds * scale for raw_seconds in operations.values()) for operations in layers.values()
                ),
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

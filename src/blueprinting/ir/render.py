"""IR Render Module - 提供 IR 的终端、树形、表格渲染功能.

使用方式:
    1. 直接调用函数: render_terminal(ir), render_tree_html(ir), render_table(ir)
    2. 通过 Mixin: IR 类型继承 RenderMixin 后可使用 ir.to_terminal(), ir.to_tree_html(), ir.to_table()

支持的 IR 类型:
    - GraphIR
    - ScheduleIR
    - TimelineIR
    - SimulationResult
"""

from __future__ import annotations

import html
from io import StringIO
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.tree import Tree
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

if TYPE_CHECKING:
    from .types import GraphIR, ScheduleIR, TimelineIR
    from .result import SimulationResult


# ==============================================================================
# 格式化工具函数
# ==============================================================================


def _format_bytes(val: Union[int, float, None]) -> str:
    """格式化字节数为人类可读格式."""
    if val is None:
        return "-"
    if not isinstance(val, (int, float)):
        return str(val)
    if val >= 1e12:
        return f"{val/1e12:.2f} TB"
    if val >= 1e9:
        return f"{val/1e9:.2f} GB"
    if val >= 1e6:
        return f"{val/1e6:.2f} MB"
    if val >= 1e3:
        return f"{val/1e3:.2f} KB"
    return f"{val:.0f} B"


def _format_time(val: Union[int, float, None]) -> str:
    """格式化时间为人类可读格式."""
    if val is None:
        return "-"
    if not isinstance(val, (int, float)):
        return str(val)
    if val >= 1:
        return f"{val:.2f} s"
    if val >= 1e-3:
        return f"{val*1e3:.2f} ms"
    return f"{val*1e6:.2f} µs"


def _safe_value(val: Any) -> Any:
    """安全获取值，处理 sympy 表达式."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return val
    # 尝试转换 sympy 表达式
    try:
        return float(val)
    except (TypeError, ValueError):
        return str(val)



# RenderMixin - 为 IR 类型提供渲染方法（完整实现）
# ==============================================================================


class GraphRenderMixin:
    """GraphIR 的渲染 Mixin.
    
    提供方法:
        - to_terminal(): Rich 终端格式
        - to_tree_html(): HTML 树结构
        - to_table(): DataFrame 表格
        - _repr_html_(): Jupyter 自动渲染
    """
    
    def to_terminal(self, max_depth: int = 4, max_children: int = 10) -> str:
        """渲染为 Rich 终端格式."""
        graph = self  # type: ignore
        if not HAS_RICH:
            return f"GraphIR({graph.name!r}, blocks={graph.count_blocks()})"
        
        console = Console(file=StringIO(), force_terminal=True, width=120)
        
        # 创建树形结构
        tree = Tree(f"[bold blue]GraphIR[/bold blue] '{graph.name}'", guide_style="dim")
        
        def _add_block_to_tree(parent_tree: Tree, block, depth: int = 0):
            if depth > max_depth:
                parent_tree.add("[dim]...[/dim]")
                return
            
            block_type = getattr(block, 'block_type', 'Block')
            block_name = getattr(block, 'name', '?')
            children = getattr(block, 'children', [])
            has_params = getattr(block, 'has_params', False)
            shard = block.attrs.get('shard', '') if hasattr(block, 'attrs') else ''
            
            type_colors = {
                "Transformer": "blue", "TransformerLayer": "magenta",
                "Attention": "red", "FFN": "yellow",
                "Linear": "green", "RMSNorm": "cyan", "GELU": "cyan",
            }
            color = type_colors.get(block_type, "white")
            
            label_parts = [f"[{color}]{block_type}[/{color}] '{block_name}'"]
            if has_params:
                label_parts.append("[dim](params)[/dim]")
            if shard:
                label_parts.append(f"[yellow]({shard})[/yellow]")
            if children:
                label_parts.append(f"[dim](children={len(children)})[/dim]")
            
            block_branch = parent_tree.add(" ".join(label_parts))
            
            for i, child in enumerate(children):
                if i >= max_children:
                    block_branch.add(f"[dim]... ({len(children) - max_children} more)[/dim]")
                    break
                _add_block_to_tree(block_branch, child, depth + 1)
        
        if graph.root:
            _add_block_to_tree(tree, graph.root)
        console.print(tree)
        
        # Block 统计表格
        table = Table(title="Block 统计", box=box.ROUNDED, show_header=True, header_style="bold magenta")
        table.add_column("Block 类型", style="cyan")
        table.add_column("数量", justify="right")
        table.add_column("有参数", justify="center")
        
        block_stats: Dict[str, Dict] = {}
        for block in graph.iter_blocks():
            bt = block.block_type
            if bt not in block_stats:
                block_stats[bt] = {"count": 0, "with_params": 0}
            block_stats[bt]["count"] += 1
            if block.has_params:
                block_stats[bt]["with_params"] += 1
        
        for bt, stats in sorted(block_stats.items()):
            table.add_row(bt, str(stats["count"]), f"{stats['with_params']}" if stats['with_params'] else "-")
        
        console.print()
        console.print(table)
        
        return console.file.getvalue()
    
    def to_tree_html(self, max_children: int = 10) -> str:
        """渲染为 HTML 树结构."""
        graph = self  # type: ignore
        if not graph.root:
            return "<div>GraphIR (empty)</div>"
        
        type_colors = {
            "Transformer": "#3b82f6", "TransformerLayer": "#8b5cf6",
            "Attention": "#ec4899", "FFN": "#f59e0b",
            "Linear": "#22c55e", "RMSNorm": "#06b6d4",
            "GELU": "#06b6d4", "Embedding": "#8b5cf6", "Module": "#1f2937",
        }
        
        def _render_block(block, depth: int = 0) -> str:
            block_type = html.escape(getattr(block, 'block_type', 'Block'))
            block_name = html.escape(getattr(block, 'name', '?'))
            children = getattr(block, 'children', [])
            has_params = getattr(block, 'has_params', False)
            attrs = getattr(block, 'attrs', {})
            shard = attrs.get('shard', '')
            
            color = type_colors.get(block_type, "#6b7280")
            
            badges = []
            if has_params:
                badges.append('<span style="color:#22c55e;margin-left:4px;">●</span>')
            if shard:
                badges.append(f'<span style="color:#f59e0b;margin-left:4px;">[{html.escape(shard)}]</span>')
            badges_html = "".join(badges)
            
            label = f'<span style="color:{color};font-weight:500;">{block_type}</span>(<span style="color:#6b7280;">{block_name}</span>){badges_html}'
            
            if not children:
                return f'<li>{label}</li>'
            
            children_html = []
            for child in children[:max_children]:
                children_html.append(_render_block(child, depth + 1))
            
            if len(children) > max_children:
                children_html.append(
                    f'<li><details><summary style="color:#9ca3af;cursor:pointer;">... ({len(children) - max_children} more)</summary>'
                    f'<ul style="margin:0;padding-left:16px;">'
                )
                for child in children[max_children:]:
                    children_html.append(_render_block(child, depth + 1))
                children_html.append('</ul></details></li>')
            
            open_tag = ' open' if depth < 2 else ''
            return f'''<li><details{open_tag}>
                <summary style="cursor:pointer;">{label} <span style="color:#9ca3af;">({len(children)})</span></summary>
                <ul style="margin:0;padding-left:16px;">{"".join(children_html)}</ul>
            </details></li>'''
        
        total_blocks = graph.count_blocks()
        params_blocks = graph.count_params_blocks()
        
        lines = [
            "<div style='font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px;'>",
            "<ul style='margin:0; padding-left:16px;'>",
            f"<li><details open><summary style='cursor:pointer;font-weight:600;'>GraphIR({html.escape(graph.name)}) "
            f"<span style='color:#6b7280;font-weight:normal;'>(blocks={total_blocks}, params={params_blocks})</span></summary>",
            "<ul style='margin:0; padding-left:16px;'>",
            _render_block(graph.root, 0),
            "</ul></details></li>", "</ul></div>"
        ]
        return "".join(lines)
    
    def to_table(self, max_rows: int = 100) -> "pd.DataFrame":
        """渲染为 DataFrame 表格."""
        if not HAS_PANDAS:
            raise ImportError("pandas is required for table rendering")
        
        graph = self  # type: ignore
        rows: List[Dict] = []
        
        def _get_path(block, parent_path: str = "") -> str:
            name = getattr(block, 'name', '?')
            return f"{parent_path}.{name}" if parent_path else name
        
        def _collect_blocks(block, parent_path: str = ""):
            if len(rows) >= max_rows:
                return
            path = _get_path(block, parent_path)
            block_type = getattr(block, 'block_type', 'Block')
            has_params = getattr(block, 'has_params', False)
            attrs = getattr(block, 'attrs', {})
            shard = attrs.get('shard', '')
            children = getattr(block, 'children', [])
            
            rows.append({
                "path": path, "block_type": block_type,
                "has_params": "✓" if has_params else "-",
                "shard": shard or "-", "children": len(children),
            })
            for child in children:
                _collect_blocks(child, path)
        
        if graph.root:
            _collect_blocks(graph.root)
        
        return pd.DataFrame(rows)
    
    def _repr_html_(self) -> str:
        """Jupyter/IPython HTML 渲染."""
        return self.to_tree_html()


class ScheduleRenderMixin:
    """ScheduleIR 的渲染 Mixin.
    
    提供方法:
        - to_terminal(): Rich 终端格式
        - to_tree_html(): HTML 树结构
        - to_table(): DataFrame 表格
        - _repr_html_(): Jupyter 自动渲染
    """
    
    def to_terminal(self, max_ops: int = 10) -> str:
        """渲染为 Rich 终端格式."""
        schedule = self  # type: ignore
        if not HAS_RICH:
            return f"ScheduleIR(stages={len(schedule.stages)}, ops={schedule.total_ops})"
        
        console = Console(file=StringIO(), force_terminal=True, width=120)
        
        num_stages = len(schedule.stages)
        total_ops = sum(len(d.ops) for stage in schedule.stages.values() for d in stage.devices.values())
        
        summary = Panel(
            f"[bold]Stages:[/bold] {num_stages}  |  "
            f"[bold]Devices:[/bold] {schedule.num_devices}  |  "
            f"[bold]Total Ops:[/bold] {total_ops}",
            title="[bold blue]ScheduleIR 概要[/bold blue]",
            border_style="blue"
        )
        console.print(summary)
        
        # Stage 树形结构
        tree = Tree("[bold]调度结构[/bold]", guide_style="dim")
        
        for stage_id, stage in sorted(schedule.stages.items()):
            stage_ops = sum(len(d.ops) for d in stage.devices.values())
            stage_branch = tree.add(f"[cyan]Stage {stage_id}[/cyan] [dim](ops={stage_ops})[/dim]")
            
            for device_id, device in sorted(stage.devices.items()):
                dev_branch = stage_branch.add(f"[yellow]Device {device_id}[/yellow] [dim](ops={len(device.ops)})[/dim]")
                
                for op in device.ops[:3]:
                    phase_color = {"fw": "green", "bw": "red", "opt": "magenta"}.get(op.phase, "white")
                    dur_str = _format_time(op.duration) if isinstance(op.duration, (int, float)) else str(op.duration)
                    dev_branch.add(f"[{phase_color}]{op.op_type}[/{phase_color}] '{op.name}' [dim]seq={op.event_seq}, dur={dur_str}[/dim]")
                
                if len(device.ops) > 3:
                    dev_branch.add(f"[dim]... ({len(device.ops) - 3} more)[/dim]")
        
        console.print(tree)
        
        # Op 表格
        table = Table(title=f"调度 Op 列表 (前{max_ops})", box=box.ROUNDED, show_header=True, header_style="bold")
        table.add_column("Seq", justify="right", style="dim")
        table.add_column("Stage", justify="center")
        table.add_column("Device", justify="center")
        table.add_column("Phase", style="cyan")
        table.add_column("Op Type", style="green")
        table.add_column("Name")
        table.add_column("Duration", justify="right")
        
        for op in list(schedule.iter_ops())[:max_ops]:
            phase_style = {"fw": "green", "bw": "red", "opt": "magenta"}.get(op.phase, "white")
            table.add_row(
                str(op.event_seq), str(op.stage), str(op.device),
                f"[{phase_style}]{op.phase}[/{phase_style}]",
                op.op_type, op.name,
                _format_time(op.duration) if isinstance(op.duration, (int, float)) else str(op.duration),
            )
        
        console.print()
        console.print(table)
        
        return console.file.getvalue()
    
    def to_tree_html(self, max_ops: int = 5) -> str:
        """渲染为 HTML 树结构."""
        schedule = self  # type: ignore
        num_stages = len(schedule.stages)
        lines = [
            "<div style='font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px;'>",
            "<ul style='margin:0; padding-left:16px;'>",
            f"<li>ScheduleIR (stages={num_stages}, devices={schedule.num_devices})",
            "<ul style='margin:0; padding-left:16px;'>",
        ]

        for stage_id, stage in sorted(schedule.stages.items()):
            stage_ops = sum(len(d.ops) for d in stage.devices.values())
            lines.append(f"<li>Stage {stage_id} <span style='color:#6b7280'>(ops={stage_ops})</span><ul style='margin:0; padding-left:16px;'>")
            
            for device_id, device in sorted(stage.devices.items()):
                dev_label = f"Device {device_id} <span style='color:#6b7280'>(ops={len(device.ops)})</span>"
                preview_parts = [f"{html.escape(op.op_type or '')}({html.escape(op.name or '')}) <span style='color:#6b7280'>(seq={op.event_seq})</span>"
                                for op in device.ops[:max_ops]]
                preview = " · ".join(preview_parts)
                
                lines.append("<li><details>")
                separator = ' <span style="color:#9ca3af">—</span> '
                preview_suffix = separator + preview if preview else ''
                lines.append(f"<summary>{dev_label}{preview_suffix}</summary>")
                lines.append("<ul style='margin:0; padding-left:16px;'>")
                
                for op in device.ops[:max_ops]:
                    lines.append(f"<li>{html.escape(op.op_type or '')}({html.escape(op.name or '')}) <span style='color:#6b7280'>(seq={op.event_seq})</span></li>")
                
                if len(device.ops) > max_ops:
                    lines.append(f"<li><details><summary><span style='color:#6b7280'>... ({len(device.ops) - max_ops} more)</span></summary><ul style='margin:0; padding-left:16px;'>")
                    for op in device.ops[max_ops:]:
                        lines.append(f"<li>{html.escape(op.op_type or '')}({html.escape(op.name or '')}) <span style='color:#6b7280'>(seq={op.event_seq})</span></li>")
                    lines.append("</ul></details></li>")
                
                lines.append("</ul></details></li>")
            lines.append("</ul></li>")

        lines.extend(["</ul></li>", "</ul></div>"])
        return "".join(lines)
    
    def to_table(self, max_rows: int = 100) -> "pd.DataFrame":
        """渲染为 DataFrame 表格."""
        if not HAS_PANDAS:
            raise ImportError("pandas is required for table rendering")
        
        schedule = self  # type: ignore
        rows: List[Dict] = []
        for op in schedule.iter_ops():
            rows.append({
                "seq": op.event_seq, "device": op.device, "stage": op.stage,
                "stream": op.stream, "op": op.op_type, "name": op.name,
                "duration": _safe_value(op.duration),
            })
            if len(rows) >= max_rows:
                break
        return pd.DataFrame(rows)
    
    def _repr_html_(self) -> str:
        """Jupyter/IPython HTML 渲染."""
        return self.to_tree_html()


class TimelineRenderMixin:
    """TimelineIR 的渲染 Mixin.
    
    提供方法:
        - to_terminal(): Rich 终端格式
        - to_tree_html(): HTML 树结构
        - to_table(): DataFrame 表格
        - _repr_html_(): Jupyter 自动渲染
    """
    
    def to_terminal(self, max_events: int = 15) -> str:
        """渲染为 Rich 终端格式."""
        timeline = self  # type: ignore
        if not HAS_RICH:
            return f"TimelineIR(events={len(timeline.events)})"
        
        console = Console(file=StringIO(), force_terminal=True, width=120)
        
        events = timeline.events
        peak = max(s.peak for s in timeline.memory_snapshots) if timeline.memory_snapshots else 0
        makespan = timeline.end_time if hasattr(timeline, 'end_time') else 0
        compute_events = [e for e in events if e.stream.value == "compute"]
        comm_events = [e for e in events if e.stream.value == "comm"]
        all_devices = set(e.device for e in events)
        num_devices = len(all_devices)
        
        stats_text = Text()
        stats_text.append("Peak Memory: ", style="bold")
        stats_text.append(f"{_format_bytes(peak)}\n", style="cyan")
        stats_text.append("End Time: ", style="bold")
        stats_text.append(f"{_format_time(makespan) if isinstance(makespan, (int, float)) else str(makespan)}\n", style="green")
        stats_text.append("Compute Events: ", style="bold")
        stats_text.append(f"{len(compute_events)}\n", style="yellow")
        stats_text.append("Comm Events: ", style="bold")
        stats_text.append(f"{len(comm_events)}", style="magenta")
        
        summary = Panel(stats_text, title="[bold blue]TimelineIR[/bold blue]",
                        subtitle=f"Events: {len(events)} | Devices: {num_devices}", border_style="blue")
        console.print(summary)
        
        # 事件表格
        table = Table(title=f"事件列表 (前{max_events})", box=box.ROUNDED, show_header=True)
        table.add_column("Seq", justify="right", style="dim")
        table.add_column("Device", justify="right")
        table.add_column("Time", justify="right")
        table.add_column("Type", style="cyan")
        table.add_column("Stream")
        table.add_column("Op Type", style="green")
        table.add_column("Size", justify="right")
        
        for seq, event in enumerate(events[:max_events]):
            type_color = {"compute_start": "green", "compute_end": "green",
                          "comm_start": "magenta", "comm_end": "magenta",
                          "alloc": "yellow", "free": "red"}.get(event.event_type.value, "white")
            table.add_row(
                str(seq), str(event.device),
                _format_time(event.time) if isinstance(event.time, (int, float)) else str(event.time),
                f"[{type_color}]{event.event_type.value}[/{type_color}]",
                event.stream.value, event.op_type or "-",
                _format_bytes(event.size) if isinstance(event.size, (int, float)) else str(event.size or "-"),
            )
        
        console.print()
        console.print(table)
        
        return console.file.getvalue()
    
    def to_tree_html(self, max_events: int = 5) -> str:
        """渲染为 HTML 树结构."""
        timeline = self  # type: ignore
        events = timeline.events
        if not events:
            return "<div>TimelineIR (empty)</div>"
        
        # 按 Device → Stream 分组
        device_stream_events: Dict[int, Dict[str, List]] = {}
        for event in events:
            device = event.device
            stream = event.stream.value
            if device not in device_stream_events:
                device_stream_events[device] = {}
            if stream not in device_stream_events[device]:
                device_stream_events[device][stream] = []
            device_stream_events[device][stream].append(event)
        
        stream_colors = {"compute": "#22c55e", "comm": "#a855f7", "memory": "#3b82f6"}
        
        lines = [
            "<div style='font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px;'>",
            "<ul style='margin:0; padding-left:16px;'>",
            f"<li><details open><summary style='cursor:pointer;font-weight:600;'>TimelineIR "
            f"<span style='color:#6b7280;font-weight:normal;'>(events={len(events)}, devices={len(device_stream_events)})</span></summary>",
            "<ul style='margin:0; padding-left:16px;'>",
        ]
        
        for device_id in sorted(device_stream_events.keys()):
            streams = device_stream_events[device_id]
            device_total = sum(len(evts) for evts in streams.values())
            lines.append(f"<li><details><summary style='cursor:pointer;'>Device {device_id} <span style='color:#6b7280;'>(events={device_total})</span></summary><ul style='margin:0; padding-left:16px;'>")
            
            for stream_name in sorted(streams.keys()):
                stream_events = streams[stream_name]
                color = stream_colors.get(stream_name, "#6b7280")
                lines.append(f"<li><details><summary style='cursor:pointer;'><span style='color:{color};font-weight:500;'>{stream_name}</span> <span style='color:#6b7280;'>({len(stream_events)} events)</span></summary><ul style='margin:0; padding-left:16px;'>")
                
                for event in stream_events[:max_events]:
                    event_type = html.escape(event.event_type.value)
                    op_type = html.escape(event.op_type or "-")
                    time_str = f"{event.time*1e3:.3f}ms" if isinstance(event.time, (int, float)) else str(event.time)
                    lines.append(f"<li><span style='color:#9ca3af;'>[{time_str}]</span> <span style='color:{color};'>{event_type}</span> {op_type}</li>")
                
                if len(stream_events) > max_events:
                    lines.append(f"<li><details><summary style='color:#9ca3af;cursor:pointer;'>... ({len(stream_events) - max_events} more)</summary><ul style='margin:0; padding-left:16px;'>")
                    for event in stream_events[max_events:]:
                        event_type = html.escape(event.event_type.value)
                        op_type = html.escape(event.op_type or "-")
                        time_str = f"{event.time*1e3:.3f}ms" if isinstance(event.time, (int, float)) else str(event.time)
                        lines.append(f"<li><span style='color:#9ca3af;'>[{time_str}]</span> <span style='color:{color};'>{event_type}</span> {op_type}</li>")
                    lines.append("</ul></details></li>")
                
                lines.append("</ul></details></li>")
            lines.append("</ul></details></li>")
        
        lines.extend(["</ul></details></li>", "</ul></div>"])
        return "".join(lines)
    
    def to_table(self, max_rows: int = 100, device: Optional[int] = None) -> "pd.DataFrame":
        """渲染为 DataFrame 表格."""
        if not HAS_PANDAS:
            raise ImportError("pandas is required for table rendering")
        
        timeline = self  # type: ignore
        rows: List[Dict] = []
        all_events = timeline.events
        events = [e for e in all_events if device is None or e.device == device]
        
        for seq, event in enumerate(events):
            rows.append({
                "seq": seq, "time": _safe_value(event.time),
                "type": event.event_type.value, "device": event.device,
                "stream": event.stream.value, "resource": event.resource_id,
                "size": _safe_value(event.size), "op_type": event.op_type or "-",
            })
            if len(rows) >= max_rows:
                break
        return pd.DataFrame(rows)
    
    def _repr_html_(self) -> str:
        """Jupyter/IPython HTML 渲染."""
        return self.to_tree_html()

class SimulationResultRenderMixin:
    """SimulationResult 的渲染 Mixin.
    
    提供方法:
        - to_terminal(): Rich 终端格式
    """
    
    def to_terminal(self) -> str:
        """渲染为 Rich 终端格式."""
        result = self  # type: ignore
        if not HAS_RICH:
            return f"SimulationResult(peak={result.peak_memory/1e9:.2f}GB, time={result.e2e_time*1e3:.2f}ms)"
        
        console = Console(file=StringIO(), force_terminal=True, width=120)
        
        main_text = Text()
        main_text.append("Peak Memory: ", style="bold")
        main_text.append(f"{result.peak_memory / 1e9:.2f} GB\n", style="cyan bold")
        main_text.append("E2E Time: ", style="bold")
        main_text.append(f"{result.e2e_time * 1e3:.2f} ms\n", style="green bold")
        main_text.append("MFU: ", style="bold")
        main_text.append(f"{result.mfu * 100:.1f}%", style="yellow bold")
        
        console.print(Panel(main_text, title="[bold blue]仿真结果[/bold blue]", border_style="blue"))
        
        # 内存分解
        if result.memory_breakdown:
            mb = result.memory_breakdown
            mem_table = Table(title="内存分解", box=box.ROUNDED, show_header=True, header_style="bold cyan")
            mem_table.add_column("类型", style="cyan")
            mem_table.add_column("大小", justify="right")
            mem_table.add_column("占比", justify="right")
            
            total = mb.total or 1
            for name, val in [("权重", mb.weights), ("激活", mb.activations),
                              ("梯度", mb.gradients), ("优化器", mb.optimizer_states)]:
                pct = val / total * 100 if total else 0
                mem_table.add_row(name, _format_bytes(val), f"{pct:.1f}%")
            mem_table.add_row("[bold]合计[/bold]", f"[bold]{_format_bytes(total)}[/bold]", "[bold]100%[/bold]")
            console.print()
            console.print(mem_table)
        
        # 时间分解
        if result.time_breakdown:
            tb = result.time_breakdown
            time_table = Table(title="时间分解", box=box.ROUNDED, show_header=True, header_style="bold green")
            time_table.add_column("阶段", style="green")
            time_table.add_column("时间", justify="right")
            time_table.add_column("占比", justify="right")
            
            total = tb.total or 1
            for name, val, color in [("前向", tb.forward, "green"), ("反向", tb.backward, "red"),
                                      ("通信", tb.communication, "magenta"), ("气泡", tb.bubble, "yellow"),
                                      ("重计算", getattr(tb, "recompute", 0), "cyan")]:
                if val:
                    pct = val / total * 100 if total else 0
                    time_table.add_row(f"[{color}]{name}[/{color}]", _format_time(val), f"{pct:.1f}%")
            time_table.add_row("[bold]合计[/bold]", f"[bold]{_format_time(total)}[/bold]", "[bold]100%[/bold]")
            console.print()
            console.print(time_table)
        
        return console.file.getvalue()

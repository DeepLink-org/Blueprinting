"""Debug passes - Print IR for debugging and visualization."""

from typing import Union

from .base import Pass
from ..types import GraphIR, ScheduleIR, TimelineIR, BlockNode


class PrintGraphPass(Pass):
    """打印 Graph IR (Block 级别).
    
    输出格式:
    =========
    [GraphIR] model_name
    │
    ├── Transformer(root)
    │   ├── Layer(layer0)
    │   │   ├── Attention(attn) [shard=tp]
    │   │   │   ├── Linear(q_proj)
    │   │   │   └── ...
    │   │   └── FFN(ffn)
    │   │       └── ...
    │   └── ...
    """
    
    def __init__(self, title: str = "Graph IR", verbose: bool = False):
        self.title = title
        self.verbose = verbose
    
    def run(self, ir: GraphIR) -> GraphIR:
        print(self._format(ir))
        return ir
    
    def _format(self, ir: GraphIR) -> str:
        lines = []
        lines.append(f"{'='*60}")
        lines.append(f"[{self.title}] {ir.name}")
        if ir.metadata and self.verbose:
            lines.append(f"  metadata: {ir.metadata}")
        lines.append(f"{'='*60}")
        
        if ir.root:
            lines.extend(self._format_block(ir.root, "", True))
        
        lines.append("")
        return "\n".join(lines)
    
    def _format_block(self, block: BlockNode, prefix: str, is_last: bool) -> list:
        lines = []
        
        # Connector
        connector = "└── " if is_last else "├── "
        
        # Block info
        attrs_str = ""
        if block.attrs:
            key_attrs = ["shard", "in_features", "out_features", "num_heads"]
            shown = {k: v for k, v in block.attrs.items() if k in key_attrs}
            if shown:
                attrs_str = f" [{', '.join(f'{k}={v}' for k, v in shown.items())}]"
        
        lines.append(f"{prefix}{connector}{block.block_type}({block.name}){attrs_str}")
        
        # Children
        child_prefix = prefix + ("    " if is_last else "│   ")
        for i, child in enumerate(block.children):
            is_child_last = (i == len(block.children) - 1)
            lines.extend(self._format_block(child, child_prefix, is_child_last))
        
        return lines


class PrintSchedulePass(Pass):
    """打印 Schedule IR (Op 级别).
    
    显示: seq, Op类型, 时间, 耗时, device, stream, 来源Block
    """
    
    def __init__(self, title: str = "Schedule IR", max_ops: int = 20):
        self.title = title
        self.max_ops = max_ops
    
    def run(self, ir: ScheduleIR) -> ScheduleIR:
        print(self._format(ir))
        return ir
    
    def _format(self, ir: ScheduleIR) -> str:
        lines = []
        lines.append(f"{'='*100}")
        lines.append(f"[{self.title}]")
        lines.append(f"  stages={len(ir.stages)}, devices={ir.num_devices}, ops={ir.total_ops}")
        lines.append(f"{'='*100}")
        
        for stage_id, stage in sorted(ir.stages.items()):
            for device_id, device in sorted(stage.devices.items()):
                lines.append(f"\nStage {stage_id}, Device {device_id}:")
                
                # 表头
                lines.append(f"  {'seq':>4}  {'Op类型':<12}  {'时间':>10}  {'耗时':>10}  {'stream':<8}  {'来源Block'}")
                lines.append(f"  {'-'*4}  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*40}")
                
                for i, op in enumerate(device.ops):
                    if i >= self.max_ops:
                        lines.append(f"\n  ... ({len(device.ops) - self.max_ops} more ops)")
                        break
                    
                    source = op.op.source_block if op.op.source_block else "-"
                    # 简化路径显示
                    if len(source) > 50:
                        source = "..." + source[-47:]
                    
                    lines.append(
                        f"  [{op.event_seq:3d}] {op.op_type:<12}  {op.start*1e3:>8.3f}ms  "
                        f"{op.duration*1e6:>8.1f}μs  {op.stream:<8}  {source}"
                    )
        
        lines.append("")
        return "\n".join(lines)


class PrintTimelinePass(Pass):
    """打印 Timeline IR (Event 级别).
    
    显示: 时间, 事件类型, Op名称, device, stream, 来源Block
    """
    
    def __init__(self, title: str = "Timeline IR", max_events: int = 30):
        self.title = title
        self.max_events = max_events
    
    def run(self, ir: TimelineIR) -> TimelineIR:
        print(self._format(ir))
        return ir
    
    def _format(self, ir: TimelineIR) -> str:
        lines = []
        lines.append(f"{'='*100}")
        lines.append(f"[{self.title}]")
        lines.append(f"  events={len(ir.events)}, end_time={ir.end_time*1e3:.3f}ms")
        lines.append(f"{'='*100}")
        
        # 表头
        lines.append(f"  {'时间':>12}  {'事件类型':<14}  {'Op名称':<15}  {'dev':>3}  {'stream':<8}  {'来源Block'}")
        lines.append(f"  {'-'*12}  {'-'*14}  {'-'*15}  {'-'*3}  {'-'*8}  {'-'*40}")
        
        for i, event in enumerate(ir.events):
            if i >= self.max_events:
                lines.append(f"\n  ... ({len(ir.events) - self.max_events} more events)")
                break
            
            # 获取 source_block (从 metadata)
            source = event.metadata.get("source_block", "-") if event.metadata else "-"
            if source and len(source) > 40:
                source = "..." + source[-37:]
            
            # 格式化时间 (μs)
            time_str = f"{event.time*1e6:>10.1f}μs"
            
            lines.append(
                f"  {time_str}  {event.event_type.value:<14}  {event.resource_id:<15}  "
                f"{event.device:>3}  {event.stream.value:<8}  {source}"
            )
        
        lines.append("")
        return "\n".join(lines)


class PrintResultPass(Pass):
    """打印最终结果."""
    
    def __init__(self, title: str = "Simulation Result"):
        self.title = title
    
    def run(self, ir):
        print(f"{'='*60}")
        print(f"[{self.title}]")
        print(f"{'='*60}")
        print(ir)
        print("")
        return ir

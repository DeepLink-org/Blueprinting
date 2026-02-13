"""TimelinePass - Convert ScheduleIR to TimelineIR.

将 Schedule IR (Op 级别) 转换为 Timeline IR (Event 级别)。

每个 ScheduledOp 展开成:
- COMPUTE_START / COMM_START 事件
- COMPUTE_END / COMM_END 事件

内存事件 (ALLOC / FREE) 从 ScheduleIR.memory_pools 机械翻译，
TimelinePass 自身不做内存估算。
"""

from __future__ import annotations

from ..types import EventType, ScheduledOp, ScheduleIR, StreamType, TimelineEvent, TimelineIR
from .base import Pass


class TimelinePass(Pass):
    """将 ScheduleIR 转换为 TimelineIR.

    除了生成 COMPUTE/COMM 事件，还生成:
    - ALLOC: 内存分配事件 (激活、权重)
    - FREE: 内存释放事件 (前向激活在反向后释放)
    """

    def __init__(self, track_memory: bool = True):
        """初始化 TimelinePass.

        Args:
            track_memory: 是否生成内存事件 (ALLOC/FREE)
        """
        self.track_memory = track_memory

    def run(self, ir: ScheduleIR) -> TimelineIR:
        timeline = TimelineIR(
            metadata=ir.metadata.copy(),
        )

        # 计算并设置 layers_per_stage（上游 Pass 应该设置，这里做保底）
        pp = ir.metadata.get("pp", 1)
        num_layers = ir.metadata.get("num_layers", 1)
        layers_per_stage = num_layers // pp if pp > 0 else num_layers
        timeline.metadata["layers_per_stage"] = layers_per_stage

        # 收集所有 Op
        all_ops = list(ir.iter_ops())

        # 累加 FLOPs
        total_flops = 0
        for op in all_ops:
            if op.op and op.op.flops:
                flops = op.op.flops
                if isinstance(flops, (int, float)):
                    total_flops += flops
        timeline.metadata["total_flops"] = total_flops

        # 遍历所有 Op，生成计算/通信事件
        for op in all_ops:
            self._add_op_events(op, timeline)

        # 从 ScheduleIR.memory_pools 机械生成 ALLOC/FREE 事件
        if self.track_memory:
            self._memory_pools = ir.memory_pools
            self._add_memory_events(all_ops, timeline)

        # 排序事件
        timeline.sort_events()

        return timeline

    def _add_op_events(self, op: ScheduledOp, timeline: TimelineIR) -> None:
        """为单个 Op 添加计算/通信事件."""
        # 判断是计算还是通信
        # 使用 op_type 的基础类型判断（移除 _BW 和 _RE 后缀）
        base_type = (
            op.op_type.replace("_BW", "").replace("_RE", "") if op.op_type else ""
        )
        is_comm = base_type in (
            "AllReduce",
            "AllGather",
            "ReduceScatter",
            "Send",
            "Recv",
        )

        if is_comm:
            start_type = EventType.COMM_START
            end_type = EventType.COMM_END
            stream = StreamType.COMM
        else:
            start_type = EventType.COMPUTE_START
            end_type = EventType.COMPUTE_END
            stream = StreamType.COMPUTE

        # 构建 metadata
        # 包含 source_block 和 Op 的 attrs（如 P2P 信息）
        event_metadata = {"source_block": op.op.source_block if op.op else None}
        if op.op and op.op.attrs:
            # 传递 P2P 相关的 attrs
            for key in ("from_stage", "to_stage", "mb", "data", "micro_batch"):
                if key in op.op.attrs:
                    event_metadata[key] = op.op.attrs[key]

        # Start 事件
        timeline.add_event(
            TimelineEvent(
                time=op.start,
                event_type=start_type,
                resource_id=op.name,
                device=op.device,
                stream=stream,
                op_type=op.op_type,
                phase=op.phase,
                metadata=event_metadata,
            )
        )

        # End 事件
        timeline.add_event(
            TimelineEvent(
                time=op.end,
                event_type=end_type,
                resource_id=op.name,
                device=op.device,
                stream=stream,
                op_type=op.op_type,
                phase=op.phase,
                metadata=event_metadata.copy(),
            )
        )

    def _add_memory_events(self, ops: list[ScheduledOp], timeline: TimelineIR) -> None:
        """从 ScheduleIR.memory_pools 机械生成 ALLOC/FREE 事件.

        TimelinePass 不做任何内存估算，只翻译上游 Schedule Pass 已经计算好的
        memory_pools。训练/推理的内存语义差异全部在 Schedule Pass 层面解决。
        """
        # 从 metadata 取出 memory_pools（ScheduleIR → TimelineIR 时 metadata 已 copy）
        # 但 memory_pools 是 ScheduleIR 上的专用字段，需要通过 _source_ir 传入
        memory_pools = getattr(self, "_memory_pools", [])

        for pool in memory_pools:
            if pool.size_bytes <= 0:
                continue

            # ALLOC 事件
            timeline.add_event(
                TimelineEvent(
                    time=pool.alloc_time,
                    event_type=EventType.ALLOC,
                    resource_id=pool.name,
                    device=pool.device,
                    stream=StreamType.MEMORY,
                    metadata={
                        "bytes": pool.size_bytes,
                        "type": pool.mem_type,
                        **(pool.metadata or {}),
                    },
                )
            )

            # FREE 事件（仅当有释放时间时）
            if pool.free_time is not None:
                timeline.add_event(
                    TimelineEvent(
                        time=pool.free_time,
                        event_type=EventType.FREE,
                        resource_id=pool.name,
                        device=pool.device,
                        stream=StreamType.MEMORY,
                        metadata={
                            "bytes": pool.size_bytes,
                            "type": pool.mem_type,
                        },
                    )
                )



"""IR Types - Three-layer IR definitions for blueprinting compiler.

================================================================================
Three-Layer IR Architecture
================================================================================

Layer 1: Graph IR (Block-level)
    描述模型结构，纯 Block 组成，不包含 Op

    Module → Block → Block → ...

    特点:
    - 高层抽象，描述模型的逻辑结构
    - Block 可以嵌套 Block
    - 不包含具体的计算 Op
    - BlockNode 关联 ops.py 中的 BlockDef
    - Pass 操作: 并行策略、重计算标记、Block 融合

Layer 2: Schedule IR (Op-level)
    描述执行计划，由 Block 展开成 Op 序列

    Stage → Device → Op → Op → ...

    特点:
    - Block 通过 BlockDef.__call__ 方法展开成 Op 序列
    - OpNode 关联 ops.py 中的 OpDef
    - 包含时序信息 (start, duration)
    - Pass 操作: 计算/通信重叠、Op 融合、流分配

Layer 3: Timeline IR (Event-level)
    描述执行时间线，细粒度事件流

    Event → Event → Event → ...

    特点:
    - 每个 Op 展开成多个事件 (start, end, alloc, free)
    - 用于精确的内存和时间模拟
    - 支持多流并行建模

================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Iterator

from sympy import Expr, Symbol

# 延迟导入避免循环依赖
if TYPE_CHECKING:
    from .ops import BlockDef, OpDef


# ==============================================================================
# Common Enums
# ==============================================================================


class Phase(Enum):
    """训练阶段枚举.

    用于区分 Op 属于哪个训练阶段:
    - FORWARD: 前向传播
    - BACKWARD: 反向传播 (包括 recompute, agrad, wgrad)
    - OPTIMIZER: 优化器更新
    """

    FORWARD = "forward"
    BACKWARD = "backward"
    OPTIMIZER = "optimizer"


class NodeType(Enum):
    """节点类型枚举 (用于兼容旧 API)."""

    MODULE = "module"
    BLOCK = "block"
    OP = "op"


@dataclass
class TensorRef:
    """张量引用.

    Attributes:
        name: 张量名称
        shape: 形状 (可包含符号表达式)
        dtype: 数据类型
        producer: 生产者 Op 路径
    """

    name: str
    shape: list[int | Expr] | None = None
    dtype: str = "float16"
    producer: str | None = None

    @property
    def nbytes(self) -> int | Expr:
        """计算字节数."""
        from functools import reduce
        from operator import mul

        dtype_sizes = {
            "float8": 1,
            "float16": 2,
            "bfloat16": 2,
            "float32": 4,
            "float64": 8,
        }
        if self.shape is None:
            return 0
        return reduce(mul, self.shape, 1) * dtype_sizes.get(self.dtype, 2)


# ==============================================================================
# Layer 1: Graph IR (Block-level)
# ==============================================================================


@dataclass
class BlockNode:
    """Block 节点 - Graph IR 的基本单元.

    Block 是模型结构的抽象，关联 ops.py 中的 BlockDef。

    可以表示:
    - 有参数的层: Linear, RMSNorm, Embedding (BlockDef.has_params() == True)
    - 结构容器: Attention, FFN, TransformerLayer (BlockDef.has_params() == False)

    Block 不包含具体的 Op，Op 在 Schedule 阶段通过
    BlockDef.__call__() 方法展开生成。

    Attributes:
        name: Block 名称 (在父节点内唯一)
        block_type: Block 类型 (对应 ops.py 中的 BlockDef.block_type)
        children: 子 Block 列表
        attrs: Block 属性 (如 in_features, out_features, shard 等)
        device: 设备分配
    """

    name: str
    block_type: str
    children: list[BlockNode] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)
    device: int | None = None

    @property
    def block_def(self) -> BlockDef | None:
        """获取关联的 BlockDef 类."""
        from .ops import get_block_def

        return get_block_def(self.block_type)

    @property
    def has_params(self) -> bool:
        """是否有可学习参数."""
        block_def = self.block_def
        return block_def.has_params() if block_def else False

    @property
    def params(self) -> list[str]:
        """获取参数名称列表."""
        block_def = self.block_def
        return block_def.params if block_def else []

    def add_child(self, child: BlockNode) -> BlockNode:
        """添加子 Block."""
        self.children.append(child)
        return child

    def iter_blocks(self) -> Iterator[BlockNode]:
        """深度优先遍历所有 Block."""
        yield self
        for child in self.children:
            yield from child.iter_blocks()

    def compute_flops(self) -> int | Expr | None:
        """计算 FLOPs (委托给 BlockDef)."""
        block_def = self.block_def
        if block_def:
            return block_def.compute_flops(self.attrs)
        return None

    def compute_param_bytes(self) -> int | Expr | None:
        """计算参数内存 (委托给 BlockDef)."""
        block_def = self.block_def
        if block_def:
            return block_def.compute_param_bytes(self.attrs)
        return None

    def __repr__(self) -> str:
        parts = [f"{self.block_type}({self.name!r})"]
        if self.children:
            parts.append(f"children={len(self.children)}")
        if self.attrs.get("shard"):
            parts.append(f"shard={self.attrs['shard']}")
        if self.has_params:
            parts.append(f"params={self.params}")
        return f"Block({', '.join(parts)})"


@dataclass
class GraphIR:
    """Graph IR - Block 级别的模型表示.

    Graph IR 是编译器的输入，描述模型的逻辑结构。
    它只包含 BlockNode，不包含具体的计算 Op。

    结构: Module (root) → Block → Block → ...

    Attributes:
        name: 模型名称
        root: 根 Block (代表整个模型)
        metadata: 模型元数据 (batch_size, seq_len, hidden 等)
        symbols: 符号表 (用于符号计算)
    """

    name: str = "model"
    root: BlockNode | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    symbols: dict[str, Symbol] = field(default_factory=dict)

    def iter_blocks(self) -> Iterator[BlockNode]:
        """遍历所有 Block."""
        if self.root:
            yield from self.root.iter_blocks()

    def count_blocks(self) -> int:
        """统计 Block 数量."""
        return sum(1 for _ in self.iter_blocks())

    def count_params_blocks(self) -> int:
        """统计有参数的 Block 数量."""
        return sum(1 for b in self.iter_blocks() if b.has_params)

    def __repr__(self) -> str:
        if not self.root:
            return "GraphIR(empty)"
        return f"GraphIR({self.name!r}, blocks={self.count_blocks()})"

    def summary(self) -> str:
        """返回多行概要信息."""
        if not self.root:
            return "GraphIR(empty)"

        lines = [
            f"GraphIR: {self.root.block_type}({self.name})",
            f"├─ blocks: {self.count_blocks()}",
            f"├─ params blocks: {self.count_params_blocks()}",
        ]

        # 顶层结构
        lines.append("├─ structure:")
        for i, child in enumerate(self.root.children[:8]):
            child_blocks = sum(1 for _ in child.iter_blocks())
            lines.append(f"│   {child.block_type}({child.name}): {child_blocks} blocks")
        if len(self.root.children) > 8:
            lines.append(f"│   ... ({len(self.root.children) - 8} more)")

        # Metadata
        meta = ", ".join(list(self.metadata.keys())[:6])
        lines.append(f"└─ metadata: {meta or '(none)'}")

        return "\n".join(lines)

    def tree(self, max_depth: int = 3) -> str:
        """返回树形结构展示."""
        if not self.root:
            return "(empty)"

        lines = []

        def _tree(block: BlockNode, depth: int, prefix: str = "", connector: str = ""):
            if depth > max_depth:
                return

            attrs_str = ""
            if block.attrs.get("shard"):
                attrs_str = f" [{block.attrs['shard']}]"

            lines.append(f"{connector}■ {block.block_type}({block.name}){attrs_str}")

            if depth < max_depth:
                for i, child in enumerate(block.children):
                    is_last = i == len(block.children) - 1
                    child_connector = prefix + ("└─ " if is_last else "├─ ")
                    next_prefix = prefix + ("   " if is_last else "│  ")
                    _tree(child, depth + 1, next_prefix, child_connector)

        _tree(self.root, 0, "", "")
        return "\n".join(lines)


# ==============================================================================
# Layer 2: Schedule IR (Op-level)
# ==============================================================================


@dataclass
class OpNode:
    """Op 节点 - Schedule IR 的基本计算单元.

    Op 是原子计算操作，关联 ops.py 中的 OpDef。
    由 BlockDef.__call__() 在 Schedule 阶段生成。

    Attributes:
        name: Op 名称
        op_type: Op 类型 (对应 ops.py 中的 OpDef.op_type)
        inputs: 输入张量名称列表
        outputs: 输出张量名称列表
        attrs: Op 属性

        # 来源信息
        source_block: 来源 Block 的路径

        # Workload 信息 (委托给 OpDef 计算)
        flops: 计算量
        memory_bytes: 内存访问量
        comm_bytes: 通信量
    """

    name: str
    op_type: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)

    # Source tracking
    source_block: str | None = None

    # Workload (can be computed via OpDef or set directly)
    flops: int | Expr | None = None
    memory_bytes: int | Expr | None = None
    comm_bytes: int | Expr | None = None

    @property
    def op_def(self) -> OpDef | None:
        """获取关联的 OpDef 类."""
        from .ops import get_op_def

        return get_op_def(self.op_type)

    @property
    def category(self) -> str:
        """获取 Op 类别 (compute/activation/comm)."""
        op_def = self.op_def
        return op_def.category if op_def else "compute"

    @property
    def is_comm(self) -> bool:
        """是否是通信 Op."""
        return self.category == "comm"

    def compute_flops(self) -> int | Expr | None:
        """计算 FLOPs (委托给 OpDef)."""
        if self.flops is not None:
            return self.flops
        op_def = self.op_def
        if op_def:
            return op_def.compute_flops(self.attrs)
        return None

    def compute_memory(self) -> int | Expr | None:
        """计算内存访问 (委托给 OpDef)."""
        if self.memory_bytes is not None:
            return self.memory_bytes
        op_def = self.op_def
        if op_def:
            return op_def.compute_memory(self.attrs)
        return None

    def compute_comm(self) -> int | Expr | None:
        """计算通信量 (委托给 OpDef)."""
        if self.comm_bytes is not None:
            return self.comm_bytes
        op_def = self.op_def
        if op_def:
            return op_def.compute_comm(self.attrs)
        return None

    def __repr__(self) -> str:
        parts = [f"{self.op_type}({self.name!r})"]
        if self.source_block:
            parts.append(f"from={self.source_block}")
        return f"Op({', '.join(parts)})"


@dataclass
class TensorLifetime:
    """张量生命周期追踪.

    Attributes:
        tensor_id: 张量 ID
        alloc_time: 分配时间
        free_time: 释放时间
        size: 字节大小
    """

    tensor_id: str
    alloc_time: float | Expr = 0
    free_time: float | Expr | None = None
    size: int | Expr = 0


@dataclass
class ScheduledOp:
    """已调度的 Op - 包含时序信息.

    ScheduledOp 是 OpNode 加上调度信息，用于时间线模拟。

    Attributes:
        op: 底层的 OpNode

        # 调度信息
        device: 设备 ID
        stage: Pipeline 阶段
        stream: 执行流 (compute, comm)
        phase: 训练阶段 (forward, backward, optimizer)

        # 时序信息
        start: 开始时间
        duration: 持续时间
        event_seq: 全局事件序号
    """

    op: OpNode

    # Scheduling
    device: int = 0
    stage: int = 0
    stream: str = "compute"
    phase: Phase = Phase.FORWARD

    # Timing
    start: float | Expr = 0
    duration: float | Expr = 0
    event_seq: int = 0

    @property
    def end(self) -> float | Expr:
        """结束时间."""
        return self.start + self.duration

    @property
    def name(self) -> str:
        return self.op.name

    @property
    def op_type(self) -> str:
        return self.op.op_type

    def __repr__(self) -> str:
        return f"ScheduledOp({self.op.op_type}({self.op.name!r}), dev={self.device}, t={self.start:.4f})"


@dataclass
class DeviceSchedule:
    """单个设备的调度.

    Attributes:
        device_id: 设备 ID
        ops: 该设备上的 Op 列表 (按时间排序)
    """

    device_id: int
    ops: list[ScheduledOp] = field(default_factory=list)

    def add_op(self, op: ScheduledOp) -> None:
        """添加 Op."""
        self.ops.append(op)

    @property
    def end_time(self) -> float | Expr:
        """该设备的结束时间."""
        if not self.ops:
            return 0
        return max(op.end for op in self.ops)

    def __repr__(self) -> str:
        return f"DeviceSchedule(device={self.device_id}, ops={len(self.ops)})"


@dataclass
class StageSchedule:
    """Pipeline 阶段的调度.

    Attributes:
        stage_id: 阶段 ID
        devices: 设备调度字典
    """

    stage_id: int
    devices: dict[int, DeviceSchedule] = field(default_factory=dict)

    def get_device(self, device_id: int) -> DeviceSchedule:
        """获取或创建设备调度."""
        if device_id not in self.devices:
            self.devices[device_id] = DeviceSchedule(device_id)
        return self.devices[device_id]

    def __repr__(self) -> str:
        op_count = sum(len(d.ops) for d in self.devices.values())
        return f"StageSchedule(stage={self.stage_id}, devices={len(self.devices)}, ops={op_count})"


@dataclass
class ScheduleIR:
    """Schedule IR - Op 级别的执行计划.

    Schedule IR 是 Graph IR 经过 Schedule Pass 转换后的结果。
    Block 被展开成 Op 序列，并分配了设备和时序。

    结构: Stage → Device → ScheduledOp

    Attributes:
        stages: 阶段调度字典
        num_devices: 设备总数
        metadata: 元数据
    """

    stages: dict[int, StageSchedule] = field(default_factory=dict)
    num_devices: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    # Event counter
    _event_seq: int = 0

    def get_stage(self, stage_id: int) -> StageSchedule:
        """获取或创建阶段调度."""
        if stage_id not in self.stages:
            self.stages[stage_id] = StageSchedule(stage_id)
        return self.stages[stage_id]

    def add_op(self, op: ScheduledOp, stage: int = 0, device: int = 0) -> None:
        """添加 Op 到指定阶段和设备."""
        op.event_seq = self._event_seq
        self._event_seq += 1
        op.stage = stage
        op.device = device
        self.get_stage(stage).get_device(device).add_op(op)

    def iter_ops(self) -> Iterator[ScheduledOp]:
        """遍历所有 Op (按 event_seq 排序)."""
        all_ops = []
        for stage in self.stages.values():
            for device in stage.devices.values():
                all_ops.extend(device.ops)
        return iter(sorted(all_ops, key=lambda op: op.event_seq))

    @property
    def total_ops(self) -> int:
        """Op 总数."""
        return sum(len(d.ops) for s in self.stages.values() for d in s.devices.values())

    def __repr__(self) -> str:
        return f"ScheduleIR(stages={len(self.stages)}, devices={self.num_devices}, ops={self.total_ops})"


# ==============================================================================
# Layer 3: Timeline IR (Event-level)
# ==============================================================================


class EventType(Enum):
    """事件类型."""

    # 内存事件
    ALLOC = "alloc"
    FREE = "free"

    # 计算事件
    COMPUTE_START = "compute_start"
    COMPUTE_END = "compute_end"

    # 通信事件
    COMM_START = "comm_start"
    COMM_END = "comm_end"

    # 同步事件
    SYNC = "sync"
    BARRIER = "barrier"


class StreamType(Enum):
    """执行流类型."""

    COMPUTE = "compute"  # 主计算流 (CUDA default stream)
    COMM = "comm"  # 通信流
    NCCL = "nccl"  # NCCL 通信流
    MEMORY = "memory"  # 内存操作 (async memcpy)
    H2D = "h2d"  # Host to device 传输
    D2H = "d2h"  # Device to host 传输


@dataclass
class TimelineEvent:
    """时间线事件 - Timeline IR 的基本单元.

    Attributes:
        time: 事件时间
        event_type: 事件类型
        resource_id: 关联资源 (op_name 或 tensor_name)
        device: 设备 ID
        stream: 执行流
        size: 字节数 (用于内存事件)
        op_type: Op 类型 (用于计算/通信事件)
        phase: 训练阶段 (forward, backward, optimizer)
        metadata: 额外元数据
    """

    time: float | Expr
    event_type: EventType
    resource_id: str
    device: int = 0
    stream: StreamType = StreamType.COMPUTE
    size: int | Expr = 0
    op_type: str = ""
    phase: Phase = Phase.FORWARD
    metadata: dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other: TimelineEvent) -> bool:
        """按时间排序."""
        if isinstance(self.time, (int, float)) and isinstance(other.time, (int, float)):
            return self.time < other.time
        return str(self.time) < str(other.time)

    def __repr__(self) -> str:
        return f"Event({self.event_type.value}, {self.resource_id}, t={self.time})"

    def is_memory_event(self) -> bool:
        """是否为内存事件."""
        return self.event_type in (EventType.ALLOC, EventType.FREE)

    def is_compute_event(self) -> bool:
        """是否为计算事件."""
        return self.event_type in (EventType.COMPUTE_START, EventType.COMPUTE_END)

    def is_comm_event(self) -> bool:
        """是否为通信事件."""
        return self.event_type in (EventType.COMM_START, EventType.COMM_END)


@dataclass
class MemorySnapshot:
    """内存快照 - 某时刻的内存状态.

    Attributes:
        time: 时间点
        device: 设备 ID
        allocated: 已分配字节数
        peak: 峰值内存
        tensors: 活跃张量列表
    """

    time: float | Expr
    device: int
    allocated: int | Expr = 0
    peak: int | Expr = 0
    tensors: list[str] = field(default_factory=list)


@dataclass
class StreamState:
    """执行流状态 - 某时刻流的状态.

    Attributes:
        device: 设备 ID
        stream: 流类型
        current_time: 当前时间
        pending_ops: 等待执行的操作列表
    """

    device: int
    stream: StreamType
    current_time: float | Expr = 0
    pending_ops: list[str] = field(default_factory=list)


@dataclass
class TimelineIR:
    """Timeline IR - Event 级别的执行时间线.

    Timeline IR 是 Schedule IR 展开成细粒度事件后的结果。
    用于精确的内存和时间模拟。

    结构: Device → Stream → Event 序列

    Attributes:
        events: 事件列表 (按时间排序)
        memory_snapshots: 内存快照列表
        metadata: 元数据
        num_devices: 设备数量
    
    Example:
        timeline = TimelineIR(num_devices=2)
        timeline.add_event(TimelineEvent(...))
        peak_mem = timeline.peak_memory(device=0)
        makespan = timeline.makespan()
    """

    events: list[TimelineEvent] = field(default_factory=list)
    memory_snapshots: list[MemorySnapshot] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    num_devices: int = 1
    _sorted: bool = field(default=True, repr=False)

    def add_event(self, event: TimelineEvent) -> None:
        """添加事件."""
        self.events.append(event)
        self._sorted = False

    def add_events(self, events: list[TimelineEvent]) -> None:
        """批量添加事件."""
        self.events.extend(events)
        self._sorted = False

    def _ensure_sorted(self) -> None:
        """确保事件按时间排序."""
        if not self._sorted:
            all_numeric = all(isinstance(e.time, (int, float)) for e in self.events)
            if all_numeric:
                self.events.sort(key=lambda e: e.time)
            else:
                def sort_key(e):
                    if isinstance(e.time, (int, float)):
                        return (e.time, "")
                    return (float("inf"), str(e.time))
                self.events.sort(key=sort_key)
            self._sorted = True

    def sort_events(self) -> None:
        """按时间排序事件."""
        self._ensure_sorted()

    def get_events(
        self,
        device: int | None = None,
        stream: StreamType | None = None,
        event_type: EventType | None = None,
    ) -> list[TimelineEvent]:
        """按条件过滤事件."""
        self._ensure_sorted()
        result = self.events
        if device is not None:
            result = [e for e in result if e.device == device]
        if stream is not None:
            result = [e for e in result if e.stream == stream]
        if event_type is not None:
            result = [e for e in result if e.event_type == event_type]
        return result

    @property
    def end_time(self) -> float | Expr:
        """时间线结束时间."""
        if not self.events:
            return 0
        return max(e.time for e in self.events)

    def peak_memory(self, device: int = 0) -> int | Expr:
        """计算设备的峰值内存使用量."""
        self._ensure_sorted()
        memory_events = [
            e for e in self.events if e.device == device and e.is_memory_event()
        ]
        if not memory_events:
            return 0

        all_numeric = all(isinstance(e.size, (int, float)) for e in memory_events)
        if all_numeric:
            current = 0
            peak = 0
            for event in memory_events:
                if event.event_type == EventType.ALLOC:
                    current += event.size
                    peak = max(peak, current)
                elif event.event_type == EventType.FREE:
                    current -= event.size
            return peak
        else:
            total = 0
            for event in memory_events:
                if event.event_type == EventType.ALLOC:
                    total = total + event.size
            return total

    def memory_trace(self, device: int = 0) -> list[MemorySnapshot]:
        """生成内存使用轨迹."""
        self._ensure_sorted()
        memory_events = [
            e for e in self.events if e.device == device and e.is_memory_event()
        ]
        snapshots = []
        current = 0
        live_tensors: list[str] = []

        for event in memory_events:
            if event.event_type == EventType.ALLOC:
                current += event.size if isinstance(event.size, (int, float)) else 0
                live_tensors.append(event.resource_id)
            elif event.event_type == EventType.FREE:
                current -= event.size if isinstance(event.size, (int, float)) else 0
                if event.resource_id in live_tensors:
                    live_tensors.remove(event.resource_id)

            snapshots.append(MemorySnapshot(
                time=event.time,
                device=device,
                allocated=current,
                tensors=list(live_tensors),
            ))
        return snapshots

    def makespan(self, device: int | None = None) -> float | Expr:
        """计算总执行时间 (makespan)."""
        self._ensure_sorted()
        events = (
            self.events if device is None
            else [e for e in self.events if e.device == device]
        )
        if not events:
            return 0

        end_events = [
            e for e in events
            if e.event_type in (EventType.COMPUTE_END, EventType.COMM_END)
        ]
        if not end_events:
            return events[-1].time if events else 0

        all_numeric = all(isinstance(e.time, (int, float)) for e in end_events)
        if all_numeric:
            return max(e.time for e in end_events)
        else:
            return end_events[-1].time

    def compute_time(self, device: int = 0) -> float | Expr:
        """计算设备的总计算时间."""
        compute_events = self.get_events(device=device, stream=StreamType.COMPUTE)
        total: float | Expr = 0
        starts: dict[str, Any] = {}

        for event in compute_events:
            if event.event_type == EventType.COMPUTE_START:
                starts[event.resource_id] = event.time
            elif event.event_type == EventType.COMPUTE_END and event.resource_id in starts:
                duration = event.time - starts[event.resource_id]
                total = total + duration
                del starts[event.resource_id]
        return total

    def comm_time(self, device: int = 0) -> float | Expr:
        """计算设备的总通信时间."""
        comm_events = self.get_events(device=device)
        total: float | Expr = 0
        starts: dict[str, Any] = {}

        for event in comm_events:
            if event.event_type == EventType.COMM_START:
                starts[event.resource_id] = event.time
            elif event.event_type == EventType.COMM_END and event.resource_id in starts:
                duration = event.time - starts[event.resource_id]
                total = total + duration
                del starts[event.resource_id]
        return total

    def bubble_time(self, device: int = 0) -> float | Expr:
        """计算气泡时间 (空闲时间)."""
        total_time = self.makespan(device=device)
        compute = self.compute_time(device=device)
        comm = self.comm_time(device=device)
        return total_time - compute - comm

    def overlap_ratio(self, device: int = 0) -> float:
        """计算计算-通信重叠率."""
        compute = self.compute_time(device=device)
        comm = self.comm_time(device=device)
        makespan = self.makespan(device=device)

        if not isinstance(compute, (int, float)):
            return 0.0
        if not isinstance(comm, (int, float)):
            return 0.0
        if not isinstance(makespan, (int, float)):
            return 0.0

        if comm == 0:
            return 1.0
        overlap = max(0, compute + comm - makespan)
        return overlap / comm if comm > 0 else 0.0

    def summary(self) -> str:
        """生成人类可读的摘要."""
        lines = ["TimelineIR Summary:"]
        lines.append(f"  Events: {len(self.events)}")
        lines.append(f"  Devices: {self.num_devices}")

        for dev in range(self.num_devices):
            lines.append(f"\n  Device {dev}:")
            peak_mem = self.peak_memory(dev)
            if isinstance(peak_mem, (int, float)):
                lines.append(f"    Peak Memory: {peak_mem/1e9:.2f} GB")
            else:
                lines.append(f"    Peak Memory: {peak_mem}")

            makespan_val = self.makespan(dev)
            if isinstance(makespan_val, (int, float)):
                lines.append(f"    Makespan: {makespan_val*1e3:.2f} ms")
            else:
                lines.append(f"    Makespan: {makespan_val}")

            compute = self.compute_time(dev)
            if isinstance(compute, (int, float)):
                lines.append(f"    Compute: {compute*1e3:.2f} ms")

            comm = self.comm_time(dev)
            if isinstance(comm, (int, float)):
                lines.append(f"    Comm: {comm*1e3:.2f} ms")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"TimelineIR(events={len(self.events)}, devices={self.num_devices})"

    def to_chrome_trace(
        self, time_unit: str = "ms", include_blocks: bool = True
    ) -> dict[str, Any]:
        """导出为 Chrome Trace 格式.

        可以在 chrome://tracing 或 https://ui.perfetto.dev 中打开。

        Args:
            time_unit: 时间单位 ("ms" 或 "us")，Chrome Trace 使用微秒
            include_blocks: 是否包含 Block/Layer 级别的聚合信息

        Returns:
            Chrome Trace 格式的字典
        """
        import re
        from collections import defaultdict

        # Chrome Trace 使用微秒
        time_scale = 1000.0 if time_unit == "ms" else 1.0

        trace_events = []

        # Track 分配 (简化版):
        # tid 1: Compute (Forward + Backward + Optimizer 合并)
        # tid 10: Communication (TP AllReduce + P2P Send/Recv)
        # tid 20: Micro-batch 级别聚合
        phase_to_tid = {
            Phase.FORWARD: 1,  # Compute
            Phase.BACKWARD: 1,  # Compute (与 Forward 合并)
            Phase.OPTIMIZER: 1,  # Compute (与 Forward/Backward 合并)
        }

        # 用于 Micro-batch 聚合的数据结构
        microbatch_spans: dict[tuple, list[float]] = defaultdict(list)

        def _eval_time(t) -> float:
            if isinstance(t, (int, float)):
                return float(t)
            if isinstance(t, Expr):
                try:
                    return float(t)
                except:
                    return 0.0
            return 0.0

        def _extract_microbatch(resource_id: str) -> str | None:
            """从 resource_id 提取 micro-batch 编号."""
            match = re.search(r"_mb(\d+)", resource_id)
            if match:
                return f"mb{match.group(1)}"
            return None

        for event in self.events:
            time_val = _eval_time(event.time)
            ts = time_val * time_scale

            # 确定 phase 类型
            ph = None
            if event.event_type == EventType.COMPUTE_START:
                ph = "B"
            elif event.event_type == EventType.COMPUTE_END:
                ph = "E"
            elif event.event_type == EventType.COMM_START:
                ph = "B"
            elif event.event_type == EventType.COMM_END:
                ph = "E"
            elif event.event_type == EventType.ALLOC or event.event_type == EventType.FREE:
                ph = "i"

            if ph is None:
                continue

            # 确定类别
            if event.event_type in (EventType.COMM_START, EventType.COMM_END):
                cat = "communication"
            elif event.event_type in (EventType.ALLOC, EventType.FREE):
                cat = "memory"
            else:
                cat = event.phase.value if event.phase else "compute"

            # 确定 tid (简化版)
            is_p2p = event.op_type in ("Send", "Recv")
            is_comm = event.stream == StreamType.COMM

            if is_p2p:
                tid = 10  # P2P 也放到 Communication track
            elif is_comm:
                tid = 10  # Communication track (TP AllReduce 等)
            else:
                tid = phase_to_tid.get(event.phase, 1)  # Compute or Optimizer

            trace_event = {
                "name": event.resource_id,
                "cat": cat,
                "ph": ph,
                "ts": ts,
                "pid": event.device,
                "tid": tid,
            }

            if event.metadata:
                trace_event["args"] = event.metadata.copy()
            else:
                trace_event["args"] = {}

            trace_event["args"]["op_type"] = event.op_type
            trace_event["args"]["phase"] = event.phase.value if event.phase else ""

            if event.event_type in (EventType.ALLOC, EventType.FREE):
                trace_event["args"]["bytes"] = event.size
                trace_event["s"] = "g"

            trace_events.append(trace_event)

            # 收集 Micro-batch 聚合信息（排除 P2P 通信，只统计计算时间）
            if include_blocks and event.event_type in (
                EventType.COMPUTE_START,
                EventType.COMPUTE_END,
            ):
                # 只收集计算事件，不包括 P2P Send/Recv
                mb = _extract_microbatch(event.resource_id)
                if mb:
                    mb_key = (event.device, mb, event.phase)
                    microbatch_spans[mb_key].append(time_val)

        # 生成 Micro-batch 级别聚合事件
        if include_blocks:
            for key, times in sorted(
                microbatch_spans.items(),
                key=lambda x: (x[0][0], x[0][1], x[0][2].value if x[0][2] else ""),
            ):
                device, mb, phase = key
                if times:
                    start = min(times) * time_scale
                    end = max(times) * time_scale

                    trace_events.append(
                        {
                            "name": f"{mb} ({phase.value})" if phase else mb,
                            "cat": "microbatch",
                            "ph": "X",
                            "ts": start,
                            "dur": end - start,
                            "pid": device,
                            "tid": 20,
                            "args": {
                                "microbatch": mb,
                                "phase": phase.value if phase else "",
                            },
                        }
                    )

        # 生成 Flow events 可视化 P2P 通信
        # 收集 Send 事件，并为每个 Send 生成到对应 Recv 的 Flow
        flow_id = 0
        send_events = {}  # (from_stage, to_stage, mb, data_type) -> (ts, device)

        for event in self.events:
            if event.op_type == "Send" and event.metadata:
                from_stage = event.metadata.get("from_stage")
                to_stage = event.metadata.get("to_stage")
                mb = event.metadata.get("mb")
                data_type = event.metadata.get("data", "")

                if from_stage is not None and to_stage is not None and mb is not None:
                    ts = _eval_time(event.time) * time_scale
                    send_events[(from_stage, to_stage, mb, data_type)] = (
                        ts,
                        event.device,
                    )

        for event in self.events:
            if event.op_type == "Recv" and event.metadata:
                from_stage = event.metadata.get("from_stage")
                to_stage = event.metadata.get("to_stage")
                mb = event.metadata.get("mb")
                data_type = event.metadata.get("data", "")

                key = (from_stage, to_stage, mb, data_type)
                if key in send_events:
                    send_ts, send_device = send_events[key]
                    recv_ts = _eval_time(event.time) * time_scale
                    recv_device = event.device

                    # Flow start (at Send)
                    trace_events.append(
                        {
                            "name": f"p2p_{data_type}",
                            "cat": "p2p_flow",
                            "ph": "s",  # Flow start
                            "ts": send_ts,
                            "pid": send_device,
                            "tid": 10,  # Communication track
                            "id": flow_id,
                            "args": {
                                "from": f"Stage {from_stage}",
                                "to": f"Stage {to_stage}",
                                "mb": mb,
                            },
                        }
                    )

                    # Flow end (at Recv)
                    trace_events.append(
                        {
                            "name": f"p2p_{data_type}",
                            "cat": "p2p_flow",
                            "ph": "f",  # Flow end
                            "ts": recv_ts,
                            "pid": recv_device,
                            "tid": 10,  # Communication track
                            "id": flow_id,
                            "bp": "e",  # Bind to enclosing slice
                            "args": {
                                "from": f"Stage {from_stage}",
                                "to": f"Stage {to_stage}",
                                "mb": mb,
                            },
                        }
                    )

                    flow_id += 1

        # 添加元数据
        metadata = []

        devices = {e.device for e in self.events}
        pp = self.metadata.get("pp", 1)

        for device in sorted(devices):
            # 进程名称：GPU {device} (Stage {device})
            stage_info = f" (Stage {device})" if pp > 1 else ""
            metadata.append(
                {
                    "name": "process_name",
                    "ph": "M",
                    "pid": device,
                    "args": {"name": f"GPU {device}{stage_info}"},
                }
            )

            # 进程排序（确保 GPU 0 在最上面）
            metadata.append(
                {
                    "name": "process_sort_index",
                    "ph": "M",
                    "pid": device,
                    "args": {"sort_index": device},
                }
            )

        # 线程名称 (简化版)
        thread_names = {
            1: "Compute",  # Forward + Backward + Optimizer 合并
            10: "Communication",  # TP AllReduce + P2P Send/Recv
        }
        if include_blocks:
            thread_names[20] = "Micro-batch"

        for tid, name in thread_names.items():
            for device in devices:
                metadata.append(
                    {
                        "name": "thread_name",
                        "ph": "M",
                        "pid": device,
                        "tid": tid,
                        "args": {"name": name},
                    }
                )

        return {
            "traceEvents": metadata + trace_events,
            "displayTimeUnit": "ms",
            "metadata": self.metadata,
        }

    def save_chrome_trace(
        self, path: str, time_unit: str = "ms", include_blocks: bool = True
    ) -> None:
        """保存为 Chrome Trace JSON 文件.

        Args:
            path: 输出文件路径
            time_unit: 时间单位
            include_blocks: 是否包含 Micro-batch 聚合轨道（默认开启，聚合只包含计算时间，P2P 单独显示）
        """
        import json

        trace = self.to_chrome_trace(time_unit, include_blocks=include_blocks)
        with open(path, "w") as f:
            json.dump(trace, f, indent=2)

    def __repr__(self) -> str:
        return f"TimelineIR(events={len(self.events)})"


# ==============================================================================
# Simulation Result
# ==============================================================================


@dataclass
class MemoryBreakdown:
    """内存分解."""

    weights: int | float = 0
    activations: int | float = 0
    gradients: int | float = 0
    optimizer_states: int | float = 0

    @property
    def total(self) -> int | float:
        return self.weights + self.activations + self.gradients + self.optimizer_states


@dataclass
class TimeBreakdown:
    """时间分解."""

    forward: float = 0
    backward: float = 0
    communication: float = 0
    bubble: float = 0
    recompute: float = 0  # 激活重计算时间 (gradient checkpointing)

    @property
    def compute(self) -> float:
        """计算时间 = forward + backward + recompute."""
        return self.forward + self.backward + self.recompute

    @property
    def total(self) -> float:
        """总时间 = compute + communication + bubble."""
        return (
            self.forward
            + self.backward
            + self.recompute
            + self.communication
            + self.bubble
        )


@dataclass
class BlockMetrics:
    """单层（Block）指标.

    用于存储单个 Transformer 层的内存和时间指标。
    """

    # 单层内存
    weights: int | float = 0
    activations: int | float = 0
    optimizer_states: int | float = 0

    # 单层时间 (per-layer per-microbatch)
    forward_time: float = 0
    backward_time: float = 0
    communication_time: float = 0

    # TP 通信时间 (前向/反向分开)
    comm_fw: float = 0  # 前向阶段的 TP 通信
    comm_bw: float = 0  # 反向阶段的 TP 通信

    @property
    def compute_time(self) -> float:
        """计算时间 = forward + backward."""
        return self.forward_time + self.backward_time

    @property
    def total_time(self) -> float:
        """总时间 = compute + communication."""
        return self.forward_time + self.backward_time + self.communication_time


@dataclass
class SimulationResult:
    """模拟结果 - 编译器最终输出.

    Attributes:
        peak_memory: 峰值内存 (bytes)
        e2e_time: 端到端时间 (seconds)
        memory_breakdown: 内存分解 (per-GPU 总内存)
        time_breakdown: 时间分解 (per-layer per-microbatch)
        total_time_breakdown: 总时间分解 (整个迭代)
        block_metrics: 单层指标 (单个 Transformer 层)
        total_flops: 总计算量
        config: 配置信息
        timeline: TimelineIR 引用 (用于导出 trace)
    """

    peak_memory: int | float = 0
    e2e_time: float = 0
    memory_breakdown: MemoryBreakdown | None = None
    time_breakdown: TimeBreakdown | None = None  # per-layer per-microbatch
    total_time_breakdown: TimeBreakdown | None = None  # 整个迭代的总时间
    block_metrics: BlockMetrics | None = None  # 单层指标
    total_flops: int | float = 0
    config: dict[str, Any] = field(default_factory=dict)
    timeline: TimelineIR | None = None  # TimelineIR 引用，用于导出 Chrome Trace
    estimate: Any = None  # SymbolicEstimate (符号化估算，可选)

    @property
    def mfu(self) -> float:
        """Model FLOPs Utilization.

        MFU = per_GPU_FLOPs / (peak_FLOPs × e2e_time)

        对于 PP 并行，每个 GPU 只处理部分层，
        所以 per_GPU_FLOPs = total_flops / pp
        """
        peak_tflops = self.config.get("peak_tflops", 0)
        pp = self.config.get("pp", 1)
        if peak_tflops == 0 or self.e2e_time == 0:
            return 0
        # per-GPU FLOPs = total_flops / pp
        per_gpu_flops = self.total_flops / pp if pp > 0 else self.total_flops
        return per_gpu_flops / (peak_tflops * 1e12 * self.e2e_time)

    def __repr__(self) -> str:
        return (
            f"SimulationResult(\n"
            f"  peak_memory={self.peak_memory/1e9:.2f} GB,\n"
            f"  e2e_time={self.e2e_time*1e3:.2f} ms,\n"
            f"  mfu={self.mfu:.1%}\n"
            f")"
        )


# ==============================================================================
# Render Mixin - 动态绑定渲染方法
# ==============================================================================


def _bind_render_methods():
    """动态绑定渲染方法到 IR 类型.
    
    这种方式避免了循环导入问题，同时让 IR 类型具有渲染能力。
    """
    from .render import (
        GraphRenderMixin,
        ScheduleRenderMixin,
        TimelineRenderMixin,
    )
    
    # 绑定 GraphIR 的渲染方法
    for name in ('to_terminal', 'to_tree_html', 'to_table', '_repr_html_'):
        if hasattr(GraphRenderMixin, name):
            setattr(GraphIR, name, getattr(GraphRenderMixin, name))
    
    # 绑定 ScheduleIR 的渲染方法
    for name in ('to_terminal', 'to_tree_html', 'to_table', '_repr_html_'):
        if hasattr(ScheduleRenderMixin, name):
            setattr(ScheduleIR, name, getattr(ScheduleRenderMixin, name))
    
    # 绑定 TimelineIR 的渲染方法
    for name in ('to_terminal', 'to_tree_html', 'to_table', '_repr_html_'):
        if hasattr(TimelineRenderMixin, name):
            setattr(TimelineIR, name, getattr(TimelineRenderMixin, name))


# 模块加载时绑定渲染方法
try:
    _bind_render_methods()
except ImportError:
    # 如果 render 模块不可用，忽略
    pass


# ==============================================================================
# Exports
# ==============================================================================

__all__ = [
    # Common
    "Phase",
    "NodeType",
    "TensorRef",
    # Graph IR
    "BlockNode",
    "GraphIR",
    # Schedule IR
    "OpNode",
    "TensorLifetime",
    "ScheduledOp",
    "DeviceSchedule",
    "StageSchedule",
    "ScheduleIR",
    # Timeline IR
    "EventType",
    "StreamType",
    "TimelineEvent",
    "MemorySnapshot",
    "StreamState",
    "TimelineIR",
    # Result
    "MemoryBreakdown",
    "TimeBreakdown",
    "BlockMetrics",
    "SimulationResult",
]

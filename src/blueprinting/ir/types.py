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
from typing import Any, Dict, List, Optional, Union, Iterator, TYPE_CHECKING

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
    - BACKWARD: 反向传播 (包括 agrad 和 wgrad)
    - OPTIMIZER: 优化器更新
    """
    FORWARD = "forward"
    BACKWARD = "backward"
    OPTIMIZER = "optimizer"


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
    children: List["BlockNode"] = field(default_factory=list)
    attrs: Dict[str, Any] = field(default_factory=dict)
    device: Optional[int] = None
    
    @property
    def block_def(self) -> Optional["BlockDef"]:
        """获取关联的 BlockDef 类."""
        from .ops import get_block_def
        return get_block_def(self.block_type)
    
    @property
    def has_params(self) -> bool:
        """是否有可学习参数."""
        block_def = self.block_def
        return block_def.has_params() if block_def else False
    
    @property
    def params(self) -> List[str]:
        """获取参数名称列表."""
        block_def = self.block_def
        return block_def.params if block_def else []
    
    def add_child(self, child: "BlockNode") -> "BlockNode":
        """添加子 Block."""
        self.children.append(child)
        return child
    
    def iter_blocks(self) -> Iterator["BlockNode"]:
        """深度优先遍历所有 Block."""
        yield self
        for child in self.children:
            yield from child.iter_blocks()
    
    def compute_flops(self) -> Optional[Union[int, Expr]]:
        """计算 FLOPs (委托给 BlockDef)."""
        block_def = self.block_def
        if block_def:
            return block_def.compute_flops(self.attrs)
        return None
    
    def compute_param_bytes(self) -> Optional[Union[int, Expr]]:
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
    root: Optional[BlockNode] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    symbols: Dict[str, Symbol] = field(default_factory=dict)
    
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
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    attrs: Dict[str, Any] = field(default_factory=dict)
    
    # Source tracking
    source_block: Optional[str] = None
    
    # Workload (can be computed via OpDef or set directly)
    flops: Optional[Union[int, Expr]] = None
    memory_bytes: Optional[Union[int, Expr]] = None
    comm_bytes: Optional[Union[int, Expr]] = None
    
    @property
    def op_def(self) -> Optional["OpDef"]:
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
    
    def compute_flops(self) -> Optional[Union[int, Expr]]:
        """计算 FLOPs (委托给 OpDef)."""
        if self.flops is not None:
            return self.flops
        op_def = self.op_def
        if op_def:
            return op_def.compute_flops(self.attrs)
        return None
    
    def compute_memory(self) -> Optional[Union[int, Expr]]:
        """计算内存访问 (委托给 OpDef)."""
        if self.memory_bytes is not None:
            return self.memory_bytes
        op_def = self.op_def
        if op_def:
            return op_def.compute_memory(self.attrs)
        return None
    
    def compute_comm(self) -> Optional[Union[int, Expr]]:
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
    start: Union[float, Expr] = 0
    duration: Union[float, Expr] = 0
    event_seq: int = 0
    
    @property
    def end(self) -> Union[float, Expr]:
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
    ops: List[ScheduledOp] = field(default_factory=list)
    
    def add_op(self, op: ScheduledOp) -> None:
        """添加 Op."""
        self.ops.append(op)
    
    @property
    def end_time(self) -> Union[float, Expr]:
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
    devices: Dict[int, DeviceSchedule] = field(default_factory=dict)
    
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
    stages: Dict[int, StageSchedule] = field(default_factory=dict)
    num_devices: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    
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
        return sum(
            len(d.ops)
            for s in self.stages.values()
            for d in s.devices.values()
        )
    
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
    COMPUTE = "compute"
    COMM = "comm"
    MEMORY = "memory"


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
    time: Union[float, Expr]
    event_type: EventType
    resource_id: str
    device: int = 0
    stream: StreamType = StreamType.COMPUTE
    size: Union[int, Expr] = 0
    op_type: str = ""
    phase: Phase = Phase.FORWARD
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other: "TimelineEvent") -> bool:
        """按时间排序."""
        if isinstance(self.time, (int, float)) and isinstance(other.time, (int, float)):
            return self.time < other.time
        return str(self.time) < str(other.time)
    
    def __repr__(self) -> str:
        return f"Event({self.event_type.value}, {self.resource_id}, t={self.time})"


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
    time: Union[float, Expr]
    device: int
    allocated: Union[int, Expr] = 0
    peak: Union[int, Expr] = 0
    tensors: List[str] = field(default_factory=list)


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
    """
    events: List[TimelineEvent] = field(default_factory=list)
    memory_snapshots: List[MemorySnapshot] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_event(self, event: TimelineEvent) -> None:
        """添加事件."""
        self.events.append(event)
    
    def sort_events(self) -> None:
        """按时间排序事件."""
        self.events.sort()
    
    @property
    def end_time(self) -> Union[float, Expr]:
        """时间线结束时间."""
        if not self.events:
            return 0
        return max(e.time for e in self.events)
    
    def to_chrome_trace(self, time_unit: str = "ms") -> Dict[str, Any]:
        """导出为 Chrome Trace 格式.
        
        可以在 chrome://tracing 或 https://ui.perfetto.dev 中打开。
        
        Args:
            time_unit: 时间单位 ("ms" 或 "us")，Chrome Trace 使用微秒
            
        Returns:
            Chrome Trace 格式的字典
        """
        # Chrome Trace 使用微秒
        time_scale = 1000.0 if time_unit == "ms" else 1.0
        
        trace_events = []
        
        # 将 device 映射到 pid，stream 映射到 tid
        # 为了更好的可视化，按 phase 分组
        phase_to_tid = {
            Phase.FORWARD: 1,
            Phase.BACKWARD: 2,
            Phase.OPTIMIZER: 3,
        }
        
        for event in self.events:
            # 计算时间戳（微秒）
            time_val = event.time
            if isinstance(time_val, Expr):
                try:
                    time_val = float(time_val)
                except (TypeError, ValueError):
                    time_val = 0
            ts = time_val * time_scale  # 转换为微秒
            
            # 确定 phase 类型
            ph = None
            if event.event_type == EventType.COMPUTE_START:
                ph = "B"  # Begin
            elif event.event_type == EventType.COMPUTE_END:
                ph = "E"  # End
            elif event.event_type == EventType.COMM_START:
                ph = "B"
            elif event.event_type == EventType.COMM_END:
                ph = "E"
            elif event.event_type == EventType.ALLOC:
                ph = "i"  # Instant event
            elif event.event_type == EventType.FREE:
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
            
            # 确定 tid（按 phase 和 stream 分组）
            base_tid = phase_to_tid.get(event.phase, 0)
            stream_offset = 10 if event.stream == StreamType.COMM else 0
            tid = base_tid + stream_offset
            
            trace_event = {
                "name": event.resource_id,
                "cat": cat,
                "ph": ph,
                "ts": ts,
                "pid": event.device,
                "tid": tid,
            }
            
            # 添加额外信息
            if event.metadata:
                trace_event["args"] = event.metadata.copy()
            else:
                trace_event["args"] = {}
            
            trace_event["args"]["op_type"] = event.op_type
            trace_event["args"]["phase"] = event.phase.value if event.phase else ""
            
            # 对于内存事件，添加大小信息
            if event.event_type in (EventType.ALLOC, EventType.FREE):
                trace_event["args"]["bytes"] = event.size
                trace_event["s"] = "g"  # global scope for instant events
            
            trace_events.append(trace_event)
        
        # 添加元数据
        metadata = []
        
        # 进程名称
        devices = set(e.device for e in self.events)
        for device in devices:
            metadata.append({
                "name": "process_name",
                "ph": "M",
                "pid": device,
                "args": {"name": f"GPU {device}"}
            })
        
        # 线程名称
        thread_names = {
            1: "Forward (Compute)",
            2: "Backward (Compute)",
            3: "Optimizer (Compute)",
            11: "Forward (Comm)",
            12: "Backward (Comm)",
            13: "Optimizer (Comm)",
        }
        for tid, name in thread_names.items():
            for device in devices:
                metadata.append({
                    "name": "thread_name",
                    "ph": "M",
                    "pid": device,
                    "tid": tid,
                    "args": {"name": name}
                })
        
        return {
            "traceEvents": metadata + trace_events,
            "displayTimeUnit": "ms",
            "metadata": self.metadata,
        }
    
    def save_chrome_trace(self, path: str, time_unit: str = "ms") -> None:
        """保存为 Chrome Trace JSON 文件.
        
        Args:
            path: 输出文件路径
            time_unit: 时间单位
        """
        import json
        trace = self.to_chrome_trace(time_unit)
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
    weights: Union[int, float] = 0
    activations: Union[int, float] = 0
    gradients: Union[int, float] = 0
    optimizer_states: Union[int, float] = 0
    
    @property
    def total(self) -> Union[int, float]:
        return self.weights + self.activations + self.gradients + self.optimizer_states


@dataclass
class TimeBreakdown:
    """时间分解."""
    forward: float = 0
    backward: float = 0
    communication: float = 0
    bubble: float = 0
    
    @property
    def compute(self) -> float:
        """计算时间 = forward + backward."""
        return self.forward + self.backward
    
    @property
    def total(self) -> float:
        """总时间 = compute + communication + bubble."""
        return self.forward + self.backward + self.communication + self.bubble


@dataclass
class BlockMetrics:
    """单层（Block）指标.
    
    用于存储单个 Transformer 层的内存和时间指标。
    """
    # 单层内存
    weights: Union[int, float] = 0
    activations: Union[int, float] = 0
    optimizer_states: Union[int, float] = 0
    
    # 单层时间 (per-layer per-microbatch)
    forward_time: float = 0
    backward_time: float = 0
    communication_time: float = 0
    
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
    peak_memory: Union[int, float] = 0
    e2e_time: float = 0
    memory_breakdown: Optional[MemoryBreakdown] = None
    time_breakdown: Optional[TimeBreakdown] = None  # per-layer per-microbatch
    total_time_breakdown: Optional[TimeBreakdown] = None  # 整个迭代的总时间
    block_metrics: Optional[BlockMetrics] = None  # 单层指标
    total_flops: Union[int, float] = 0
    config: Dict[str, Any] = field(default_factory=dict)
    timeline: Optional["TimelineIR"] = None  # TimelineIR 引用，用于导出 Chrome Trace
    
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
# Exports
# ==============================================================================

__all__ = [
    # Common
    "Phase",
    # Graph IR
    "BlockNode",
    "GraphIR",
    # Schedule IR
    "OpNode",
    "ScheduledOp",
    "DeviceSchedule",
    "StageSchedule",
    "ScheduleIR",
    # Timeline IR
    "EventType",
    "StreamType",
    "TimelineEvent",
    "MemorySnapshot",
    "TimelineIR",
    # Result
    "MemoryBreakdown",
    "TimeBreakdown",
    "BlockMetrics",
    "SimulationResult",
]

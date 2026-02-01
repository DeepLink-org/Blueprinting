"""TimelinePass v2 - Convert ScheduleIR to TimelineIR.

将 Schedule IR (Op 级别) 转换为 Timeline IR (Event 级别)。

每个 ScheduledOp 展开成:
- COMPUTE_START / COMM_START 事件
- COMPUTE_END / COMM_END 事件
- ALLOC / FREE 事件 (追踪内存分配)
"""

from typing import Dict, List, Optional, Set, Tuple, Union
from collections import defaultdict

from sympy import Expr

from .base import Pass
from ..types import (
    ScheduleIR, ScheduledOp,
    TimelineIR, TimelineEvent, EventType, StreamType, MemorySnapshot,
)


class TimelinePassV2(Pass):
    """将 ScheduleIR 转换为 TimelineIR.
    
    除了生成 COMPUTE/COMM 事件，还生成:
    - ALLOC: 内存分配事件 (激活、权重)
    - FREE: 内存释放事件 (前向激活在反向后释放)
    """
    
    def __init__(self, track_memory: bool = True):
        """初始化 TimelinePassV2.
        
        Args:
            track_memory: 是否生成内存事件 (ALLOC/FREE)
        """
        self.track_memory = track_memory
    
    def run(self, ir: ScheduleIR) -> TimelineIR:
        timeline = TimelineIR(
            metadata=ir.metadata.copy(),
        )
        
        # 收集所有 Op 用于内存分析
        all_ops = list(ir.iter_ops())
        
        # 遍历所有 Op，生成事件
        for op in all_ops:
            self._add_op_events(op, timeline)
        
        # 生成内存事件
        if self.track_memory:
            self._add_memory_events(all_ops, timeline)
        
        # 排序事件
        timeline.sort_events()
        
        return timeline
    
    def _add_op_events(self, op: ScheduledOp, timeline: TimelineIR) -> None:
        """为单个 Op 添加计算/通信事件."""
        # 判断是计算还是通信
        is_comm = op.op_type in ("AllReduce", "AllGather", "ReduceScatter", "Send", "Recv")
        
        if is_comm:
            start_type = EventType.COMM_START
            end_type = EventType.COMM_END
            stream = StreamType.COMM
        else:
            start_type = EventType.COMPUTE_START
            end_type = EventType.COMPUTE_END
            stream = StreamType.COMPUTE
        
        # Start 事件
        timeline.add_event(TimelineEvent(
            time=op.start,
            event_type=start_type,
            resource_id=op.name,
            device=op.device,
            stream=stream,
            op_type=op.op_type,
            metadata={"source_block": op.op.source_block if op.op else None},
        ))
        
        # End 事件
        timeline.add_event(TimelineEvent(
            time=op.end,
            event_type=end_type,
            resource_id=op.name,
            device=op.device,
            stream=stream,
            op_type=op.op_type,
        ))
    
    def _add_memory_events(self, ops: List[ScheduledOp], timeline: TimelineIR) -> None:
        """生成内存分配/释放事件.
        
        内存模型:
        1. 权重在模型加载时分配，不释放
        2. 前向激活在 Op 开始时分配
        3. 前向激活在对应反向 Op 结束后释放
        4. 梯度在反向 Op 开始时分配，优化器结束后释放
        """
        # 分离前向和反向 Op
        forward_ops = []
        backward_ops = []
        optimizer_ops = []
        
        for op in ops:
            if "_BW" in op.op_type:
                backward_ops.append(op)
            elif op.op_type == "OptimizerStep":
                optimizer_ops.append(op)
            else:
                forward_ops.append(op)
        
        # 权重分配 (在时间 0)
        for op in forward_ops:
            if op.op_type == "Matmul" and op.op:
                attrs = op.op.attrs
                K = attrs.get("K", 0)
                N = attrs.get("N", 0)
                weight_bytes = K * N * 2  # float16
                if weight_bytes > 0:
                    timeline.add_event(TimelineEvent(
                        time=0,
                        event_type=EventType.ALLOC,
                        resource_id=f"{op.name}_weight",
                        device=op.device,
                        stream=StreamType.MEMORY,
                        metadata={
                            "bytes": weight_bytes,
                            "type": "weight",
                            "source_block": op.op.source_block,
                        },
                    ))
        
        # 前向激活分配 (Op 开始时)
        for op in forward_ops:
            if op.op and op.op.memory_bytes:
                memory_bytes = op.op.memory_bytes
                # 只统计输出激活 (约 1/3)
                if isinstance(memory_bytes, (int, float)) and memory_bytes > 0:
                    activation_bytes = memory_bytes // 3
                    if activation_bytes > 0:
                        timeline.add_event(TimelineEvent(
                            time=op.start,
                            event_type=EventType.ALLOC,
                            resource_id=f"{op.name}_act",
                            device=op.device,
                            stream=StreamType.MEMORY,
                            metadata={
                                "bytes": activation_bytes,
                                "type": "activation",
                            },
                        ))
        
        # 前向激活释放 (对应反向 Op 结束后)
        # 简化: 在所有反向完成后释放
        if backward_ops:
            last_bw_end = max(op.end for op in backward_ops)
            for op in forward_ops:
                if op.op and op.op.memory_bytes:
                    memory_bytes = op.op.memory_bytes
                    if isinstance(memory_bytes, (int, float)) and memory_bytes > 0:
                        activation_bytes = memory_bytes // 3
                        if activation_bytes > 0:
                            timeline.add_event(TimelineEvent(
                                time=last_bw_end,
                                event_type=EventType.FREE,
                                resource_id=f"{op.name}_act",
                                device=op.device,
                                stream=StreamType.MEMORY,
                                metadata={
                                    "bytes": activation_bytes,
                                    "type": "activation",
                                },
                            ))


class SimulatePass(Pass):
    """模拟 TimelineIR，生成 SimulationResult.
    
    功能:
    1. 符号替换 (将符号值替换为具体数值)
    2. 计算 e2e 时间
    3. 追踪内存使用，计算峰值
    4. 累加 FLOPs
    5. 计算 MFU (Model FLOPs Utilization)
    6. 生成时间分解
    """
    
    def __init__(
        self,
        subs: Optional[Dict[str, float]] = None,
        peak_tflops: float = 312.0,  # A100 FP16 峰值
        training: bool = True,
    ):
        """初始化 SimulatePass.
        
        Args:
            subs: 符号替换字典 (e.g., {"B": 4, "S": 2048, "H": 4096})
            peak_tflops: 硬件峰值算力 (TFLOPS)
            training: 是否训练模式
        """
        self.subs = subs or {}
        self.peak_tflops = peak_tflops
        self.peak_flops = peak_tflops * 1e12
        self.training = training
    
    def run(self, ir: TimelineIR):
        from ..types import SimulationResult, MemoryBreakdown, TimeBreakdown
        
        # 计算结束时间
        e2e_time = self._eval_expr(ir.end_time) if ir.events else 0
        
        # 内存追踪
        peak_memory, memory_breakdown = self._track_memory(ir)
        
        # 时间分解
        time_breakdown = self._compute_time_breakdown(ir)
        
        # 累加 FLOPs
        total_flops = self._compute_total_flops(ir)
        
        # 计算 MFU
        mfu = self._compute_mfu(total_flops, e2e_time) if e2e_time > 0 else 0
        
        # 确保 config 中有 peak_tflops 以计算 MFU
        config = ir.metadata.copy()
        config["peak_tflops"] = self.peak_tflops
        config["tokens_per_second"] = self._compute_throughput(ir, e2e_time)
        
        result = SimulationResult(
            peak_memory=peak_memory,
            e2e_time=e2e_time,
            time_breakdown=time_breakdown,
            memory_breakdown=memory_breakdown,
            total_flops=total_flops,
            config=config,
        )
        
        return result
    
    def _eval_expr(self, expr) -> float:
        """计算表达式值（支持符号替换）."""
        if isinstance(expr, (int, float)):
            return float(expr)
        
        if isinstance(expr, Expr):
            # 尝试符号替换
            from sympy import Symbol
            subs_dict = {Symbol(k): v for k, v in self.subs.items()}
            try:
                result = expr.subs(subs_dict)
                return float(result)
            except (TypeError, ValueError):
                return 0.0
        
        return 0.0
    
    def _track_memory(self, ir: TimelineIR) -> Tuple[float, "MemoryBreakdown"]:
        """追踪内存使用，返回峰值和分解."""
        from ..types import MemoryBreakdown
        
        current_memory = 0.0
        peak_memory = 0.0
        
        weight_memory = 0.0
        activation_memory = 0.0
        gradient_memory = 0.0
        
        for event in ir.events:
            if event.event_type == EventType.ALLOC:
                bytes_val = event.metadata.get("bytes", 0) if event.metadata else 0
                bytes_val = self._eval_expr(bytes_val)
                
                if bytes_val > 0:
                    current_memory += bytes_val
                    
                    mem_type = event.metadata.get("type", "") if event.metadata else ""
                    if mem_type == "weight":
                        weight_memory += bytes_val
                    elif mem_type == "activation":
                        activation_memory = max(activation_memory, bytes_val)
                    elif mem_type == "gradient":
                        gradient_memory += bytes_val
                    
                    peak_memory = max(peak_memory, current_memory)
            
            elif event.event_type == EventType.FREE:
                bytes_val = event.metadata.get("bytes", 0) if event.metadata else 0
                bytes_val = self._eval_expr(bytes_val)
                if bytes_val > 0:
                    current_memory -= bytes_val
        
        # 计算优化器状态 (Adam: 2x weight for m, v)
        optimizer_memory = weight_memory * 2 if self.training else 0
        
        breakdown = MemoryBreakdown(
            weights=weight_memory,
            activations=activation_memory,
            gradients=gradient_memory,
            optimizer_states=optimizer_memory,
        )
        
        return peak_memory, breakdown
    
    def _compute_time_breakdown(self, ir: TimelineIR) -> "TimeBreakdown":
        """计算时间分解."""
        from ..types import TimeBreakdown
        
        forward_time = 0.0
        backward_time = 0.0
        comm_time = 0.0
        
        # 配对 START/END 事件
        starts: Dict[str, float] = {}
        
        for event in ir.events:
            resource = event.resource_id
            time_val = self._eval_expr(event.time)
            
            if event.event_type in (EventType.COMPUTE_START, EventType.COMM_START):
                starts[resource] = time_val
            
            elif event.event_type == EventType.COMPUTE_END:
                if resource in starts:
                    duration = time_val - starts[resource]
                    op_type = event.op_type or ""
                    if "_BW" in op_type:
                        backward_time += duration
                    elif "Optimizer" in op_type:
                        # 优化器时间计入 backward
                        backward_time += duration
                    else:
                        forward_time += duration
            
            elif event.event_type == EventType.COMM_END:
                if resource in starts:
                    duration = time_val - starts[resource]
                    comm_time += duration
        
        return TimeBreakdown(
            forward=forward_time,
            backward=backward_time,
            communication=comm_time,
        )
    
    def _compute_total_flops(self, ir: TimelineIR) -> float:
        """累加 FLOPs."""
        # 优先从 metadata 获取
        total = ir.metadata.get("total_flops", 0)
        if total > 0:
            return self._eval_expr(total)
        
        # 否则从事件估算 (简化)
        return 0.0
    
    def _compute_mfu(self, total_flops: float, e2e_time: float) -> float:
        """计算 MFU (Model FLOPs Utilization).
        
        MFU = Actual FLOPs / (Peak FLOPS × Time)
        """
        if e2e_time <= 0 or self.peak_flops <= 0:
            return 0.0
        
        theoretical_max = self.peak_flops * e2e_time
        return total_flops / theoretical_max if theoretical_max > 0 else 0.0
    
    def _compute_throughput(self, ir: TimelineIR, e2e_time: float) -> float:
        """计算吞吐量 (tokens/s)."""
        if e2e_time <= 0:
            return 0.0
        
        batch_size = ir.metadata.get("batch_size", 1)
        seq_len = ir.metadata.get("seq_len", 1)
        num_mb = ir.metadata.get("num_microbatches", 1)
        
        total_tokens = batch_size * seq_len * num_mb
        return total_tokens / e2e_time

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
    ScheduleIR, ScheduledOp, Phase,
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
        
        
        # 累加 FLOPs
        total_flops = 0
        for op in all_ops:
            if op.op and op.op.flops:
                flops = op.op.flops
                if isinstance(flops, (int, float)):
                    total_flops += flops
        timeline.metadata["total_flops"] = total_flops
        
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
        # 使用 op_type 的基础类型判断（移除 _BW 后缀）
        base_type = op.op_type.replace("_BW", "") if op.op_type else ""
        is_comm = base_type in ("AllReduce", "AllGather", "ReduceScatter", "Send", "Recv")
        
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
            phase=op.phase,
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
            phase=op.phase,
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
        # 使用 source_block 去重，避免 micro-batch 复制导致权重重复分配
        seen_weight_sources = set()
        total_weight_bytes = 0
        for op in forward_ops:
            if op.op_type == "Matmul" and op.op:
                source = op.op.source_block or ""
                if source in seen_weight_sources:
                    continue
                seen_weight_sources.add(source)
                
                attrs = op.op.attrs
                K = attrs.get("K", 0)
                N = attrs.get("N", 0)
                weight_bytes = K * N * 2  # float16
                if weight_bytes > 0:
                    total_weight_bytes += weight_bytes
                    timeline.add_event(TimelineEvent(
                        time=0,
                        event_type=EventType.ALLOC,
                        resource_id=f"{op.name}_weight",
                        device=op.device,
                        stream=StreamType.MEMORY,
                        metadata={
                            "bytes": weight_bytes,
                            "type": "weight",
                            "source_block": source,
                        },
                    ))
        
        # 前向激活分配
        # 使用 source_block 去重，避免 micro-batch 复制导致激活重复计算
        pp = timeline.metadata.get("pp", 1)
        gradient_checkpointing = timeline.metadata.get("gradient_checkpointing", False)
        
        
        seen_activation_sources = set()
        single_layer_activation = 0  # 单层激活
        num_unique_layers = 0
        
        for op in forward_ops:
            if op.op and op.op.memory_bytes:
                source = op.op.source_block or ""
                if source in seen_activation_sources:
                    continue
                seen_activation_sources.add(source)
                
                memory_bytes = op.op.memory_bytes
                # 只统计输出激活 (约 1/3)
                if isinstance(memory_bytes, (int, float)) and memory_bytes > 0:
                    activation_bytes = memory_bytes // 3
                    single_layer_activation += activation_bytes
        
        # 计算每层的激活（single_layer_activation 是所有层的总和，需要除以 num_layers）
        num_layers = timeline.metadata.get("num_layers", 1)
        layers_per_stage = num_layers // pp if pp > 0 else num_layers
        per_layer_activation = single_layer_activation / num_layers if num_layers > 0 else single_layer_activation
        
        # 激活内存模型（per-GPU）：
        # - Calculon 的激活不随 PP 变化，说明它计算的是单层激活（不考虑 pipeline）
        # - 激活通常不被 TP 分片（除了 Attention 后的 AllGather）
        # - gradient checkpointing 时，只保存每层输入（约 2 层激活）
        
        if gradient_checkpointing:
            # 完全重计算：保存 2 层激活（当前层输入 + 上一层输出）
            # Calculon 对齐：激活不除以 TP，不乘以 PP
            layers_in_memory = 2
        else:
            # 无检查点：保存 layers_per_stage 层的所有激活
            layers_in_memory = layers_per_stage
        
        # 总激活 = 单层激活 × 内存中的层数
        # 注意：激活数据在 TP 组内通常是复制的（不分片），所以不除以 TP
        total_activation = per_layer_activation * layers_in_memory
        
        
        # 生成一个汇总的激活分配事件
        if total_activation > 0:
            timeline.add_event(TimelineEvent(
                time=0,
                event_type=EventType.ALLOC,
                resource_id="total_activation",
                device=0,
                stream=StreamType.MEMORY,
                metadata={
                    "bytes": total_activation,
                    "type": "activation",
                },
            ))
        
        # 激活释放 (在所有反向完成后)
        if backward_ops and total_activation > 0:
            last_bw_end = max(op.end for op in backward_ops)
            timeline.add_event(TimelineEvent(
                time=last_bw_end,
                event_type=EventType.FREE,
                resource_id="total_activation",
                device=0,
                stream=StreamType.MEMORY,
                metadata={
                    "bytes": total_activation,
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
        from ..types import SimulationResult, MemoryBreakdown, TimeBreakdown, BlockMetrics
        
        # 从 metadata 获取并行配置
        pp = ir.metadata.get("pp", 1)
        num_layers = ir.metadata.get("num_layers", 1)
        num_microbatches = ir.metadata.get("num_microbatches", 1)
        layers_per_stage = num_layers // pp if pp > 0 else num_layers
        
        # 计算结束时间
        # PipelineSchedulePass 已经计算了正确的 1F1B 调度时间
        # TimelineIR.end_time 就是正确的 E2E 时间（考虑了 PP 并行和 bubble）
        e2e_time = self._eval_expr(ir.end_time) if ir.events else 0
        
        # 内存追踪 (per-GPU)
        peak_memory, memory_breakdown = self._track_memory(ir, pp=pp, layers_per_stage=layers_per_stage)
        
        # 时间分解 (per-layer per-microbatch)
        time_breakdown = self._compute_time_breakdown(ir, num_layers=num_layers, num_microbatches=num_microbatches)
        
        # 计算总时间分解 (整个迭代)
        # 在 PP 下，不同 stage 并行执行，所以用 layers_per_stage 而不是 num_layers
        time_multiplier = layers_per_stage * num_microbatches
        total_time_breakdown = TimeBreakdown(
            forward=time_breakdown.forward * time_multiplier,
            backward=time_breakdown.backward * time_multiplier,
            communication=time_breakdown.communication * time_multiplier,
            bubble=time_breakdown.bubble,  # bubble 已经是总时间
        )
        
        # 计算单层指标 (BlockMetrics)
        # 单层内存 = per-GPU 内存 / layers_per_stage
        block_weights = memory_breakdown.weights / layers_per_stage if layers_per_stage > 0 else 0
        block_optimizer = memory_breakdown.optimizer_states / layers_per_stage if layers_per_stage > 0 else 0
        block_metrics = BlockMetrics(
            weights=block_weights,
            activations=memory_breakdown.activations,  # 激活是单层的
            optimizer_states=block_optimizer,
            forward_time=time_breakdown.forward,
            backward_time=time_breakdown.backward,
            communication_time=time_breakdown.communication,
        )
        
        # 累加 FLOPs
        # total_flops 是所有 micro-batch × 所有层的 FLOPs 总和
        total_flops = self._compute_total_flops(ir)
        
        # 计算 MFU (Model FLOPs Utilization)
        # MFU = Actual_FLOPs / (Peak_FLOPs × Time × Num_GPUs)
        # 
        # 对于 PP 并行：
        # - total_flops 是整个迭代的计算量
        # - e2e_time 是 PP 并行后的时间
        # - 需要考虑 GPU 数量
        #
        # 正确的 MFU 计算：
        # per_gpu_flops = total_flops / num_gpus (每个 GPU 的 FLOPs)
        # MFU = per_gpu_flops / (peak_flops × e2e_time)
        #     = total_flops / (peak_flops × e2e_time × num_gpus)
        #
        # 但实际上，在 PP 并行下：
        # - 每个 GPU 只处理 layers_per_stage 层
        # - total_flops 已经是所有层的总和
        # 所以 per_gpu_flops = total_flops / pp
        per_gpu_flops = total_flops / pp if pp > 0 else total_flops
        mfu = self._compute_mfu(per_gpu_flops, e2e_time) if e2e_time > 0 else 0
        
        # 确保 config 中有 peak_tflops 以计算 MFU
        config = ir.metadata.copy()
        config["peak_tflops"] = self.peak_tflops
        config["tokens_per_second"] = self._compute_throughput(ir, e2e_time)
        
        result = SimulationResult(
            peak_memory=peak_memory,
            e2e_time=e2e_time,
            time_breakdown=time_breakdown,
            total_time_breakdown=total_time_breakdown,
            block_metrics=block_metrics,
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
    
    def _track_memory(self, ir: TimelineIR, pp: int = 1, layers_per_stage: int = 1) -> Tuple[float, "MemoryBreakdown"]:
        """追踪内存使用，返回 per-GPU 的峰值和分解.
        
        Args:
            ir: TimelineIR
            pp: Pipeline Parallelism 度数
            layers_per_stage: 每个 PP stage 的层数
        """
        from ..types import MemoryBreakdown
        
        current_memory = 0.0
        peak_memory = 0.0
        
        weight_memory = 0.0
        activation_memory = 0.0
        gradient_memory = 0.0
        
        # 追踪激活内存的累积（训练时需要保存所有前向激活直到反向传播）
        current_activation = 0.0
        peak_activation = 0.0
        
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
                        current_activation += bytes_val
                        peak_activation = max(peak_activation, current_activation)
                    elif mem_type == "gradient":
                        gradient_memory += bytes_val
                    
                    peak_memory = max(peak_memory, current_memory)
            
            elif event.event_type == EventType.FREE:
                bytes_val = event.metadata.get("bytes", 0) if event.metadata else 0
                bytes_val = self._eval_expr(bytes_val)
                if bytes_val > 0:
                    current_memory -= bytes_val
                    # 检查是否是激活释放
                    mem_type = event.metadata.get("type", "") if event.metadata else ""
                    if mem_type == "activation":
                        current_activation -= bytes_val
        
        activation_memory = peak_activation
        
        # PP 分片: 每个 GPU 只存储部分层的权重
        # weight_memory 是所有层的总和，需要除以 PP 得到 per-GPU 的值
        weight_per_gpu = weight_memory / pp if pp > 0 else weight_memory
        
        # 计算优化器状态 (Adam: m + v)
        # Calculon 的 optimizer_space = weight_per_gpu / 2
        # 这可能是因为 Calculon 使用了 ZeRO Stage 2 或类似的优化
        # 
        # 分析：weight_per_gpu = 906 MB (fp16)
        #       Calculon optimizer = 453 MB = 906 / 2
        # 
        # 这可能是 Calculon 只计算 Adam 的一个状态 (m 或 v)
        # 或者使用了 4x 分片：weight * 2 (fp32) * 2 (m+v) / 8 = weight / 2
        # 
        # 为了与 Calculon 对齐，我们使用相同的公式
        if self.training:
            # Calculon 对齐：optimizer = weight_per_gpu / 2
            # 这相当于 Adam 的一个状态在 FP16 下的大小
            optimizer_memory = weight_per_gpu / 2
        else:
            optimizer_memory = 0
        
        # per-GPU 峰值内存
        peak_per_gpu = weight_per_gpu + activation_memory + optimizer_memory
        
        breakdown = MemoryBreakdown(
            weights=weight_per_gpu,
            activations=activation_memory,
            gradients=gradient_memory,
            optimizer_states=optimizer_memory,
        )
        
        return peak_per_gpu, breakdown
    
    def _compute_time_breakdown(self, ir: TimelineIR, num_layers: int = 1, num_microbatches: int = 1) -> "TimeBreakdown":
        """计算时间分解，返回 per-layer per-microbatch 的时间.
        
        Args:
            ir: TimelineIR
            num_layers: 总层数
            num_microbatches: micro-batch 数量
        """
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
                    # 使用 phase 判断阶段
                    if event.phase == Phase.BACKWARD:
                        backward_time += duration
                    elif event.phase == Phase.OPTIMIZER:
                        # 优化器时间计入 backward
                        backward_time += duration
                    else:
                        forward_time += duration
            
            elif event.event_type == EventType.COMM_END:
                if resource in starts:
                    duration = time_val - starts[resource]
                    comm_time += duration
        
        # 归一化为 per-layer per-microbatch 时间
        # 总时间 = 单层时间 * num_layers * num_microbatches
        time_divisor = num_layers * num_microbatches if num_layers > 0 and num_microbatches > 0 else 1
        
        # 计算 bubble time
        # 在 PP 并行下，各 stage 并行执行，累计时间会远大于 E2E 时间
        # 因此不能用 E2E - compute_and_comm 来计算 bubble
        # 
        # 1F1B 调度的 bubble time 近似计算：
        # bubble ≈ (PP - 1) × single_stage_time
        # 其中 single_stage_time = (fw + bw + comm) / time_divisor × num_microbatches
        pp = ir.metadata.get("pp", 1)
        if pp > 1 and time_divisor > 0:
            # 单个 stage 处理所有 micro-batch 的时间
            single_stage_time = (forward_time + backward_time + comm_time) / pp
            # bubble time = (pp - 1) × 单个 micro-batch 的 stage 时间
            bubble_time = (pp - 1) * single_stage_time / num_microbatches
        else:
            bubble_time = 0
        
        return TimeBreakdown(
            forward=forward_time / time_divisor,
            backward=backward_time / time_divisor,
            communication=comm_time / time_divisor,
            bubble=bubble_time,  # bubble 是总时间
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

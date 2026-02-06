"""TimelinePass v2 - Convert ScheduleIR to TimelineIR.

将 Schedule IR (Op 级别) 转换为 Timeline IR (Event 级别)。

每个 ScheduledOp 展开成:
- COMPUTE_START / COMM_START 事件
- COMPUTE_END / COMM_END 事件
- ALLOC / FREE 事件 (追踪内存分配)
"""

from __future__ import annotations

from sympy import Expr

from ..result import MemoryBreakdown, TimeBreakdown
from ..types import EventType, Phase, ScheduledOp, ScheduleIR, StreamType, TimelineEvent, TimelineIR
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
        """生成内存分配/释放事件.

        真实训练语义的内存模型:
        1. 权重：在模型加载时分配，训练期间不释放
        2. 前向激活：Op 完成时分配，对应反向 Op 完成后释放
        3. 梯度检查点：只保留 checkpoint 激活，层内中间激活立即复用
        4. 梯度：反向 Op 产生，优化器更新后释放

        不使用 Calculon 的"峰值公式"，而是通过真实的 ALLOC/FREE 事件追踪。
        """
        import re

        from ..types import Phase

        # 按 phase 分离 Op（使用 Phase 而不是 op_type 后缀）
        forward_ops = []
        backward_ops = []  # 包含 recompute (_RE) 和 backward (_BW)
        optimizer_ops = []

        for op in ops:
            if op.phase == Phase.FORWARD:
                forward_ops.append(op)
            elif op.phase == Phase.BACKWARD:
                backward_ops.append(op)
            elif op.phase == Phase.OPTIMIZER or op.op_type == "OptimizerStep":
                optimizer_ops.append(op)
            elif "_BW" in op.op_type or "_RE" in op.op_type:
                backward_ops.append(op)
            else:
                forward_ops.append(op)

        # 1. 权重分配 (在时间 0，训练期间不释放)
        # 使用 source_block 去重，避免 micro-batch 复制导致权重重复分配
        seen_weight_sources = set()
        dtype_bytes = timeline.metadata.get("dtype_bytes", 2)  # fp16 default

        for op in forward_ops:
            if op.op_type == "Matmul" and op.op:
                source = op.op.source_block or ""
                if source in seen_weight_sources:
                    continue
                seen_weight_sources.add(source)

                attrs = op.op.attrs
                K = attrs.get("K", 0)
                N = attrs.get("N", 0)
                weight_bytes = K * N * dtype_bytes
                if weight_bytes > 0:
                    timeline.add_event(
                        TimelineEvent(
                            time=0,
                            event_type=EventType.ALLOC,
                            resource_id=f"{source}_weight",
                            device=op.device,
                            stream=StreamType.MEMORY,
                            metadata={
                                "bytes": weight_bytes,
                                "type": "weight",
                                "source_block": source,
                            },
                        )
                    )

        # 2. 激活内存 - 峰值估计模型
        # 使用峰值估计而非完整生命周期追踪，从 Op 结构推导单层激活大小。
        pp = timeline.metadata.get("pp", 1)
        gradient_checkpointing = timeline.metadata.get("gradient_checkpointing", False)
        num_layers = timeline.metadata.get("num_layers", 1)
        layers_per_stage = num_layers // pp if pp > 0 else num_layers

        # 提取 layer_idx 的辅助函数
        def extract_layer_idx(source: str | None) -> int | None:
            if not source:
                return None
            patterns = [
                r"TransformerLayer\(layer(\d+)\)",
                r"layer[_\.]?(\d+)",
                r"layers[_\.](\d+)",
                r"\.h\.(\d+)\.",
                r"block[_\.](\d+)",
            ]
            for pattern in patterns:
                match = re.search(pattern, source)
                if match:
                    return int(match.group(1))
            return None

        # 计算 Op 的输出激活大小
        # Calculon 的 working-set 语义：只计算输出激活（layer.get_activation()）
        # 不包含输入激活，因为 Calculon 假设 full recompute 时可以重新计算输入
        def get_op_activation_bytes(op: ScheduledOp) -> int:
            if not op.op or not op.op.attrs:
                return 0
            attrs = op.op.attrs

            if op.op_type == "Matmul":
                M = attrs.get("M", 0)
                N = attrs.get("N", 0)
                # 只计算输出激活 (M × N)，与 Calculon 对齐
                return M * N * dtype_bytes
            elif op.op_type in ("RMSNorm", "LayerNorm"):
                normalized_shape = attrs.get("normalized_shape", 0)
                batch_seq = timeline.metadata.get(
                    "batch_size", 1
                ) * timeline.metadata.get("seq_len", 2048)
                # 输出激活大小
                return batch_seq * normalized_shape * dtype_bytes
            elif op.op_type == "Softmax" or op.op_type in ("SiLU", "GELU", "ReLU"):
                num_elements = attrs.get("num_elements", 0)
                return num_elements * dtype_bytes
            return 0

        # 按 layer_idx 分组 forward ops（只取第一个 micro-batch 计算单层激活）
        layer_ops: dict[int, list[ScheduledOp]] = {}
        for op in forward_ops:
            source = op.op.source_block if op.op else ""
            layer_idx = extract_layer_idx(source)
            if layer_idx is None:
                layer_idx = 0
            # 只收集第一个 micro-batch 的 Op（用于计算单层激活大小）
            mb = op.op.attrs.get("micro_batch", 0) if op.op else 0
            if mb == 0:
                if layer_idx not in layer_ops:
                    layer_ops[layer_idx] = []
                layer_ops[layer_idx].append(op)

        # 计算单层激活大小（从第一层的所有 Op 推导）
        # 不再按 source_block 去重，每个 Op 的输出激活都要计算
        per_layer_activation = 0
        if layer_ops:
            min_layer_idx = min(layer_ops.keys())
            first_layer_ops = layer_ops.get(min_layer_idx, [])
            for op in first_layer_ops:
                per_layer_activation += get_op_activation_bytes(op)

        # 计算峰值激活
        if gradient_checkpointing:
            # Full checkpoint：只需要单层工作激活
            peak_activation = per_layer_activation
        else:
            # 无 checkpoint：需要保存所有层的激活
            peak_activation = per_layer_activation * layers_per_stage

        # 生成汇总的激活内存事件
        if peak_activation > 0:
            timeline.add_event(
                TimelineEvent(
                    time=0,
                    event_type=EventType.ALLOC,
                    resource_id="peak_activation",
                    device=0,
                    stream=StreamType.MEMORY,
                    metadata={
                        "bytes": peak_activation,
                        "type": "activation",
                        "model": "peak_estimate",
                        "per_layer": per_layer_activation,
                        "layers_per_stage": layers_per_stage,
                        "gradient_checkpointing": gradient_checkpointing,
                    },
                )
            )

            # 在反向结束后释放
            if backward_ops:
                last_bw_end = max(op.end for op in backward_ops)
                timeline.add_event(
                    TimelineEvent(
                        time=last_bw_end,
                        event_type=EventType.FREE,
                        resource_id="peak_activation",
                        device=0,
                        stream=StreamType.MEMORY,
                        metadata={"bytes": peak_activation, "type": "activation"},
                    )
                )


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
        subs: dict[str, float] | None = None,
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
        """观测 TimelineIR，统计各项指标.

        SimulatePass 是观测者，只对 TimelineIR 进行统计，不做计算：
        - 从事件统计总时间（forward, backward, communication）
        - 从事件追踪峰值内存
        - 从 metadata 读取配置，计算派生指标
        """
        from ..result import BlockMetrics, SimulationResult, TimeBreakdown

        ir.metadata.get("pp", 1)
        num_microbatches = ir.metadata.get("num_microbatches", 1)
        layers_per_stage = ir.metadata.get("layers_per_stage", 1)
        total_flops = ir.metadata.get("total_flops", 0)

        peak_memory, memory_breakdown = self._observe_memory(ir)

        total_time_breakdown, total_comm_fw, total_comm_bw, iteration_time = (
            self._observe_time(ir)
        )

        e2e_time = iteration_time

        # per-layer per-microbatch 时间
        time_divisor = (
            layers_per_stage * num_microbatches
            if layers_per_stage > 0 and num_microbatches > 0
            else 1
        )
        time_breakdown = TimeBreakdown(
            forward=total_time_breakdown.forward / time_divisor,
            backward=total_time_breakdown.backward / time_divisor,
            communication=total_time_breakdown.communication / time_divisor,
            bubble=total_time_breakdown.bubble,  # bubble 是总时间
        )

        # per-layer per-microbatch 通信时间分解
        comm_fw_per_layer = total_comm_fw / time_divisor if time_divisor > 0 else 0
        comm_bw_per_layer = total_comm_bw / time_divisor if time_divisor > 0 else 0

        # 单层指标（与 Calculon block_*_space 对齐：均为 per-layer per-GPU）
        block_metrics = BlockMetrics(
            weights=(
                memory_breakdown.weights / layers_per_stage
                if layers_per_stage > 0
                else 0
            ),
            activations=(
                memory_breakdown.activations / layers_per_stage
                if layers_per_stage > 0
                else memory_breakdown.activations
            ),
            optimizer_states=(
                memory_breakdown.optimizer_states / layers_per_stage
                if layers_per_stage > 0
                else 0
            ),
            forward_time=time_breakdown.forward,
            backward_time=time_breakdown.backward,
            communication_time=time_breakdown.communication,
            comm_fw=comm_fw_per_layer,
            comm_bw=comm_bw_per_layer,
        )

        config = ir.metadata.copy()
        config["peak_tflops"] = self.peak_tflops
        config["tokens_per_second"] = self._compute_throughput(ir, e2e_time)

        # 从 metadata 提取 SymbolicEstimate（如果存在）
        estimate = ir.metadata.get("symbolic_estimate", None)

        return SimulationResult(
            peak_memory=peak_memory,
            e2e_time=e2e_time,
            time_breakdown=time_breakdown,
            total_time_breakdown=total_time_breakdown,
            block_metrics=block_metrics,
            memory_breakdown=memory_breakdown,
            total_flops=total_flops,
            config=config,
            timeline=ir,  # 保存 TimelineIR 引用
            estimate=estimate,
        )

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

    def _observe_memory(self, ir: TimelineIR) -> tuple[float, MemoryBreakdown]:
        """观测内存使用（遍历事件统计）.

        这是观测者模式：只统计，不计算。
        返回 per-GPU 的峰值内存和内存分解。
        """
        pp = ir.metadata.get("pp", 1)

        current_memory = 0.0
        peak_memory = 0.0

        weight_memory = 0.0
        activation_memory = 0.0
        gradient_memory = 0.0

        # 追踪激活内存的累积（训练时需要保存所有前向激活直到反向传播）
        current_activation = 0.0
        peak_activation = 0.0

        # 按时间排序事件（FREE 在同一时间优先于 ALLOC）
        # 这样才能正确计算峰值内存（先释放后分配，允许内存复用）
        def event_sort_key(e):
            # 同一时间内，FREE 在 ALLOC 之前（先释放后分配）
            # 使用 (time, type_order) 排序
            type_order = 1 if e.event_type == EventType.ALLOC else 0
            return (self._eval_expr(e.time), type_order)

        sorted_events = sorted(ir.events, key=event_sort_key)

        for event in sorted_events:
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

        # 计算优化器状态内存
        # 从 metadata 获取优化器配置
        optimizer_config_dict = ir.metadata.get("optimizer_config", {})

        if self.training and weight_per_gpu > 0:
            # 导入 OptimizerConfig
            from .optimizer import OptimizerConfig

            if optimizer_config_dict:
                opt_config = OptimizerConfig.from_dict(optimizer_config_dict)
            else:
                # 默认配置：Adam with master weights, no ZeRO
                opt_config = OptimizerConfig(
                    optimizer_type="adam",
                    master_weights=True,
                    zero_stage=0,
                    dp=ir.metadata.get("dp", 1),
                )

            # 计算每个参数的优化器状态大小 (bytes)
            # 注意：weight_per_gpu 是 fp16 权重的大小
            # 参数数量 = weight_per_gpu / dtype_bytes
            dtype_bytes = opt_config.dtype_bytes or 2
            num_params = weight_per_gpu / dtype_bytes

            # 优化器状态内存 = 参数数量 × 每参数优化器状态字节
            optimizer_bytes_per_param = opt_config.get_optimizer_memory_per_param()
            optimizer_memory = num_params * optimizer_bytes_per_param

            # 梯度内存（使用 Calculon 的内存优化策略）：
            # - 只保留 1 份完整梯度 (FP32) 用于当前层计算
            # - (blocks_per_proc - 1) 份分片梯度 (FP16/dp) 用于累积和通信
            # 这比简单的 layers_per_stage × per_layer_grad 更节省内存
            layers_per_stage = ir.metadata.get("layers_per_stage", 1)
            dp = opt_config.dp or 1
            grad_dtype_bytes = (
                opt_config.grad_accumulation_dtype_bytes or 4
            )  # FP32 for accumulation

            # 单层参数数量
            single_layer_params = (
                num_params / layers_per_stage if layers_per_stage > 0 else num_params
            )

            # 单层梯度（未分片，FP32）- 用于当前层的梯度计算
            block_weight_grad_no_sharding = single_layer_params * grad_dtype_bytes

            # 单层梯度（分片，FP16/dp）- Calculon 使用 FP16 用于通信优化
            # 与 Calculon 一致：sharded gradients 使用 training dtype (FP16)，然后除以 dp
            block_weight_grad_sharded = (
                single_layer_params * dtype_bytes / dp
                if dp > 1
                else single_layer_params * dtype_bytes
            )

            # 总梯度内存 = 1份完整(FP32) + (layers_per_stage-1)份分片(FP16/dp)
            if layers_per_stage <= 1:
                gradient_memory = block_weight_grad_no_sharding
            else:
                gradient_memory = (
                    block_weight_grad_no_sharding
                    + block_weight_grad_sharded * (layers_per_stage - 1)
                )
        else:
            optimizer_memory = 0
            gradient_memory = 0

        # per-GPU 峰值内存
        peak_per_gpu = (
            weight_per_gpu + activation_memory + gradient_memory + optimizer_memory
        )

        breakdown = MemoryBreakdown(
            weights=weight_per_gpu,
            activations=activation_memory,
            gradients=gradient_memory,
            optimizer_states=optimizer_memory,
        )

        return peak_per_gpu, breakdown

    def _observe_time(
        self, ir: TimelineIR
    ) -> tuple[TimeBreakdown, float, float, float]:
        """观测时间，返回总时间、通信分解和迭代时间（遍历事件统计）.

        这是观测者模式：只统计，不计算。
        返回的是单个 PP stage 执行的实际时间（考虑 PP 并行）。

        注意：累计时间是所有 stage 的总和，但各 stage 并行执行，
        所以实际时间 = 累计时间 / pp

        Returns:
            TimeBreakdown: 时间分解
            float: 前向阶段的通信时间 (comm_fw)
            float: 反向阶段的通信时间 (comm_bw)
        """
        # 通过 Phase 分类累加时间
        forward_cumulative = 0.0
        backward_cumulative = 0.0  # 包含 recompute + agrad + wgrad
        comm_cumulative = 0.0
        comm_fw_cumulative = 0.0
        comm_bw_cumulative = 0.0

        # 记录迭代的开始和结束时间
        iteration_start = float("inf")
        iteration_end = 0.0

        # 配对 START/END 事件
        starts: dict[str, float] = {}
        start_phases: dict[str, Phase] = {}
        start_op_types: dict[str, str] = {}

        for event in ir.events:
            resource = event.resource_id
            time_val = self._eval_expr(event.time)

            if event.event_type in (EventType.COMPUTE_START, EventType.COMM_START):
                starts[resource] = time_val
                start_phases[resource] = event.phase
                start_op_types[resource] = event.op_type or ""
                iteration_start = min(iteration_start, time_val)

            elif event.event_type == EventType.COMPUTE_END:
                iteration_end = max(iteration_end, time_val)
                if resource in starts:
                    duration = time_val - starts[resource]
                    phase = start_phases.get(resource, event.phase)
                    op_type = start_op_types.get(resource, event.op_type or "")

                    # 通过 Phase 分类，但 recompute（_RE 后缀）不计入 backward
                    # 与 Calculon 对齐：bw_time = agrad + wgrad（不含 recompute）
                    if phase == Phase.FORWARD:
                        forward_cumulative += duration
                    elif phase == Phase.BACKWARD:
                        # recompute Op 不计入 backward，单独统计
                        if "_RE" not in op_type:
                            backward_cumulative += duration
                    elif phase == Phase.OPTIMIZER:
                        backward_cumulative += duration  # optimizer 计入 backward

            elif event.event_type == EventType.COMM_END:
                iteration_end = max(iteration_end, time_val)
                if resource in starts:
                    duration = time_val - starts[resource]
                    comm_cumulative += duration
                    phase = start_phases.get(resource, event.phase)
                    if phase == Phase.BACKWARD:
                        comm_bw_cumulative += duration
                    else:
                        comm_fw_cumulative += duration

        # PP 并行：各 stage 并行执行
        pp = ir.metadata.get("pp", 1)
        forward_time = forward_cumulative / pp if pp > 0 else forward_cumulative
        backward_time = backward_cumulative / pp if pp > 0 else backward_cumulative
        comm_time = comm_cumulative / pp if pp > 0 else comm_cumulative
        comm_fw = comm_fw_cumulative / pp if pp > 0 else comm_fw_cumulative
        comm_bw = comm_bw_cumulative / pp if pp > 0 else comm_bw_cumulative

        # 单独统计 recompute 时间（通过 _RE 后缀）
        recompute_time = self._compute_recompute_time(ir, pp)

        # 计算 bubble time
        # 每个 stage 的实际执行时间 = FW + recompute + BW + comm
        num_microbatches = ir.metadata.get("num_microbatches", 1)
        if pp > 1 and num_microbatches > 0:
            single_stage_time = (
                forward_time + recompute_time + backward_time + comm_time
            )
            bubble_time = (pp - 1) * single_stage_time / num_microbatches
        else:
            bubble_time = 0

        # 迭代时间 = 最后一个 op 结束时间 - 第一个 op 开始时间
        iteration_time = (
            iteration_end - iteration_start if iteration_start < float("inf") else 0
        )

        return (
            TimeBreakdown(
                forward=forward_time,
                backward=backward_time,
                communication=comm_time,
                bubble=bubble_time,
                recompute=recompute_time,
            ),
            comm_fw,
            comm_bw,
            iteration_time,
        )

    def _compute_recompute_time(self, ir: TimelineIR, pp: int) -> float:
        """单独统计 recompute 时间（通过 _RE 后缀识别）."""
        recompute_cumulative = 0.0
        starts: dict[str, float] = {}
        start_op_types: dict[str, str] = {}

        for event in ir.events:
            resource = event.resource_id
            time_val = self._eval_expr(event.time)

            if event.event_type == EventType.COMPUTE_START:
                starts[resource] = time_val
                start_op_types[resource] = event.op_type or ""

            elif event.event_type == EventType.COMPUTE_END:
                if resource in starts:
                    op_type = start_op_types.get(resource, event.op_type or "")
                    if "_RE" in op_type:
                        duration = time_val - starts[resource]
                        recompute_cumulative += duration

        return recompute_cumulative / pp if pp > 0 else recompute_cumulative

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

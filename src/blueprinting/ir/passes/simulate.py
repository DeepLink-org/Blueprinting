"""SimulatePass - Observe TimelineIR and produce SimulationResult.

SimulatePass 是纯观测者：遍历 TimelineIR 的事件流，统计各项指标，
不做任何训练/推理特定的内存计算。所有内存语义由上游 Schedule Pass
通过 memory_pools → ALLOC/FREE 事件提供。

功能:
1. 符号替换 (将符号值替换为具体数值)
2. 计算 e2e 时间
3. 追踪内存使用，计算峰值
4. 累加 FLOPs
5. 计算 MFU (Model FLOPs Utilization)
6. 生成时间分解

参数管理:
- 使用 @hp.param("system") 从 hyperparameter scope 自动注入硬件参数
"""

from __future__ import annotations

import hyperparameter as hp
from sympy import Expr

from ..result import MemoryBreakdown, TimeBreakdown
from ..types import EventType, Phase, TimelineIR
from .base import Pass


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

    @hp.param("system")
    def __init__(
        self,
        subs: dict[str, float] | None = None,
        peak_tflops: float = 312.0,  # A100 FP16 峰值
    ):
        """初始化 SimulatePass.

        Args:
            subs: 符号替换字典 (e.g., {"B": 4, "S": 2048, "H": 4096})
            peak_tflops: 硬件峰值算力 (TFLOPS)
        """
        self.subs = subs or {}
        self.peak_tflops = peak_tflops
        self.peak_flops = peak_tflops * 1e12

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
        """纯观测：遍历 ALLOC/FREE 事件统计峰值内存和分类 breakdown.

        不做任何训练/推理特定的内存计算。
        所有内存池（weight, activation, gradient, optimizer, kv_cache…）
        都由上游 Schedule Pass 通过 memory_pools → ALLOC/FREE 事件提供。
        """
        current_memory = 0.0
        peak_memory = 0.0

        # 按 mem_type 分类统计（取每类的峰值）
        type_current: dict[str, float] = {}
        type_peak: dict[str, float] = {}

        # 按时间排序（同一时刻 FREE 先于 ALLOC，允许内存复用）
        def event_sort_key(e):
            type_order = 1 if e.event_type == EventType.ALLOC else 0
            return (self._eval_expr(e.time), type_order)

        sorted_events = sorted(ir.events, key=event_sort_key)

        for event in sorted_events:
            if event.event_type == EventType.ALLOC:
                bytes_val = event.metadata.get("bytes", 0) if event.metadata else 0
                bytes_val = self._eval_expr(bytes_val)
                if bytes_val > 0:
                    current_memory += bytes_val
                    peak_memory = max(peak_memory, current_memory)

                    mem_type = event.metadata.get("type", "other") if event.metadata else "other"
                    type_current[mem_type] = type_current.get(mem_type, 0) + bytes_val
                    type_peak[mem_type] = max(type_peak.get(mem_type, 0), type_current[mem_type])

            elif event.event_type == EventType.FREE:
                bytes_val = event.metadata.get("bytes", 0) if event.metadata else 0
                bytes_val = self._eval_expr(bytes_val)
                if bytes_val > 0:
                    current_memory -= bytes_val
                    mem_type = event.metadata.get("type", "other") if event.metadata else "other"
                    type_current[mem_type] = type_current.get(mem_type, 0) - bytes_val

        breakdown = MemoryBreakdown(pools=type_peak)
        return peak_memory, breakdown

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

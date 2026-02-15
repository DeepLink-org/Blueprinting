"""SymbolicEstimatePass - 从 ScheduleIR 生成符号化聚合估算.

这是一个**透明 Pass** (ScheduleIR -> ScheduleIR)：
- 输入 ScheduleIR，原样输出 ScheduleIR（不修改）
- 副作用：生成 SymbolicEstimate，存入 ir.metadata["symbolic_estimate"]
- 后续 TimelinePass/SimulatePass 自动携带 metadata
- SimulatePass 在构建 SimulationResult 时提取 estimate

聚合规则:
- 遍历所有 Op，按 (phase, op_type) 聚合 duration
- 保持符号形式（用 SymPy Expr 的 + 运算累加）
- PP 并行：各项除以 pp
- Bubble: (pp-1) * stage_time / num_microbatches
- 内存：从 Op attrs 推导权重/激活，从权重推导梯度/优化器态
"""

from __future__ import annotations

from typing import Any

import hyperparameter as hp

from ..estimate import SymbolicEstimate
from ..types import Phase, ScheduleIR
from .base import Pass

# 通信 Op 类型
_COMM_OPS = frozenset(
    {
        "AllReduce",
        "AllGather",
        "ReduceScatter",
        "Send",
        "Recv",
        # 反向版本
        "AllReduce_BW",
        "AllGather_BW",
        "ReduceScatter_BW",
        "Send_BW",
        "Recv_BW",
    }
)


def _is_comm_op(op_type: str) -> bool:
    """判断是否为通信 Op."""
    if op_type in _COMM_OPS:
        return True
    # 去掉 _BW/_RE 后缀再判断
    base = op_type.replace("_BW", "").replace("_RE", "")
    return base in ("AllReduce", "AllGather", "ReduceScatter", "Send", "Recv")


def _safe_add(a: Any, b: Any) -> Any:
    """安全累加（处理 0 + expr 的情况）."""
    if a == 0:
        return b
    if b == 0:
        return a
    return a + b


class SymbolicEstimatePass(Pass):
    """ScheduleIR -> ScheduleIR 的透明 Pass.

    从 ScheduleIR 聚合 Op 级数据为符号化估算结果，
    存入 ir.metadata["symbolic_estimate"]，不修改 ScheduleIR 本身。

    Args:
        overlap_ratio: 计算/通信重叠比例（默认 0，可通过 calibrate 校准）
    """

    def __init__(self, overlap_ratio: float = 0.0):
        self.overlap_ratio = overlap_ratio

    def run(self, ir: ScheduleIR) -> ScheduleIR:
        """执行聚合，生成 SymbolicEstimate 并存入 metadata."""
        estimate = self._aggregate(ir)
        ir.metadata["symbolic_estimate"] = estimate
        return ir  # 原样返回

    def _aggregate(self, ir: ScheduleIR) -> SymbolicEstimate:
        """从 ScheduleIR 聚合 Op 级数据."""
        metadata = ir.metadata
        pp = metadata.get("pp", 1)
        num_microbatches = metadata.get("num_microbatches", 1)
        dtype_bytes = metadata.get("dtype_bytes", 2)
        scope = hp.scope.current()
        gradient_checkpointing = scope.parallel.gradient_checkpointing | metadata.get("gradient_checkpointing", False)
        num_layers = metadata.get("num_layers", 1)
        layers_per_stage = num_layers // pp if pp > 0 else num_layers

        # 1. 时间聚合
        forward_time: Any = 0
        backward_time: Any = 0
        recompute_time: Any = 0
        comm_time: Any = 0

        # 分解字典
        time_by_op: dict[str, Any] = {}
        comm_by_type: dict[str, Any] = {}

        for op in ir.iter_ops():
            duration = op.duration
            if duration == 0:
                continue

            op_type = op.op_type or ""
            is_comm = _is_comm_op(op_type)

            if is_comm:
                # 通信 Op
                comm_time = _safe_add(comm_time, duration)
                base_comm = op_type.replace("_BW", "").replace("_RE", "")
                comm_by_type[base_comm] = _safe_add(
                    comm_by_type.get(base_comm, 0), duration
                )
            elif "_RE" in op_type:
                # 重计算 Op
                recompute_time = _safe_add(recompute_time, duration)
            elif op.phase == Phase.FORWARD:
                forward_time = _safe_add(forward_time, duration)
            elif op.phase == Phase.BACKWARD:
                backward_time = _safe_add(backward_time, duration)
            elif op.phase == Phase.OPTIMIZER:
                # 优化器计入 backward（与 SimulatePass 对齐）
                backward_time = _safe_add(backward_time, duration)

            # 按 op_type 分解（去掉 _BW/_RE 后缀归类）
            base_op = op_type.replace("_BW", "").replace("_RE", "")
            time_by_op[base_op] = _safe_add(time_by_op.get(base_op, 0), duration)

        # PP 并行：各 stage 并行执行，per-stage 时间 = 总时间 / pp
        if pp > 1:
            forward_time = forward_time / pp
            backward_time = backward_time / pp
            recompute_time = recompute_time / pp
            comm_time = comm_time / pp
            # 分解字典也需要除以 pp
            time_by_op = {k: v / pp for k, v in time_by_op.items()}
            comm_by_type = {k: v / pp for k, v in comm_by_type.items()}

        # Bubble time: (pp-1) * stage_time / num_microbatches
        if pp > 1 and num_microbatches > 0:
            stage_time = _safe_add(
                _safe_add(forward_time, backward_time),
                _safe_add(recompute_time, comm_time),
            )
            bubble_time = (pp - 1) * stage_time / num_microbatches
        else:
            bubble_time = 0

        # 按 phase 分解
        time_by_phase = {
            "forward": forward_time,
            "backward": backward_time,
            "recompute": recompute_time,
            "communication": comm_time,
            "bubble": bubble_time,
        }

        # 2. 内存聚合
        weight_memory = self._aggregate_weight_memory(ir, pp, dtype_bytes)
        activation_memory = self._aggregate_activation_memory(
            ir, layers_per_stage, gradient_checkpointing, dtype_bytes
        )
        gradient_memory, optimizer_memory = self._aggregate_grad_opt_memory(
            ir, weight_memory, layers_per_stage, metadata
        )

        return SymbolicEstimate(
            forward_time=forward_time,
            backward_time=backward_time,
            recompute_time=recompute_time,
            comm_time=comm_time,
            bubble_time=bubble_time,
            overlap_ratio=self.overlap_ratio,
            weight_memory=weight_memory,
            activation_memory=activation_memory,
            gradient_memory=gradient_memory,
            optimizer_memory=optimizer_memory,
            time_breakdown_by_op=time_by_op,
            time_breakdown_by_phase=time_by_phase,
            comm_breakdown=comm_by_type,
            metadata=metadata.copy(),
        )

    def _aggregate_weight_memory(
        self, ir: ScheduleIR, pp: int, dtype_bytes: int
    ) -> Any:
        """聚合权重内存（按 source_block 去重，除以 PP）."""
        seen_sources: set[str] = set()
        total_weight: Any = 0

        for op in ir.iter_ops():
            if op.op_type != "Matmul" or not op.op:
                continue
            # 只统计前向 Op（避免 _BW/_RE 重复计算）
            if op.phase != Phase.FORWARD:
                continue

            source = op.op.source_block or ""
            if source in seen_sources:
                continue
            seen_sources.add(source)

            attrs = op.op.attrs
            K = attrs.get("K", 0)
            N = attrs.get("N", 0)
            weight_bytes = K * N * dtype_bytes
            if weight_bytes != 0:
                total_weight = _safe_add(total_weight, weight_bytes)

        # PP 分片
        if pp > 1:
            total_weight = total_weight / pp

        return total_weight

    def _aggregate_activation_memory(
        self,
        ir: ScheduleIR,
        layers_per_stage: int,
        gradient_checkpointing: bool,
        dtype_bytes: int,
    ) -> Any:
        """聚合激活内存（单层激活 x 层数）."""
        import re

        # 提取 layer_idx
        def extract_layer_idx(source: str | None) -> int | None:
            if not source:
                return None
            patterns = [
                r"TransformerLayer\(layer(\d+)\)",
                r"layer[_\.]?(\d+)",
                r"layers[_\.](\d+)",
            ]
            for pattern in patterns:
                match = re.search(pattern, source)
                if match:
                    return int(match.group(1))
            return None

        # 按 layer_idx 收集前向 Op（只取 micro_batch=0）
        layer_ops: dict[int, list] = {}
        for op in ir.iter_ops():
            if op.phase != Phase.FORWARD:
                continue
            if not op.op:
                continue
            mb = op.op.attrs.get("micro_batch", 0) if op.op.attrs else 0
            if mb != 0:
                continue
            source = op.op.source_block or ""
            layer_idx = extract_layer_idx(source) or 0
            if layer_idx not in layer_ops:
                layer_ops[layer_idx] = []
            layer_ops[layer_idx].append(op)

        # 计算单层激活
        per_layer_activation: Any = 0
        if layer_ops:
            min_idx = min(layer_ops.keys())
            for op in layer_ops[min_idx]:
                act_bytes = self._get_op_activation_bytes(op, dtype_bytes)
                if act_bytes != 0:
                    per_layer_activation = _safe_add(per_layer_activation, act_bytes)

        # 峰值激活
        if gradient_checkpointing:
            return per_layer_activation  # 只需单层
        else:
            return per_layer_activation * layers_per_stage

    def _get_op_activation_bytes(self, op, dtype_bytes: int) -> Any:
        """计算单个 Op 的输出激活字节数."""
        if not op.op or not op.op.attrs:
            return 0
        attrs = op.op.attrs
        if op.op_type == "Matmul":
            M = attrs.get("M", 0)
            N = attrs.get("N", 0)
            return M * N * dtype_bytes
        elif op.op_type in ("RMSNorm", "LayerNorm"):
            normalized_shape = attrs.get("normalized_shape", 0)
            batch_seq = attrs.get("batch_seq", 0)
            if batch_seq == 0:
                return 0
            return batch_seq * normalized_shape * dtype_bytes
        elif op.op_type in ("Softmax", "SiLU", "GELU", "ReLU"):
            num_elements = attrs.get("num_elements", 0)
            return num_elements * dtype_bytes
        return 0

    def _aggregate_grad_opt_memory(
        self,
        ir: ScheduleIR,
        weight_memory: Any,
        layers_per_stage: int,
        metadata: dict,
    ) -> tuple[Any, Any]:
        """从权重推导梯度和优化器状态内存."""
        from .optimizer import OptimizerConfig

        training = metadata.get("training", True)
        if not training or weight_memory == 0:
            return 0, 0

        optimizer_config_dict = metadata.get("optimizer_config", {})
        if optimizer_config_dict:
            opt_config = OptimizerConfig.from_dict(optimizer_config_dict)
        else:
            opt_config = OptimizerConfig(
                optimizer_type="adam",
                master_weights=True,
                zero_stage=0,
                dp=metadata.get("dp", 1),
            )

        dtype_bytes = opt_config.dtype_bytes or 2
        dp = opt_config.dp or 1
        grad_dtype_bytes = opt_config.grad_accumulation_dtype_bytes or 4

        # 参数数量 = weight_memory / dtype_bytes
        num_params = weight_memory / dtype_bytes

        # 优化器状态
        opt_bytes_per_param = opt_config.get_optimizer_memory_per_param()
        optimizer_memory = num_params * opt_bytes_per_param

        # 梯度内存（与 SimulatePass 对齐）
        single_layer_params = (
            num_params / layers_per_stage if layers_per_stage > 0 else num_params
        )
        block_grad_no_shard = single_layer_params * grad_dtype_bytes
        block_grad_sharded = (
            single_layer_params * dtype_bytes / dp
            if dp > 1
            else single_layer_params * dtype_bytes
        )

        if layers_per_stage <= 1:
            gradient_memory = block_grad_no_shard
        else:
            gradient_memory = block_grad_no_shard + block_grad_sharded * (
                layers_per_stage - 1
            )

        return gradient_memory, optimizer_memory

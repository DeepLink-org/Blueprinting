"""OptimizerPass - Generate backward Ops and optimizer update Ops.

这个 Pass 负责:
1. 为训练场景追加反向 Op (backward pass)
2. 追加优化器更新 Op (optimizer step)
3. 处理 gradient checkpointing (recompute)

只在训练模式下工作，推理模式跳过。
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..types import OpNode, Phase, ScheduledOp, ScheduleIR
from .base import Pass


@dataclass
class OptimizerConfig:
    """优化器配置.

    Attributes:
        optimizer_type: 优化器类型 ('adam', 'sgd', 'adamw')
        dtype_bytes: 数据类型字节数
        master_weights: 是否使用 FP32 master weights
        gradient_checkpointing: 是否启用梯度检查点
        recompute_mode: 重计算模式 ('full', 'attn_only', 'none')
        zero_stage: ZeRO 优化阶段 (0=无, 1=optimizer分片, 2=optimizer+grad分片, 3=全分片)
        dp: Data Parallelism 度数 (用于 ZeRO 分片计算)
        grad_accumulation_dtype_bytes: 梯度累积数据类型字节数 (4=FP32, 2=FP16)
    """

    optimizer_type: str = "adam"
    dtype_bytes: int = 2  # FP16
    master_weights: bool = True
    gradient_checkpointing: bool = False
    recompute_mode: str = "attn_only"
    zero_stage: int = 0  # 0 = no ZeRO
    dp: int = 1  # Data parallelism degree
    grad_accumulation_dtype_bytes: int = (
        4  # FP32 for gradient accumulation (matches Calculon)
    )

    def get_optimizer_memory_per_param(self, include_gradients: bool = False) -> float:
        """获取每个参数的优化器状态内存 (bytes).

        真实训练语义：
        - Adam/AdamW: FP32 m + FP32 v + FP32 master weights (可选) = 8~12 bytes/param
        - SGD with momentum: FP32 m + FP32 master weights (可选) = 4~8 bytes/param
        - 梯度: dtype_bytes per param (可选包含)

        ZeRO 分片：
        - Stage 1: optimizer states 分片到 DP ranks
        - Stage 2: optimizer states + gradients 分片
        - Stage 3: optimizer states + gradients + weights 分片
        """
        # 基础优化器状态 (per param, in bytes)
        if self.optimizer_type in ("adam", "adamw"):
            # Adam: m (FP32) + v (FP32) = 8 bytes
            # + master weights (FP32) if enabled = 4 bytes
            optimizer_bytes = 8.0
            if self.master_weights:
                optimizer_bytes += 4.0  # FP32 master weights
        elif self.optimizer_type == "sgd":
            # SGD with momentum: m (FP32) = 4 bytes
            optimizer_bytes = 4.0
            if self.master_weights:
                optimizer_bytes += 4.0  # FP32 master weights
        else:
            optimizer_bytes = 0.0

        # 可选包含梯度
        if include_gradients:
            optimizer_bytes += self.dtype_bytes  # gradient in training dtype

        # ZeRO 分片
        if self.zero_stage >= 1 and self.dp > 1:
            # Stage 1+: optimizer states 分片
            optimizer_bytes = optimizer_bytes / self.dp

        return optimizer_bytes

    def get_gradient_memory_per_param(self) -> float:
        """获取每个参数的梯度内存 (bytes).

        梯度累积通常使用 FP32 以保持数值精度（与 Calculon 一致）。
        ZeRO Stage 2+ 会将梯度分片到 DP ranks。
        """
        # 使用梯度累积数据类型（默认 FP32=4 bytes，与 Calculon 一致）
        grad_bytes = float(self.grad_accumulation_dtype_bytes)

        if self.zero_stage >= 2 and self.dp > 1:
            grad_bytes = grad_bytes / self.dp

        return grad_bytes

    def get_optimizer_flops_per_param(self) -> int:
        """获取每个参数的优化器更新 FLOPs."""
        if self.optimizer_type in ("adam", "adamw"):
            return 10  # m, v 更新 + weight 更新
        elif self.optimizer_type == "sgd":
            return 3
        return 0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典，用于存入 metadata."""
        return {
            "optimizer_type": self.optimizer_type,
            "dtype_bytes": self.dtype_bytes,
            "master_weights": self.master_weights,
            "gradient_checkpointing": self.gradient_checkpointing,
            "recompute_mode": self.recompute_mode,
            "zero_stage": self.zero_stage,
            "dp": self.dp,
            "grad_accumulation_dtype_bytes": self.grad_accumulation_dtype_bytes,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OptimizerConfig":
        """从字典创建配置."""
        return cls(
            optimizer_type=d.get("optimizer_type", "adam"),
            dtype_bytes=d.get("dtype_bytes", 2),
            master_weights=d.get("master_weights", True),
            gradient_checkpointing=d.get("gradient_checkpointing", False),
            recompute_mode=d.get("recompute_mode", "attn_only"),
            zero_stage=d.get("zero_stage", 0),
            dp=d.get("dp", 1),
            grad_accumulation_dtype_bytes=d.get("grad_accumulation_dtype_bytes", 4),
        )


class OptimizerPass(Pass):
    """为训练场景追加反向 Op 和优化器 Op.

    工作流程:
    1. 遍历前向 Op，生成对应的反向 Op
    2. 追加优化器更新 Op
    3. 处理 gradient checkpointing

    输入: ScheduleIR (只有前向 Op)
    输出: ScheduleIR (前向 + 反向 + 优化器 Op)
    """

    def __init__(
        self,
        optimizer_config: Optional[OptimizerConfig] = None,
        training: bool = True,
        # 硬件参数
        memory_bandwidth: float = 2.0e12,  # 2 TB/s
        peak_flops: float = 312e12,  # 312 TFLOPS
    ):
        """初始化 OptimizerPass.

        Args:
            optimizer_config: 优化器配置
            training: 是否训练模式（推理模式跳过）
            memory_bandwidth: 内存带宽 (bytes/s)
            peak_flops: 峰值算力 (FLOPS)
        """
        self.config = optimizer_config or OptimizerConfig()
        self.training = training
        self.memory_bandwidth = memory_bandwidth
        self.peak_flops = peak_flops

    def run(self, ir: ScheduleIR) -> ScheduleIR:
        """执行优化器 Pass."""
        if not self.training:
            ir.metadata["training"] = False
            return ir

        ir.metadata["training"] = True

        # 收集前向 Op
        forward_ops: List[ScheduledOp] = list(ir.iter_ops())

        # 生成反向 Op
        backward_ops = self._generate_backward_ops(forward_ops)

        # 计算反向开始时间
        if forward_ops:
            fw_end_time = max(op.start + op.duration for op in forward_ops)
        else:
            fw_end_time = 0

        # 设置反向 Op 的时间并添加到 ScheduleIR
        current_time = fw_end_time
        for bw_op in backward_ops:
            bw_op.start = current_time
            current_time = current_time + bw_op.duration
            ir.add_op(bw_op, stage=bw_op.stage, device=bw_op.device)

        # 按 stage 分组计算权重，为每个 stage 创建独立的 optimizer op
        weight_by_stage = self._compute_weights_by_stage(forward_ops)

        # 获取 PP 并行度
        pp = ir.metadata.get("pp", 1)
        num_stages = max(pp, max(weight_by_stage.keys()) + 1) if weight_by_stage else 1

        # 为每个 stage 生成优化器 Op（并行执行）
        total_weight_bytes = sum(weight_by_stage.values())
        if total_weight_bytes > 0:
            for stage in range(num_stages):
                stage_weight = weight_by_stage.get(stage, 0)
                if stage_weight > 0:
                    optimizer_op = self._create_optimizer_op(
                        stage_weight, current_time, stage
                    )
                    ir.add_op(optimizer_op, stage=stage, device=stage)

            # 更新 current_time（所有 stage 的 optimizer 并行执行，取最长时间）
            if weight_by_stage:
                max_stage_weight = max(weight_by_stage.values())
                dummy_op = self._create_optimizer_op(max_stage_weight, current_time, 0)
                current_time = current_time + dummy_op.duration

        # 更新 metadata
        ir.metadata["forward_ops"] = len(forward_ops)
        ir.metadata["backward_ops"] = len(backward_ops)
        ir.metadata["total_weight_bytes"] = total_weight_bytes
        # 存储优化器配置，供 SimulatePass 计算内存使用
        ir.metadata["optimizer_config"] = self.config.to_dict()

        return ir

    def _generate_backward_ops(
        self, forward_ops: List[ScheduledOp]
    ) -> List[ScheduledOp]:
        """为前向 Op 生成对应的反向 Op.

        反向 Op 按逆序排列。
        如果启用了 gradient_checkpointing，会在反向 Op 前插入 recompute Op。
        """
        backward_ops = []

        # 逆序遍历前向 Op
        for fw_op in reversed(forward_ops):
            # 如果启用 gradient checkpointing，先生成 recompute Op
            if self.config.gradient_checkpointing:
                recompute_ops = self._create_recompute_for_op(fw_op)
                backward_ops.extend(recompute_ops)

            # 生成反向 Op
            bw_ops = self._create_backward_for_op(fw_op)
            backward_ops.extend(bw_ops)

        return backward_ops

    def _create_recompute_for_op(self, fw_op: ScheduledOp) -> List[ScheduledOp]:
        """为前向 Op 创建重计算 Op.

        重计算 Op 在反向传播时重新执行前向计算以恢复激活值。
        仅在 gradient_checkpointing 启用时调用。
        """
        op_type = fw_op.op_type

        # 跳过不需要重计算的 Op
        # 通信 Op 不需要重计算
        if op_type in (
            "Send",
            "Recv",
            "OptimizerStep",
            "AllReduce",
            "AllGather",
            "ReduceScatter",
        ):
            return []

        # 根据 recompute_mode 决定哪些 Op 需要重计算
        if self.config.recompute_mode == "attn_only":
            # 只重计算 attention 相关的 Op
            if not self._is_attention_op(fw_op):
                return []
        elif self.config.recompute_mode == "none":
            return []
        # recompute_mode == "full" 时重计算所有 Op

        # 创建 recompute Op
        re_name = f"{fw_op.op.name}_recompute" if fw_op.op else f"recompute_{id(fw_op)}"
        attrs = fw_op.op.attrs.copy() if fw_op.op else {}
        attrs["is_recompute"] = True

        re_op = OpNode(
            name=re_name,
            op_type=f"{op_type}_RE",  # 用 _RE 后缀标记重计算 Op
            inputs=[],
            outputs=[],
            attrs=attrs,
            source_block=fw_op.op.source_block if fw_op.op else None,
            flops=fw_op.op.flops if fw_op.op else 0,
            memory_bytes=fw_op.op.memory_bytes if fw_op.op else 0,
        )

        sched_re = ScheduledOp(
            op=re_op,
            device=fw_op.device,
            stage=fw_op.stage,
            stream=fw_op.stream,
            phase=Phase.BACKWARD,  # recompute 属于 BACKWARD phase
            start=0,  # 由调用者设置
            duration=fw_op.duration,  # 重计算时间约等于前向时间
        )

        return [sched_re]

    def _is_attention_op(self, op: ScheduledOp) -> bool:
        """判断是否是 attention 相关的 Op."""
        if not op.op or not op.op.source_block:
            return False
        source = op.op.source_block.lower()
        return "attn" in source or "attention" in source

    def _create_backward_for_op(self, fw_op: ScheduledOp) -> List[ScheduledOp]:
        """为单个前向 Op 创建反向 Op.

        不同 Op 类型有不同的反向计算:
        - Matmul: agrad + wgrad (2 个 matmul)
        - 激活函数: elementwise backward
        - 通信 Op: 对应的反向通信
        """
        op_type = fw_op.op_type
        attrs = fw_op.op.attrs if fw_op.op else {}

        # 跳过不需要反向的 Op
        if op_type in ("Send", "Recv", "OptimizerStep"):
            return []

        # 计算反向 workload
        bw_flops, bw_memory, bw_duration = self._compute_backward_workload(fw_op)

        if bw_duration <= 0:
            return []

        # 创建反向 Op
        bw_name = f"{fw_op.op.name}_bw" if fw_op.op else f"bw_{id(fw_op)}"
        bw_op = OpNode(
            name=bw_name,
            op_type=f"{op_type}_BW",
            inputs=[],
            outputs=[],
            attrs=attrs.copy(),
            source_block=fw_op.op.source_block if fw_op.op else None,
            flops=bw_flops,
            memory_bytes=bw_memory,
        )

        sched_bw = ScheduledOp(
            op=bw_op,
            device=fw_op.device,
            stage=fw_op.stage,
            stream=fw_op.stream,
            phase=Phase.BACKWARD,
            start=0,  # 由调用者设置
            duration=bw_duration,
        )

        return [sched_bw]

    def _compute_backward_workload(self, fw_op: ScheduledOp) -> tuple:
        """计算反向 Op 的 workload.

        Returns:
            (flops, memory_bytes, duration)
        """
        op_type = fw_op.op_type

        # Matmul 反向: 约 2x 前向
        if op_type == "Matmul":
            fw_flops = fw_op.op.flops if fw_op.op else 0
            fw_memory = fw_op.op.memory_bytes if fw_op.op else 0

            # agrad + wgrad ≈ 2x forward
            bw_flops = (fw_flops or 0) * 2
            bw_memory = (fw_memory or 0) * 2
            bw_duration = fw_op.duration * 2

            return bw_flops, bw_memory, bw_duration

        # Norm 反向: 约 2x 前向
        elif op_type in ("RMSNorm", "LayerNorm"):
            fw_flops = fw_op.op.flops if fw_op.op else 0
            fw_memory = fw_op.op.memory_bytes if fw_op.op else 0

            bw_flops = (fw_flops or 0) * 2
            bw_memory = (fw_memory or 0) * 2
            bw_duration = fw_op.duration * 2

            return bw_flops, bw_memory, bw_duration

        # 激活函数反向: 约等于前向
        elif op_type in ("SiLU", "GELU", "ReLU", "Softmax"):
            fw_flops = fw_op.op.flops if fw_op.op else 0
            fw_memory = fw_op.op.memory_bytes if fw_op.op else 0

            bw_flops = fw_flops or 0
            bw_memory = fw_memory or 0
            bw_duration = fw_op.duration

            return bw_flops, bw_memory, bw_duration

        # 通信 Op 反向: 对应的反向通信
        elif op_type == "AllReduce":
            # AllReduce 反向还是 AllReduce
            return 0, 0, fw_op.duration
        elif op_type == "AllGather":
            # AllGather 反向是 ReduceScatter
            return 0, 0, fw_op.duration
        elif op_type == "ReduceScatter":
            # ReduceScatter 反向是 AllGather
            return 0, 0, fw_op.duration

        # 其他: 简单估算
        else:
            return 0, 0, fw_op.duration

    def _compute_total_weights(self, forward_ops: List[ScheduledOp]) -> float:
        """计算总权重字节数."""
        total = 0
        seen_sources = set()

        for op in forward_ops:
            if op.op_type == "Matmul" and op.op:
                # 从 source_block 去重
                source = op.op.source_block or ""
                if source in seen_sources:
                    continue
                seen_sources.add(source)

                # 计算权重: K * N * dtype_bytes
                attrs = op.op.attrs
                K = attrs.get("K", 1)
                N = attrs.get("N", 1)
                weight_bytes = K * N * self.config.dtype_bytes
                total += weight_bytes

        return total

    def _compute_weights_by_stage(
        self, forward_ops: List[ScheduledOp]
    ) -> Dict[int, float]:
        """按 stage 计算权重字节数."""
        weights_by_stage: Dict[int, float] = {}
        seen_sources: Dict[int, set] = {}

        for op in forward_ops:
            if op.op_type == "Matmul" and op.op:
                stage = op.stage

                # 初始化该 stage 的去重集合
                if stage not in seen_sources:
                    seen_sources[stage] = set()
                    weights_by_stage[stage] = 0

                # 从 source_block 去重
                source = op.op.source_block or ""
                if source in seen_sources[stage]:
                    continue
                seen_sources[stage].add(source)

                # 计算权重: K * N * dtype_bytes
                attrs = op.op.attrs
                K = attrs.get("K", 1)
                N = attrs.get("N", 1)
                weight_bytes = K * N * self.config.dtype_bytes
                weights_by_stage[stage] += weight_bytes

        return weights_by_stage

    def _create_optimizer_op(
        self, weight_bytes: float, start_time: float, stage: int = 0
    ) -> ScheduledOp:
        """创建优化器更新 Op.

        Args:
            weight_bytes: 权重字节数
            start_time: 开始时间
            stage: PP stage (默认为 0)
        """
        num_params = weight_bytes / self.config.dtype_bytes

        # 内存访问: 读写优化器状态
        mem_per_param = self.config.get_optimizer_memory_per_param()
        total_memory = num_params * mem_per_param

        # FLOPs
        flops_per_param = self.config.get_optimizer_flops_per_param()
        total_flops = num_params * flops_per_param

        # Duration: 内存带宽限制
        memory_time = total_memory / self.memory_bandwidth
        compute_time = total_flops / self.peak_flops
        duration = memory_time + compute_time  # 优化器是 element-wise，不能 overlap

        op = OpNode(
            name=f"optimizer_step_s{stage}",
            op_type="OptimizerStep",
            inputs=[],
            outputs=[],
            attrs={
                "weight_bytes": weight_bytes,
                "num_params": num_params,
                "optimizer_type": self.config.optimizer_type,
                "stage": stage,
            },
            flops=total_flops,
            memory_bytes=total_memory,
        )

        return ScheduledOp(
            op=op,
            device=stage,  # PP 并行: device = stage
            stage=stage,
            stream="compute",
            phase=Phase.OPTIMIZER,
            start=start_time,
            duration=duration,
        )

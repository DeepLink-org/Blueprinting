"""OptimizerPass - Generate backward Ops and optimizer update Ops.

这个 Pass 负责:
1. 为训练场景追加反向 Op (backward pass)
2. 追加优化器更新 Op (optimizer step)
3. 处理 gradient checkpointing (recompute)

只在训练模式下工作，推理模式跳过。
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from sympy import Expr

from .base import Pass
from ..types import ScheduleIR, ScheduledOp, OpNode


@dataclass
class OptimizerConfig:
    """优化器配置.
    
    Attributes:
        optimizer_type: 优化器类型 ('adam', 'sgd', 'adamw')
        dtype_bytes: 数据类型字节数
        master_weights: 是否使用 FP32 master weights
        gradient_checkpointing: 是否启用梯度检查点
        recompute_mode: 重计算模式 ('full', 'attn_only', 'none')
    """
    optimizer_type: str = "adam"
    dtype_bytes: int = 2  # FP16
    master_weights: bool = True
    gradient_checkpointing: bool = False
    recompute_mode: str = "attn_only"
    
    def get_optimizer_memory_per_param(self) -> int:
        """获取每个参数的优化器状态内存 (bytes)."""
        if self.optimizer_type in ("adam", "adamw"):
            # Adam: m + v + (master weights)
            if self.master_weights:
                return 12  # FP32 m + FP32 v + FP32 master = 12 bytes
            else:
                return 8   # FP32 m + FP32 v = 8 bytes
        elif self.optimizer_type == "sgd":
            if self.master_weights:
                return 8   # FP32 m + FP32 master
            else:
                return 4   # FP32 m
        return 0
    
    def get_optimizer_flops_per_param(self) -> int:
        """获取每个参数的优化器更新 FLOPs."""
        if self.optimizer_type in ("adam", "adamw"):
            return 10  # m, v 更新 + weight 更新
        elif self.optimizer_type == "sgd":
            return 3
        return 0


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
        peak_flops: float = 312e12,        # 312 TFLOPS
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
        
        # 计算总权重
        total_weight_bytes = self._compute_total_weights(forward_ops)
        
        # 生成优化器 Op
        if total_weight_bytes > 0:
            optimizer_op = self._create_optimizer_op(total_weight_bytes, current_time)
            ir.add_op(optimizer_op, stage=0, device=0)
            current_time = current_time + optimizer_op.duration
        
        # 更新 metadata
        ir.metadata["forward_ops"] = len(forward_ops)
        ir.metadata["backward_ops"] = len(backward_ops)
        ir.metadata["total_weight_bytes"] = total_weight_bytes
        
        return ir
    
    def _generate_backward_ops(self, forward_ops: List[ScheduledOp]) -> List[ScheduledOp]:
        """为前向 Op 生成对应的反向 Op.
        
        反向 Op 按逆序排列。
        """
        backward_ops = []
        
        # 逆序遍历前向 Op
        for fw_op in reversed(forward_ops):
            bw_ops = self._create_backward_for_op(fw_op)
            backward_ops.extend(bw_ops)
        
        return backward_ops
    
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
    
    def _create_optimizer_op(self, weight_bytes: float, start_time: float) -> ScheduledOp:
        """创建优化器更新 Op."""
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
            name="optimizer_step",
            op_type="OptimizerStep",
            inputs=[],
            outputs=[],
            attrs={
                "weight_bytes": weight_bytes,
                "num_params": num_params,
                "optimizer_type": self.config.optimizer_type,
            },
            flops=total_flops,
            memory_bytes=total_memory,
        )
        
        return ScheduledOp(
            op=op,
            device=0,
            stage=0,
            stream="compute",
            start=start_time,
            duration=duration,
        )

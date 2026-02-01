"""SchedulePass - Compute workload and timing for ScheduleIR.

这个 Pass 负责:
1. 为每个 Op 计算 workload (flops, memory_bytes, comm_bytes)
2. 使用 roofline 模型计算 duration
3. 设置 start 时间（顺序调度）
4. 处理设备分配

职责分离:
- ExpandPass: 纯展开，生成 Op 结构
- SchedulePass: 计算 workload、duration、设备分配
"""

from typing import Any, Dict, Optional, Union

from sympy import Expr, Symbol

from .base import Pass
from ..types import ScheduleIR, ScheduledOp, OpNode
from ..symmax import SymMax


def _symbolic_max(*args):
    """符号安全的 max 函数."""
    has_symbolic = any(isinstance(a, Expr) for a in args)
    if has_symbolic:
        return SymMax(*args)
    return max(args)


class SchedulePass(Pass):
    """计算 ScheduleIR 中每个 Op 的 workload 和 timing.
    
    这个 Pass 遍历 ScheduleIR 中的所有 Op，计算:
    - workload: flops, memory_bytes, comm_bytes
    - duration: 基于 roofline 模型
    - start: 顺序调度的开始时间
    
    输入: ScheduleIR (无 workload 和 timing)
    输出: ScheduleIR (有 workload 和 timing)
    """
    
    def __init__(
        self,
        # 硬件参数
        peak_tflops: float = 312.0,      # A100 FP16 峰值
        memory_bandwidth: float = 2.0e12, # 2 TB/s
        network_bandwidth: float = 400e9, # 400 Gbps
        # 数据类型
        dtype_bytes: int = 2,             # float16
    ):
        """初始化 SchedulePass.
        
        Args:
            peak_tflops: 计算峰值 (TFLOPS)
            memory_bandwidth: 内存带宽 (bytes/s)
            network_bandwidth: 网络带宽 (bytes/s)
            dtype_bytes: 数据类型字节数
        """
        self.peak_tflops = peak_tflops
        self.peak_flops = peak_tflops * 1e12
        self.memory_bandwidth = memory_bandwidth
        self.network_bandwidth = network_bandwidth
        self.dtype_bytes = dtype_bytes
    
    def run(self, ir: ScheduleIR) -> ScheduleIR:
        """执行调度计算."""
        metadata = ir.metadata
        
        # 提取常用参数
        batch = metadata.get("batch_size", metadata.get("batch", 1))
        seq = metadata.get("seq_len", metadata.get("seq", 2048))
        hidden = metadata.get("hidden", 4096)
        feedforward = metadata.get("feedforward", hidden * 4)
        
        batch_seq = batch * seq
        
        # 当前时间 (用于顺序调度)
        current_time: Union[float, Expr] = 0
        
        # 遍历所有 Op
        for op in ir.iter_ops():
            # 计算 workload
            self._compute_workload(op.op, batch_seq, hidden, feedforward, metadata)
            
            # 计算 duration
            duration = self._compute_duration(op.op)
            op.duration = duration
            
            # 设置 start 时间 (顺序调度)
            op.start = current_time
            current_time = current_time + duration
        
        return ir
    
    def _compute_workload(
        self,
        op: OpNode,
        batch_seq,
        hidden,
        feedforward,
        metadata: Dict,
    ) -> None:
        """计算单个 Op 的 workload."""
        op_type = op.op_type
        attrs = op.attrs
        
        if op_type == "Matmul":
            self._compute_matmul(op, attrs)
        elif op_type in ("RMSNorm", "LayerNorm"):
            self._compute_norm(op, attrs, batch_seq)
        elif op_type == "Softmax":
            self._compute_softmax(op, attrs)
        elif op_type in ("SiLU", "GELU", "ReLU"):
            self._compute_activation(op, attrs, batch_seq, feedforward)
        elif op_type in ("Add", "Mul"):
            self._compute_elementwise(op, attrs, batch_seq, hidden)
        elif op_type in ("AllReduce", "AllGather", "ReduceScatter"):
            self._compute_collective(op, attrs)
        elif op_type in ("Send", "Recv"):
            self._compute_p2p(op, attrs)
        else:
            # 未知类型，设为 0
            op.flops = 0
            op.memory_bytes = 0
            op.comm_bytes = 0
    
    def _compute_matmul(self, op: OpNode, attrs: Dict) -> None:
        """计算 Matmul workload."""
        M = attrs.get("M", 1)
        K = attrs.get("K", 1)
        N = attrs.get("N", 1)
        
        # FLOPs: 2 * M * K * N
        op.flops = 2 * M * K * N
        
        # Memory: (M*K + K*N + M*N) * dtype_bytes
        op.memory_bytes = (M * K + K * N + M * N) * self.dtype_bytes
        
        op.comm_bytes = 0
    
    def _compute_norm(self, op: OpNode, attrs: Dict, batch_seq) -> None:
        """计算 RMSNorm/LayerNorm workload."""
        normalized_shape = attrs.get("normalized_shape", 4096)
        num_elements = batch_seq * normalized_shape
        
        # FLOPs: ~5 ops per element (square, sum, rsqrt, mul, add)
        op.flops = 5 * num_elements
        
        # Memory: 2 * num_elements (read + write)
        op.memory_bytes = 2 * num_elements * self.dtype_bytes
        
        op.comm_bytes = 0
    
    def _compute_softmax(self, op: OpNode, attrs: Dict) -> None:
        """计算 Softmax workload."""
        num_elements = attrs.get("num_elements", 1)
        
        # FLOPs: ~5 ops per element (max, sub, exp, sum, div)
        op.flops = 5 * num_elements
        
        # Memory: 2 * num_elements
        op.memory_bytes = 2 * num_elements * self.dtype_bytes
        
        op.comm_bytes = 0
    
    def _compute_activation(self, op: OpNode, attrs: Dict, batch_seq, feedforward) -> None:
        """计算激活函数 workload (SiLU, GELU, ReLU)."""
        num_elements = attrs.get("num_elements", batch_seq * feedforward)
        
        # SiLU: x * sigmoid(x) ≈ 4 ops
        # GELU: 0.5 * x * (1 + tanh(...)) ≈ 8 ops
        # ReLU: max(0, x) ≈ 1 op
        if op.op_type == "SiLU":
            op.flops = 4 * num_elements
        elif op.op_type == "GELU":
            op.flops = 8 * num_elements
        else:  # ReLU
            op.flops = num_elements
        
        op.memory_bytes = 2 * num_elements * self.dtype_bytes
        op.comm_bytes = 0
    
    def _compute_elementwise(self, op: OpNode, attrs: Dict, batch_seq, hidden) -> None:
        """计算 Add/Mul workload."""
        num_elements = attrs.get("num_elements", batch_seq * hidden)
        
        op.flops = num_elements
        op.memory_bytes = 3 * num_elements * self.dtype_bytes  # 2 read + 1 write
        op.comm_bytes = 0
    
    def _compute_collective(self, op: OpNode, attrs: Dict) -> None:
        """计算集合通信 workload (AllReduce, AllGather, ReduceScatter)."""
        data_size = attrs.get("data_size", 1)
        num_peers = attrs.get("num_peers", 8)
        
        op.flops = 0
        op.memory_bytes = 0
        
        # Ring AllReduce: 2 * (n-1)/n * data_size
        # AllGather: (n-1)/n * data_size
        # ReduceScatter: (n-1)/n * data_size
        if op.op_type == "AllReduce":
            op.comm_bytes = 2 * (num_peers - 1) / num_peers * data_size * self.dtype_bytes
        else:
            op.comm_bytes = (num_peers - 1) / num_peers * data_size * self.dtype_bytes
    
    def _compute_p2p(self, op: OpNode, attrs: Dict) -> None:
        """计算点对点通信 workload (Send, Recv)."""
        data_size = attrs.get("data_size", 1)
        
        op.flops = 0
        op.memory_bytes = 0
        op.comm_bytes = data_size * self.dtype_bytes
    
    def _compute_duration(self, op: OpNode) -> Union[float, Expr]:
        """基于 roofline 模型计算 duration."""
        flops = op.flops or 0
        memory = op.memory_bytes or 0
        comm = op.comm_bytes or 0
        
        # 计算时间
        if flops > 0:
            compute_time = flops / self.peak_flops
        else:
            compute_time = 0
        
        # 内存时间
        if memory > 0:
            memory_time = memory / self.memory_bandwidth
        else:
            memory_time = 0
        
        # 通信时间
        if comm > 0:
            comm_time = comm / self.network_bandwidth
        else:
            comm_time = 0
        
        # Roofline: max(compute, memory)
        # 对于通信 Op，取通信时间
        if op.op_type in ("AllReduce", "AllGather", "ReduceScatter", "Send", "Recv"):
            duration = comm_time
        else:
            duration = _symbolic_max(compute_time, memory_time)
        
        return duration

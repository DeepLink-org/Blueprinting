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
        network_efficiency: float = 0.65, # 网络效率 (与 Calculon 对齐)
        network_latency: float = 10e-6,  # 10µs 网络延迟
        compute_efficiency: float = 0.95, # 计算效率 (与 Calculon 对齐)
        # 数据类型
        dtype_bytes: int = 2,             # float16
        # 通信模型参数 (与 Calculon 对齐)
        # all_reduce_offset: Calculon 使用 offset=1 表示双向通信
        # 公式: comm_size_effective = comm_size * (1 + offset/num_peers)
        all_reduce_offset: float = 1.0,
    ):
        """初始化 SchedulePass.
        
        Args:
            peak_tflops: 计算峰值 (TFLOPS)
            memory_bandwidth: 内存带宽 (bytes/s)
            network_bandwidth: 网络带宽 (bytes/s)
            network_efficiency: 网络效率 (0-1，默认 0.65 与 Calculon H100 配置对齐)
            network_latency: 网络延迟 (seconds)
            compute_efficiency: 计算效率 (0-1，默认 0.95)
            dtype_bytes: 数据类型字节数
            all_reduce_offset: AllReduce 通信偏移量 (Calculon 模型，默认 1.0)
        """
        self.peak_tflops = peak_tflops
        self.peak_flops = peak_tflops * 1e12 * compute_efficiency  # 应用效率
        self.memory_bandwidth = memory_bandwidth
        self.network_bandwidth = network_bandwidth
        self.network_efficiency = network_efficiency
        self.network_latency = network_latency
        self.compute_efficiency = compute_efficiency
        self.dtype_bytes = dtype_bytes
        self.all_reduce_offset = all_reduce_offset
    
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
        """计算集合通信 workload (AllReduce, AllGather, ReduceScatter).
        
        使用与 Calculon 相同的通信模型:
        - AllReduce: op_size = data_size * (1 + offset/num_peers)
          其中 offset=1 表示双向通信（reduce + broadcast）
        - AllGather/ReduceScatter: op_size = data_size * (n-1)/n
        
        这个模型的物理意义:
        - AllReduce 的 offset=1 近似于两阶段通信（reduce-scatter + all-gather）
        - 最终通信量 = 原始数据 + 每个节点的 chunk
        """
        data_size = attrs.get("data_size", 1)
        num_peers = attrs.get("num_peers", 8)
        
        op.flops = 0
        op.memory_bytes = 0
        
        # 基础通信量 (bytes)
        base_comm_bytes = data_size * self.dtype_bytes
        
        if op.op_type == "AllReduce":
            # Calculon 模型: comm_size * (1 + offset/num_peers)
            # 这比 Ring 公式 2*(n-1)/n 更准确地反映了实际通信开销
            op.comm_bytes = base_comm_bytes * (1 + self.all_reduce_offset / num_peers)
        elif op.op_type == "AllGather":
            # AllGather: 收集所有节点的数据，通信量 = (n-1)/n * data_size
            op.comm_bytes = base_comm_bytes * (num_peers - 1) / num_peers
        elif op.op_type == "ReduceScatter":
            # ReduceScatter: 分散归约，通信量 = (n-1)/n * data_size
            op.comm_bytes = base_comm_bytes * (num_peers - 1) / num_peers
        else:
            op.comm_bytes = base_comm_bytes
        
    
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
        
        # 通信时间 (Calculon 模型: latency + comm_bytes / (bandwidth * efficiency))
        # 这是一个物理上有意义的模型:
        # - network_efficiency 反映了实际带宽利用率 (NVLink 通常 0.65)
        # - network_latency 是固定的启动延迟
        if comm > 0:
            effective_bandwidth = self.network_bandwidth * self.network_efficiency
            comm_time = self.network_latency + comm / effective_bandwidth
        else:
            comm_time = 0
        
        # Roofline: max(compute, memory)
        # 对于通信 Op，取通信时间
        if op.op_type in ("AllReduce", "AllGather", "ReduceScatter", "Send", "Recv"):
            duration = comm_time
        else:
            duration = _symbolic_max(compute_time, memory_time)
        
        return duration

"""WorkloadPass - Fill workload information in GraphIR.

This pass computes FLOPs, memory access, and communication bytes
for each operation in the graph based on operation type and attributes.
"""

from typing import Dict, Optional, Union

from sympy import Expr, Symbol

from ..graph import GraphIR, OpNode
from .base import Pass


class WorkloadPass(Pass):
    """Pass to compute workload metrics for graph operations.
    
    This pass fills in:
    - flops_fw, flops_bw: Forward and backward FLOPs
    - memory_fw, memory_bw: Memory accessed
    - comm_bytes_fw, comm_bytes_bw: Communication bytes
    - weight_bytes: Weight tensor size
    - activation_bytes: Activation tensor size
    
    The formulas are based on standard deep learning operations
    and match the calculations in blueprinting.nn.modules.
    """
    
    def __init__(self, dtype_bytes: int = 2, count_add: bool = True):
        """Initialize WorkloadPass.
        
        Args:
            dtype_bytes: Bytes per element (default 2 for float16)
            count_add: Whether to count additions in FLOPs
        """
        self.dtype_bytes = dtype_bytes
        self.count_add = count_add
    
    def run(self, ir: GraphIR) -> GraphIR:
        """Execute the workload pass."""
        for node in ir.nodes.values():
            self._compute_workload(node, ir)
        return ir
    
    def _compute_workload(self, node: OpNode, ir: GraphIR) -> None:
        """Compute workload for a single node."""
        op_type = node.op_type
        attrs = node.attrs
        
        if op_type == "Linear":
            self._compute_linear(node, attrs)
        elif op_type == "RMSNorm":
            self._compute_rmsnorm(node, attrs)
        elif op_type == "Attention":
            self._compute_attention(node, attrs)
        elif op_type == "SiLU":
            self._compute_silu(node, attrs)
        elif op_type == "Add":
            self._compute_add(node, attrs)
        elif op_type == "Mul":
            self._compute_mul(node, attrs)
        elif op_type == "Softmax":
            self._compute_softmax(node, attrs)
        elif op_type in ("AllReduce", "AllGather", "ReduceScatter"):
            self._compute_comm(node, attrs)
        else:
            # Unknown op - set to 0
            node.flops_fw = 0
            node.flops_bw = 0
            node.memory_fw = 0
            node.memory_bw = 0
        
        # Aggregate flops
        if node.flops_fw is not None and node.flops_bw is not None:
            node.flops = node.flops_fw + node.flops_bw
    
    def _compute_linear(self, node: OpNode, attrs: Dict) -> None:
        """Compute workload for Linear layer.
        
        Forward: Y = X @ W + b
        FLOPs: 2 * M * N * K (with add) or M * N * K (without)
        where M = batch * seq, N = in_features, K = out_features
        
        Backward:
        - dX = dY @ W^T: 2 * M * K * N
        - dW = X^T @ dY: 2 * M * N * K
        """
        in_features = attrs.get("in_features", 1)
        out_features = attrs.get("out_features", 1)
        bias = attrs.get("bias", True)
        
        # Get batch_seq from graph symbols or attrs
        batch_seq = attrs.get("batch_seq", Symbol("batch_seq"))
        
        # FLOPs
        if self.count_add:
            if bias:
                # M * N * (2*K - 1) + M*K (bias add)
                node.flops_fw = batch_seq * in_features * (2 * out_features)
            else:
                node.flops_fw = batch_seq * in_features * (2 * out_features - 1)
        else:
            node.flops_fw = batch_seq * in_features * out_features
        
        # Backward: split into agrad (dX) and wgrad (dW)
        # dX = dY @ W^T: [B*S, out] @ [out, in] -> [B*S, in]
        # FLOPs: 2 * B*S * out * in = 2 * B*S * in * out (same as FW)
        node.flops_agrad = batch_seq * out_features * (2 * in_features)
        
        # dW = X^T @ dY: [in, B*S] @ [B*S, out] -> [in, out]
        # FLOPs: 2 * in * B*S * out = 2 * B*S * in * out (same as FW)
        node.flops_wgrad = batch_seq * in_features * (2 * out_features)
        
        # Total backward FLOPs (for compatibility)
        node.flops_bw = node.flops_agrad + node.flops_wgrad
        
        # Memory: input + output + weights
        node.memory_fw = (batch_seq * in_features + batch_seq * out_features + in_features * out_features) * self.dtype_bytes
        
        # AGrad memory: dY (input), W (read), dX (output)
        node.memory_agrad = (batch_seq * out_features + in_features * out_features + batch_seq * in_features) * self.dtype_bytes
        
        # WGrad memory: X (stored), dY (input), dW (output)
        node.memory_wgrad = (batch_seq * in_features + batch_seq * out_features + in_features * out_features) * self.dtype_bytes
        
        # Total backward memory (for compatibility)
        node.memory_bw = (batch_seq * in_features + batch_seq * out_features + in_features * out_features + in_features * out_features) * self.dtype_bytes
        
        # Weight bytes
        weight_size = in_features * out_features
        if bias:
            weight_size += out_features
        node.weight_bytes = weight_size * self.dtype_bytes
        
        # Activation bytes (input stored for backward)
        node.activation_bytes = batch_seq * in_features * self.dtype_bytes
    
    def _compute_rmsnorm(self, node: OpNode, attrs: Dict) -> None:
        """Compute workload for RMSNorm.
        
        Forward: 4N ops (square, mean, rsqrt, mul)
        Backward: ~8N ops
        """
        normalized_shape = attrs.get("normalized_shape", Symbol("hidden"))
        batch_seq = attrs.get("batch_seq", Symbol("batch_seq"))
        
        num_elements = batch_seq * normalized_shape
        
        # Forward: square + mean + rsqrt + scale = 4N
        node.flops_fw = 4 * num_elements
        
        # Backward: ~8N
        node.flops_bw = 8 * num_elements
        
        # Memory
        node.memory_fw = 2 * num_elements * self.dtype_bytes  # input + output
        node.memory_bw = 2 * num_elements * self.dtype_bytes
        
        # Weight bytes (gamma parameter)
        node.weight_bytes = normalized_shape * self.dtype_bytes
    
    def _compute_attention(self, node: OpNode, attrs: Dict) -> None:
        """Compute workload for Attention (Q@K^T, softmax, @V).
        
        Q@K^T: 2 * B * H * S * S * D
        Softmax: 5 * B * H * S * S
        Score@V: 2 * B * H * S * D * S
        """
        num_heads = attrs.get("num_heads", Symbol("num_heads"))
        head_dim = attrs.get("head_dim", Symbol("head_dim"))
        seq_len = attrs.get("seq_len", Symbol("seq_len"))
        batch_size = attrs.get("batch_size", Symbol("batch_size"))
        
        # QK^T: [B, H, S, D] @ [B, H, D, S] -> [B, H, S, S]
        qk_flops = 2 * batch_size * num_heads * seq_len * seq_len * head_dim
        
        # Softmax: 5 * B * H * S * S
        softmax_flops = 5 * batch_size * num_heads * seq_len * seq_len
        
        # Score@V: [B, H, S, S] @ [B, H, S, D] -> [B, H, S, D]
        sv_flops = 2 * batch_size * num_heads * seq_len * head_dim * seq_len
        
        node.flops_fw = qk_flops + softmax_flops + sv_flops
        node.flops_bw = 2 * node.flops_fw  # Backward is roughly 2x forward
        
        # Memory: Q, K, V, Score, Output
        activation_size = batch_size * num_heads * seq_len * head_dim
        score_size = batch_size * num_heads * seq_len * seq_len
        node.memory_fw = (3 * activation_size + score_size + activation_size) * self.dtype_bytes
        node.memory_bw = node.memory_fw
        
        # Activations stored for backward
        node.activation_bytes = score_size * self.dtype_bytes
    
    def _compute_silu(self, node: OpNode, attrs: Dict) -> None:
        """Compute workload for SiLU activation.
        
        SiLU(x) = x * sigmoid(x)
        Forward: 4N (sigmoid: exp + add + div, then mul)
        Backward: 6N
        """
        num_elements = attrs.get("num_elements", Symbol("num_elements"))
        
        node.flops_fw = 4 * num_elements
        node.flops_bw = 6 * num_elements
        
        node.memory_fw = 2 * num_elements * self.dtype_bytes
        node.memory_bw = 2 * num_elements * self.dtype_bytes
        
        node.weight_bytes = 0
        node.activation_bytes = num_elements * self.dtype_bytes
    
    def _compute_add(self, node: OpNode, attrs: Dict) -> None:
        """Compute workload for element-wise Add.
        
        Forward: N additions
        Backward: 0 (gradient is just passed through)
        """
        num_elements = attrs.get("num_elements", Symbol("num_elements"))
        
        node.flops_fw = num_elements
        node.flops_bw = 0  # dy/dx = 1, no computation
        
        node.memory_fw = 3 * num_elements * self.dtype_bytes  # 2 inputs + 1 output
        node.memory_bw = 2 * num_elements * self.dtype_bytes
        
        node.weight_bytes = 0
    
    def _compute_mul(self, node: OpNode, attrs: Dict) -> None:
        """Compute workload for element-wise Mul.
        
        Forward: N multiplications
        Backward: 2N (gradient requires multiplying by the other input)
        """
        num_elements = attrs.get("num_elements", Symbol("num_elements"))
        
        node.flops_fw = num_elements
        node.flops_bw = 2 * num_elements
        
        node.memory_fw = 3 * num_elements * self.dtype_bytes
        node.memory_bw = 4 * num_elements * self.dtype_bytes
        
        node.weight_bytes = 0
        node.activation_bytes = 2 * num_elements * self.dtype_bytes
    
    def _compute_softmax(self, node: OpNode, attrs: Dict) -> None:
        """Compute workload for Softmax.
        
        Forward: 5N (max, sub, exp, sum, div)
        Backward: 8N
        """
        num_elements = attrs.get("num_elements", Symbol("num_elements"))
        
        node.flops_fw = 5 * num_elements
        node.flops_bw = 8 * num_elements
        
        node.memory_fw = 2 * num_elements * self.dtype_bytes
        node.memory_bw = 2 * num_elements * self.dtype_bytes
        
        node.weight_bytes = 0
        node.activation_bytes = num_elements * self.dtype_bytes
    
    def _compute_comm(self, node: OpNode, attrs: Dict) -> None:
        """Compute workload for communication operations.
        
        Communication ops have no FLOPs but have communication bytes.
        """
        num_peers = attrs.get("num_peers", 1)
        data_size = attrs.get("data_size", Symbol("data_size"))
        comm_type = node.op_type
        
        node.flops_fw = 0
        node.flops_bw = 0
        node.memory_fw = 0
        node.memory_bw = 0
        node.weight_bytes = 0
        
        # Communication bytes depend on the collective type
        if comm_type == "AllReduce":
            # Ring all-reduce: 2 * (n-1)/n * data_size
            node.comm_bytes_fw = 2 * (num_peers - 1) / num_peers * data_size * self.dtype_bytes
            node.comm_bytes_bw = node.comm_bytes_fw
        elif comm_type == "AllGather":
            # All-gather: (n-1)/n * data_size * n = (n-1) * data_size
            node.comm_bytes_fw = (num_peers - 1) * data_size * self.dtype_bytes
            node.comm_bytes_bw = 0
        elif comm_type == "ReduceScatter":
            # Reduce-scatter: (n-1)/n * data_size
            node.comm_bytes_fw = (num_peers - 1) / num_peers * data_size * self.dtype_bytes
            node.comm_bytes_bw = 0
        else:
            node.comm_bytes_fw = 0
            node.comm_bytes_bw = 0
        
        if node.comm_bytes_fw is not None:
            node.comm_bytes = node.comm_bytes_fw + (node.comm_bytes_bw or 0)

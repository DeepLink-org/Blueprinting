"""WorkloadPass - Fill workload information in hierarchical GraphIR.

This pass computes FLOPs, memory access, and communication bytes
for each operation in the graph based on operation type and attributes.
"""

from typing import Dict, Optional, Union

from sympy import Expr, Symbol

from ..graph import GraphIR, OpNode
from .base import Pass


class WorkloadPass(Pass):
    """Pass to compute workload metrics for graph operations.
    
    This pass traverses the hierarchical GraphIR and fills in workload
    information for each OpNode:
    - flops_fw, flops_bw: Forward and backward FLOPs
    - memory_fw, memory_bw: Memory accessed
    - comm_bytes_fw, comm_bytes_bw: Communication bytes
    - weight_bytes: Weight tensor size
    - activation_bytes: Activation tensor size
    """
    
    name = "WorkloadPass"
    
    def __init__(self, dtype_bytes: int = 2, count_add: bool = True):
        """Initialize WorkloadPass.
        
        Args:
            dtype_bytes: Bytes per element (default 2 for float16)
            count_add: Whether to count additions in FLOPs
        """
        self.dtype_bytes = dtype_bytes
        self.count_add = count_add
    
    def run(self, ir: GraphIR) -> GraphIR:
        """Execute the workload pass.
        
        Traverses the hierarchical graph and computes workload for each op.
        """
        metadata = ir.metadata
        batch = metadata.get("batch_size", metadata.get("batch", Symbol("batch")))
        seq = metadata.get("seq_len", metadata.get("seq", Symbol("seq")))
        hidden = metadata.get("hidden", Symbol("hidden"))
        feedforward = metadata.get("feedforward", Symbol("feedforward"))
        
        batch_seq = batch * seq
        
        for op in ir.iter_ops():
            self._compute_workload(op, batch_seq, hidden, feedforward, metadata)
        
        return ir
    
    def _compute_workload(self, node: OpNode, batch_seq, hidden, feedforward,
                           metadata: Dict) -> None:
        """Compute workload for a single op node."""
        op_type = node.op_type
        attrs = node.attrs
        
        bs = attrs.get("batch_seq", batch_seq)
        
        if op_type == "Linear":
            self._compute_linear(node, attrs, bs)
        elif op_type in ("RMSNorm", "LayerNorm"):
            self._compute_norm(node, attrs, bs)
        elif op_type == "Attention":
            self._compute_attention(node, attrs, metadata)
        elif op_type == "SiLU":
            self._compute_silu(node, attrs, bs, feedforward)
        elif op_type == "GELU":
            self._compute_gelu(node, attrs, bs, feedforward)
        elif op_type == "Add":
            self._compute_add(node, attrs, bs, hidden)
        elif op_type == "Mul":
            self._compute_mul(node, attrs, bs, hidden)
        elif op_type == "Softmax":
            self._compute_softmax(node, attrs)
        elif op_type in ("AllReduce", "AllGather", "ReduceScatter"):
            self._compute_comm(node, attrs)
        else:
            node.flops_fw = 0
            node.flops_bw = 0
            node.memory_fw = 0
            node.memory_bw = 0
        
        if node.flops_fw is not None and node.flops_bw is not None:
            node.flops = node.flops_fw + node.flops_bw
    
    def _compute_linear(self, node: OpNode, attrs: Dict, batch_seq) -> None:
        """Compute workload for Linear layer."""
        in_features = attrs.get("in_features", 1)
        out_features = attrs.get("out_features", 1)
        bias = attrs.get("bias", True)
        
        if self.count_add:
            if bias:
                node.flops_fw = batch_seq * in_features * (2 * out_features)
            else:
                node.flops_fw = batch_seq * in_features * (2 * out_features - 1)
        else:
            node.flops_fw = batch_seq * in_features * out_features
        
        node.flops_agrad = batch_seq * out_features * (2 * in_features)
        node.flops_wgrad = batch_seq * in_features * (2 * out_features)
        node.flops_bw = node.flops_agrad + node.flops_wgrad
        
        node.memory_fw = (batch_seq * in_features + batch_seq * out_features + 
                          in_features * out_features) * self.dtype_bytes
        node.memory_agrad = (batch_seq * out_features + in_features * out_features + 
                             batch_seq * in_features) * self.dtype_bytes
        node.memory_wgrad = (batch_seq * in_features + batch_seq * out_features + 
                             in_features * out_features) * self.dtype_bytes
        node.memory_bw = (batch_seq * in_features + batch_seq * out_features + 
                          2 * in_features * out_features) * self.dtype_bytes
        
        weight_size = in_features * out_features
        if bias:
            weight_size = weight_size + out_features
        node.weight_bytes = weight_size * self.dtype_bytes
        
        node.activation_bytes = batch_seq * in_features * self.dtype_bytes
    
    def _compute_norm(self, node: OpNode, attrs: Dict, batch_seq) -> None:
        """Compute workload for RMSNorm/LayerNorm."""
        normalized_shape = attrs.get("normalized_shape", Symbol("hidden"))
        num_elements = batch_seq * normalized_shape
        
        node.flops_fw = 4 * num_elements
        node.flops_bw = 8 * num_elements
        
        node.memory_fw = 2 * num_elements * self.dtype_bytes
        node.memory_bw = 2 * num_elements * self.dtype_bytes
        
        node.weight_bytes = normalized_shape * self.dtype_bytes
    
    def _compute_attention(self, node: OpNode, attrs: Dict, metadata: Dict) -> None:
        """Compute workload for Attention."""
        num_heads = attrs.get("num_heads", metadata.get("num_heads", Symbol("num_heads")))
        head_dim = attrs.get("head_dim", metadata.get("head_dim", Symbol("head_dim")))
        seq_len = attrs.get("seq_len", metadata.get("seq_len", Symbol("seq_len")))
        batch_size = attrs.get("batch_size", metadata.get("batch_size", Symbol("batch_size")))
        
        qk_flops = 2 * batch_size * num_heads * seq_len * seq_len * head_dim
        softmax_flops = 5 * batch_size * num_heads * seq_len * seq_len
        sv_flops = 2 * batch_size * num_heads * seq_len * head_dim * seq_len
        
        node.flops_fw = qk_flops + softmax_flops + sv_flops
        node.flops_bw = 2 * node.flops_fw
        
        activation_size = batch_size * num_heads * seq_len * head_dim
        score_size = batch_size * num_heads * seq_len * seq_len
        
        # Match Calculon-style attention breakdown:
        # QK matmul + Softmax + Dropout(mask) + Attn matmul
        qk_mem = (2 * activation_size + score_size) * self.dtype_bytes
        softmax_mem = 2 * score_size * self.dtype_bytes
        dropout_mem = 2 * score_size * self.dtype_bytes + score_size  # mask = 1 byte/elem
        attn_mem = (2 * activation_size + score_size) * self.dtype_bytes
        
        node.memory_fw = qk_mem + softmax_mem + dropout_mem + attn_mem
        node.memory_bw = node.memory_fw
        
        # Store attention scores (softmax outputs) as main activation
        node.activation_bytes = score_size * self.dtype_bytes
    
    def _compute_silu(self, node: OpNode, attrs: Dict, batch_seq, ff) -> None:
        """Compute workload for SiLU activation."""
        num_elements = attrs.get("num_elements", batch_seq * ff)
        
        node.flops_fw = 4 * num_elements
        node.flops_bw = 6 * num_elements
        
        node.memory_fw = 2 * num_elements * self.dtype_bytes
        node.memory_bw = 2 * num_elements * self.dtype_bytes
        
        node.weight_bytes = 0
        node.activation_bytes = num_elements * self.dtype_bytes
    
    def _compute_gelu(self, node: OpNode, attrs: Dict, batch_seq, ff) -> None:
        """Compute workload for GELU activation."""
        num_elements = attrs.get("num_elements", batch_seq * ff)
        
        node.flops_fw = 8 * num_elements
        node.flops_bw = 10 * num_elements
        
        node.memory_fw = 2 * num_elements * self.dtype_bytes
        node.memory_bw = 2 * num_elements * self.dtype_bytes
        
        node.weight_bytes = 0
        node.activation_bytes = num_elements * self.dtype_bytes
    
    def _compute_add(self, node: OpNode, attrs: Dict, batch_seq, hidden) -> None:
        """Compute workload for element-wise Add."""
        num_elements = attrs.get("num_elements", batch_seq * hidden)
        
        node.flops_fw = num_elements
        node.flops_bw = 0
        
        node.memory_fw = 3 * num_elements * self.dtype_bytes
        node.memory_bw = 2 * num_elements * self.dtype_bytes
        
        node.weight_bytes = 0
    
    def _compute_mul(self, node: OpNode, attrs: Dict, batch_seq, hidden) -> None:
        """Compute workload for element-wise Mul."""
        num_elements = attrs.get("num_elements", batch_seq * hidden)
        
        node.flops_fw = num_elements
        node.flops_bw = 2 * num_elements
        
        node.memory_fw = 3 * num_elements * self.dtype_bytes
        node.memory_bw = 4 * num_elements * self.dtype_bytes
        
        node.weight_bytes = 0
        node.activation_bytes = 2 * num_elements * self.dtype_bytes
    
    def _compute_softmax(self, node: OpNode, attrs: Dict) -> None:
        """Compute workload for Softmax."""
        num_elements = attrs.get("num_elements", Symbol("num_elements"))
        
        node.flops_fw = 5 * num_elements
        node.flops_bw = 8 * num_elements
        
        node.memory_fw = 2 * num_elements * self.dtype_bytes
        node.memory_bw = 2 * num_elements * self.dtype_bytes
        
        node.weight_bytes = 0
        node.activation_bytes = num_elements * self.dtype_bytes
    
    def _compute_comm(self, node: OpNode, attrs: Dict) -> None:
        """Compute workload for communication operations."""
        num_peers = attrs.get("num_peers", attrs.get("tp", 1))
        data_size = attrs.get("data_size", Symbol("data_size"))
        comm_type = node.op_type
        
        node.flops_fw = 0
        node.flops_bw = 0
        node.memory_fw = 0
        node.memory_bw = 0
        node.weight_bytes = 0
        
        if comm_type == "AllReduce":
            node.comm_bytes_fw = 2 * (num_peers - 1) / num_peers * data_size * self.dtype_bytes
            node.comm_bytes_bw = node.comm_bytes_fw
        elif comm_type == "AllGather":
            node.comm_bytes_fw = (num_peers - 1) * data_size * self.dtype_bytes
            node.comm_bytes_bw = 0
        elif comm_type == "ReduceScatter":
            node.comm_bytes_fw = (num_peers - 1) / num_peers * data_size * self.dtype_bytes
            node.comm_bytes_bw = 0
        else:
            node.comm_bytes_fw = 0
            node.comm_bytes_bw = 0
        
        if node.comm_bytes_fw is not None:
            node.comm_bytes = node.comm_bytes_fw + (node.comm_bytes_bw or 0)

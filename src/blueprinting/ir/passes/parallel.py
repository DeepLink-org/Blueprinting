"""ParallelPass - Apply parallel strategies to hierarchical GraphIR.

This pass applies tensor parallelism (TP), pipeline parallelism (PP),
and data parallelism (DP) strategies to the graph.
"""

import math
import re
from typing import Dict, List, Optional

from sympy import Expr, Symbol

from ..graph import GraphIR, OpNode, BlockNode, ModuleNode
from .base import Pass


class ParallelPass(Pass):
    """Pass to apply parallel strategies to the graph.
    
    This pass:
    1. Marks each op with its sharding strategy
    2. Inserts communication ops into blocks
    3. Adjusts FLOPs/memory by parallel factors
    4. Assigns devices for pipeline stages
    """
    
    name = "ParallelPass"
    
    def __init__(
        self,
        tp: int = 1,
        pp: int = 1,
        dp: int = 1,
        tp_comm_type: str = "ar",
        sequence_parallel: bool = False,
    ):
        """Initialize ParallelPass.
        
        Args:
            tp: Tensor parallelism degree
            pp: Pipeline parallelism degree
            dp: Data parallelism degree
            tp_comm_type: TP communication type ("ar" or "rs_ag")
            sequence_parallel: Whether to use sequence parallelism
        """
        self.tp = tp
        self.pp = pp
        self.dp = dp
        self.tp_comm_type = tp_comm_type
        self.sequence_parallel = sequence_parallel
    
    def run(self, ir: GraphIR) -> GraphIR:
        """Execute the parallel pass."""
        result = ir.copy()
        
        result.symbols["TP"] = Symbol("TP")
        result.symbols["PP"] = Symbol("PP")
        result.symbols["DP"] = Symbol("DP")
        
        if self.tp > 1:
            self._apply_tensor_parallel(result)
        
        if self.pp > 1:
            self._apply_pipeline_parallel(result)
        
        result.metadata["tp"] = self.tp
        result.metadata["pp"] = self.pp
        result.metadata["dp"] = self.dp
        result.metadata["tp_comm_type"] = self.tp_comm_type
        result.metadata["sequence_parallel"] = self.sequence_parallel
        
        return result
    
    def _apply_tensor_parallel(self, ir: GraphIR) -> None:
        """Apply tensor parallelism to the graph."""
        if not ir.root:
            return
        
        self._apply_tp_to_children(ir.root.children)
    
    def _apply_tp_to_children(self, children: List) -> None:
        """Apply TP to a list of children (blocks/ops)."""
        for child in children:
            if isinstance(child, BlockNode):
                self._apply_tp_to_children(child.children)
                self._insert_block_comms(child)
            elif isinstance(child, OpNode):
                self._apply_tp_to_op(child)
    
    def _apply_tp_to_op(self, node: OpNode) -> None:
        """Apply TP to a single op."""
        shard = node.attrs.get("shard") or node.shard
        
        if node.op_type == "Linear" and shard:
            if shard == "tp_col":
                self._adjust_column_parallel(node)
            elif shard == "tp_row":
                self._adjust_row_parallel(node)
        
        elif node.op_type == "Attention":
            self._adjust_attention_parallel(node)
        
        elif node.op_type in ("SiLU", "GELU", "Mul"):
            if shard == "tp_col" or self._in_split_context(node):
                self._adjust_elementwise_parallel(node)
        
        elif node.op_type in ("RMSNorm", "LayerNorm", "Add") and self.sequence_parallel:
            self._adjust_sequence_parallel(node)
    
    def _in_split_context(self, node: OpNode) -> bool:
        """Check if op is in a split tensor context."""
        return False
    
    def _insert_block_comms(self, block: BlockNode) -> None:
        """Insert communication ops at block boundaries."""
        if block.block_type in ("Attention", "FFN"):
            # Insert pre-comm ops for RS/AG style TP (to match Calculon)
            if self.tp_comm_type in ("rs_ag", "p2p_rs_ag"):
                first_op = None
                for child in block.children:
                    if isinstance(child, OpNode) and child.op_type not in ("AllReduce", "AllGather", "ReduceScatter"):
                        first_op = child
                        break
                if first_op is not None:
                    ag_op = self._create_tp_comm_op("AllGather", first_op)
                    rs_op = self._create_tp_comm_op("ReduceScatter", first_op)
                    block.children.insert(0, rs_op)
                    block.children.insert(0, ag_op)

            idx = 0
            while idx < len(block.children):
                child = block.children[idx]
                if isinstance(child, OpNode) and child.op_type not in ("AllReduce", "AllGather", "ReduceScatter"):
                    if child.shard == "tp_row":
                        if self.tp_comm_type == "ar":
                            comm_op = self._create_allreduce_op(child, block)
                            block.children.insert(idx + 1, comm_op)
                            idx += 1
                        elif self.tp_comm_type in ("rs_ag", "p2p_rs_ag"):
                            rs_op = self._create_tp_comm_op("ReduceScatter", child)
                            ag_op = self._create_tp_comm_op("AllGather", child)
                            block.children.insert(idx + 1, rs_op)
                            block.children.insert(idx + 2, ag_op)
                            idx += 2
                idx += 1
    
    def _create_allreduce_op(self, source_op: OpNode, parent_block: BlockNode) -> OpNode:
        """Create an AllReduce communication op."""
        comm_name = f"{source_op.name}_allreduce"
        data_elems = self._infer_comm_elements(source_op)
        dtype_bytes = 2
        comm_bytes = data_elems * dtype_bytes if data_elems else 0
        
        comm_op = OpNode(
            name=comm_name,
            op_type="AllReduce",
            inputs=[f"{source_op.name}_out"],
            outputs=[f"{comm_name}_out"],
            attrs={"tp": self.tp, "data_size": data_elems},
        )
        
        comm_op.comm_bytes_fw = comm_bytes
        comm_op.comm_bytes_bw = comm_bytes
        comm_op.comm_bytes = comm_bytes
        # Model local buffer read/write similar to Calculon TPComm
        comm_op.memory_fw = (data_elems * dtype_bytes * 2) if data_elems else 0
        comm_op.memory_bw = comm_op.memory_fw
        comm_op.flops_fw = 0
        comm_op.flops_bw = 0
        comm_op.flops = 0
        comm_op.shard = "tp_comm"
        
        return comm_op

    def _create_tp_comm_op(self, op_type: str, source_op: OpNode) -> OpNode:
        """Create a TP communication op (AllGather/ReduceScatter)."""
        comm_name = f"{source_op.name}_{op_type.lower()}"
        data_elems = self._infer_comm_elements(source_op)
        dtype_bytes = 2
        comm_bytes = data_elems * dtype_bytes if data_elems else 0
        
        comm_op = OpNode(
            name=comm_name,
            op_type=op_type,
            inputs=[f"{source_op.name}_out"],
            outputs=[f"{comm_name}_out"],
            attrs={"tp": self.tp, "data_size": data_elems},
        )
        
        comm_op.comm_bytes_fw = comm_bytes
        comm_op.comm_bytes_bw = comm_bytes
        comm_op.comm_bytes = comm_bytes
        comm_op.memory_fw = (data_elems * dtype_bytes * 2) if data_elems else 0
        comm_op.memory_bw = comm_op.memory_fw
        comm_op.flops_fw = 0
        comm_op.flops_bw = 0
        comm_op.flops = 0
        comm_op.shard = "tp_comm"
        
        return comm_op

    def _infer_comm_elements(self, op: OpNode) -> Optional[float]:
        """Infer number of elements to communicate from the source op."""
        attrs = op.attrs
        
        if op.op_type == "Linear":
            batch_seq = attrs.get("batch_seq")
            out_features = attrs.get("out_features")
            in_features = attrs.get("in_features")
            if batch_seq is not None and out_features is not None:
                if op.shard == "tp_col" and in_features is not None:
                    return batch_seq * in_features
                return batch_seq * out_features
        
        if op.op_type in ("RMSNorm", "LayerNorm"):
            batch_seq = attrs.get("batch_seq")
            normalized_shape = attrs.get("normalized_shape")
            if batch_seq is not None and normalized_shape is not None:
                return batch_seq * normalized_shape
        
        if op.op_type == "Attention":
            batch_size = attrs.get("batch_size")
            num_heads = attrs.get("num_heads")
            seq_len = attrs.get("seq_len")
            head_dim = attrs.get("head_dim")
            if None not in (batch_size, num_heads, seq_len, head_dim):
                return batch_size * num_heads * seq_len * head_dim
        
        num_elements = attrs.get("num_elements")
        if num_elements is not None:
            return num_elements
        
        activation_bytes = getattr(op, "activation_bytes", None)
        if activation_bytes:
            return activation_bytes / 2
        
        return None
    
    def _adjust_column_parallel(self, node: OpNode) -> None:
        """Adjust node for column parallelism."""
        node.shard = "tp_col"
        
        for attr in ['flops_fw', 'flops_bw', 'flops_agrad', 'flops_wgrad', 'flops']:
            val = getattr(node, attr, None)
            if val is not None:
                setattr(node, attr, val / self.tp)
        
        if node.weight_bytes is not None:
            node.weight_bytes = node.weight_bytes / self.tp
        
        self._adjust_memory_column(node)
    
    def _adjust_row_parallel(self, node: OpNode) -> None:
        """Adjust node for row parallelism."""
        node.shard = "tp_row"
        
        for attr in ['flops_fw', 'flops_bw', 'flops_agrad', 'flops_wgrad', 'flops']:
            val = getattr(node, attr, None)
            if val is not None:
                setattr(node, attr, val / self.tp)
        
        if node.weight_bytes is not None:
            node.weight_bytes = node.weight_bytes / self.tp
        
        self._adjust_memory_row(node)
    
    def _adjust_memory_column(self, node: OpNode) -> None:
        """Adjust memory for column parallel."""
        in_f = node.attrs.get("in_features")
        out_f = node.attrs.get("out_features")
        batch_seq = node.attrs.get("batch_seq")
        
        if in_f is not None and out_f is not None and batch_seq is not None:
            dtype_bytes = 2
            mem_fw = (batch_seq * in_f + batch_seq * out_f / self.tp + 
                      in_f * out_f / self.tp) * dtype_bytes
            mem_bw = (batch_seq * in_f + batch_seq * out_f / self.tp + 
                      in_f * out_f / self.tp * 2) * dtype_bytes
            node.memory_fw = mem_fw
            node.memory_bw = mem_bw
        else:
            approx_ratio = (self.tp + 2) / (3 * self.tp)
            for attr in ['memory_fw', 'memory_bw', 'memory_agrad', 'memory_wgrad']:
                val = getattr(node, attr, None)
                if val is not None:
                    setattr(node, attr, val * approx_ratio)
    
    def _adjust_memory_row(self, node: OpNode) -> None:
        """Adjust memory for row parallel."""
        in_f = node.attrs.get("in_features")
        out_f = node.attrs.get("out_features")
        batch_seq = node.attrs.get("batch_seq")
        
        if in_f is not None and out_f is not None and batch_seq is not None:
            dtype_bytes = 2
            mem_fw = (batch_seq * in_f / self.tp + batch_seq * out_f + 
                      in_f / self.tp * out_f) * dtype_bytes
            mem_bw = (batch_seq * in_f / self.tp + batch_seq * out_f + 
                      in_f / self.tp * out_f * 2) * dtype_bytes
            node.memory_fw = mem_fw
            node.memory_bw = mem_bw
        else:
            approx_ratio = (self.tp + 2) / (3 * self.tp)
            for attr in ['memory_fw', 'memory_bw', 'memory_agrad', 'memory_wgrad']:
                val = getattr(node, attr, None)
                if val is not None:
                    setattr(node, attr, val * approx_ratio)
    
    def _adjust_attention_parallel(self, node: OpNode) -> None:
        """Adjust Attention for tensor parallelism."""
        node.shard = "tp_col"
        
        for attr in ['flops_fw', 'flops_bw', 'flops_agrad', 'flops_wgrad', 'flops']:
            val = getattr(node, attr, None)
            if val is not None:
                setattr(node, attr, val / self.tp)
        
        for attr in ['memory_fw', 'memory_bw']:
            val = getattr(node, attr, None)
            if val is not None:
                setattr(node, attr, val / self.tp)
    
    def _adjust_elementwise_parallel(self, node: OpNode) -> None:
        """Adjust element-wise ops for tensor parallelism."""
        node.shard = "tp_col"
        
        for attr in ['flops_fw', 'flops_bw']:
            val = getattr(node, attr, None)
            if val is not None:
                setattr(node, attr, val / self.tp)
        
        for attr in ['memory_fw', 'memory_bw']:
            val = getattr(node, attr, None)
            if val is not None:
                setattr(node, attr, val / self.tp)
        
        num_elem = node.attrs.get("num_elements")
        if num_elem is not None:
            node.attrs["num_elements"] = num_elem / self.tp
    
    def _adjust_sequence_parallel(self, node: OpNode) -> None:
        """Adjust for sequence parallelism."""
        node.shard = "seq_par"
        
        for attr in ['flops_fw', 'flops_bw']:
            val = getattr(node, attr, None)
            if val is not None:
                setattr(node, attr, val / self.tp)
    
    def _apply_pipeline_parallel(self, ir: GraphIR) -> None:
        """Apply pipeline parallelism to the graph."""
        if not ir.root:
            return
        
        num_layers = ir.metadata.get("num_layers", len(ir.root.children))
        if num_layers % self.pp != 0:
            raise ValueError(
                f"Pipeline parallelism requires num_layers to be a multiple of pp. "
                f"Got num_layers={num_layers}, pp={self.pp}."
            )
        layers_per_stage = max(1, math.ceil(num_layers / self.pp))
        last_stage = min((max(num_layers - 1, 0)) // layers_per_stage, self.pp - 1)
        extra_stage = min(last_stage + 1, self.pp - 1)
        
        for i, child in enumerate(ir.root.children):
            if isinstance(child, BlockNode):
                layer_idx = self._extract_layer_index(child.name)
                if layer_idx is None:
                    # Non-layer blocks (e.g., final output) can be placed on an extra stage
                    # if pipeline stages are available beyond the last layer stage.
                    child.device = extra_stage
                    self._set_device_recursive(child, extra_stage)
                    continue
                
                stage = min(layer_idx // layers_per_stage, self.pp - 1)
                child.device = stage
                
                self._set_device_recursive(child, stage)
    
    def _set_device_recursive(self, node, device: int) -> None:
        """Set device for all ops in a block."""
        if isinstance(node, OpNode):
            node.device = device
        elif isinstance(node, BlockNode):
            node.device = device
            for child in node.children:
                self._set_device_recursive(child, device)
    
    def _extract_layer_index(self, name: str) -> Optional[int]:
        """Extract layer index from block name."""
        match = re.search(r"layer(\d+)", name)
        if match:
            return int(match.group(1))
        
        match = re.search(r"block(\d+)", name)
        if match:
            return int(match.group(1))
        
        return None

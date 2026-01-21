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
            last_op = None
            for child in block.children:
                if isinstance(child, OpNode):
                    last_op = child
            
            if last_op and last_op.shard == "tp_row" and self.tp_comm_type == "ar":
                comm_op = self._create_allreduce_op(last_op.name, block)
                idx = block.children.index(last_op)
                block.children.insert(idx + 1, comm_op)
    
    def _create_allreduce_op(self, source_name: str, parent_block: BlockNode) -> OpNode:
        """Create an AllReduce communication op."""
        comm_name = f"{source_name}_allreduce"
        
        comm_op = OpNode(
            name=comm_name,
            op_type="AllReduce",
            inputs=[f"{source_name}_out"],
            outputs=[f"{comm_name}_out"],
            attrs={"tp": self.tp},
        )
        
        comm_op.comm_bytes_fw = 0
        comm_op.comm_bytes_bw = 0
        comm_op.comm_bytes = 0
        comm_op.flops_fw = 0
        comm_op.flops_bw = 0
        comm_op.flops = 0
        comm_op.shard = "tp_comm"
        
        return comm_op
    
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
        layers_per_stage = max(1, math.ceil(num_layers / self.pp))
        
        for i, child in enumerate(ir.root.children):
            if isinstance(child, BlockNode):
                layer_idx = self._extract_layer_index(child.name)
                if layer_idx is None:
                    layer_idx = i
                
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

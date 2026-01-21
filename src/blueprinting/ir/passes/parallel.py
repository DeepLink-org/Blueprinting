"""ParallelPass - Apply parallel strategies to GraphIR.

This pass applies tensor parallelism (TP), pipeline parallelism (PP),
and data parallelism (DP) strategies to the graph.
"""

from typing import Dict, List, Optional

from sympy import Expr, Symbol

from ..graph import GraphIR, OpNode
from .base import Pass


class ParallelPass(Pass):
    """Pass to apply parallel strategies to the graph.
    
    This pass:
    1. Marks each node with its sharding strategy
    2. Inserts communication nodes (AllReduce, AllGather, ReduceScatter)
    3. Adjusts FLOPs/memory by parallel factors
    4. Assigns devices for pipeline stages
    """
    
    def __init__(
        self,
        tp: int = 1,
        pp: int = 1,
        dp: int = 1,
        tp_comm_type: str = "ar",  # "ar" (all-reduce) or "rs_ag" (reduce-scatter + all-gather)
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
        # Create a copy to avoid modifying the original
        result = ir.copy()
        
        # Add parallel symbols
        result.symbols["TP"] = Symbol("TP")
        result.symbols["PP"] = Symbol("PP")
        result.symbols["DP"] = Symbol("DP")
        
        # Apply tensor parallelism
        if self.tp > 1:
            self._apply_tensor_parallel(result)
        
        # Apply pipeline parallelism
        if self.pp > 1:
            self._apply_pipeline_parallel(result)
        
        # Store parallel config in metadata
        result.metadata["tp"] = self.tp
        result.metadata["pp"] = self.pp
        result.metadata["dp"] = self.dp
        result.metadata["tp_comm_type"] = self.tp_comm_type
        result.metadata["sequence_parallel"] = self.sequence_parallel
        
        return result
    
    def _apply_tensor_parallel(self, ir: GraphIR) -> None:
        """Apply tensor parallelism to the graph.
        
        For TP:
        - Column parallel Linear (e.g., QKV, gate/up): split output dim
        - Row parallel Linear (e.g., out proj, down): split input dim
        - Attention: split across heads
        - Element-wise ops (SiLU, Mul) after column parallel: operate on split data
        - Communication inserted between row and column parallel
        """
        nodes_to_add = []
        edges_to_add = []
        
        for node_id, node in list(ir.nodes.items()):
            shard = node.attrs.get("shard")
            
            if node.op_type == "Linear" and shard:
                if shard == "tp_col":
                    # Column parallel: output is split
                    self._adjust_column_parallel(node)
                elif shard == "tp_row":
                    # Row parallel: input is split, need AllReduce after
                    self._adjust_row_parallel(node)
                    
                    # Insert communication
                    if self.tp_comm_type == "ar":
                        comm_node = self._create_allreduce_node(node, ir)
                        if comm_node:
                            nodes_to_add.append(comm_node)
                            edges_to_add.append((node_id, comm_node.id))
                            
                            # Update successors to use comm output
                            for src, dst in list(ir.edges):
                                if src == node_id:
                                    ir.edges.remove((src, dst))
                                    edges_to_add.append((comm_node.id, dst))
                    elif self.tp_comm_type == "rs_ag":
                        rs_node = self._create_reducescatter_node(node, ir)
                        ag_node = self._create_allgather_node(node, ir)
                        if rs_node and ag_node:
                            nodes_to_add.extend([rs_node, ag_node])
                            edges_to_add.append((node_id, rs_node.id))
                            edges_to_add.append((rs_node.id, ag_node.id))
                            
                            # Update successors to use AllGather output
                            for src, dst in list(ir.edges):
                                if src == node_id:
                                    ir.edges.remove((src, dst))
                                    edges_to_add.append((ag_node.id, dst))
            
            elif node.op_type == "Attention" and shard == "tp_col":
                # Attention is split across heads
                self._adjust_attention_parallel(node)
            
            elif node.op_type in ("SiLU", "Mul", "Add") and shard == "tp_col":
                # Element-wise ops on split tensor
                self._adjust_elementwise_parallel(node)
            
            elif node.op_type in ("RMSNorm", "Add") and self.sequence_parallel:
                # Sequence parallel: these ops operate on split sequence
                self._adjust_sequence_parallel(node)
        
        # Add new nodes and edges
        for node in nodes_to_add:
            ir.nodes[node.id] = node
        ir.edges.extend(edges_to_add)
    
    def _adjust_column_parallel(self, node: OpNode) -> None:
        """Adjust node for column parallelism (split output dim).
        
        For column parallel Linear [in, out] -> [in, out/tp]:
        - FLOPs: divided by tp (proportional to out dim)
        - Weight: in * out/tp (divided by tp)
        - Memory: input (B*S*in) + output (B*S*out/tp) + weight (in*out/tp)
                  Not simply divided by tp because input is not split
        """
        node.shard = "tp_col"
        
        # FLOPs are proportional to output dim, so divide by tp
        if node.flops_fw is not None:
            node.flops_fw = node.flops_fw / self.tp
        if node.flops_bw is not None:
            node.flops_bw = node.flops_bw / self.tp
        if node.flops_agrad is not None:
            node.flops_agrad = node.flops_agrad / self.tp
        if node.flops_wgrad is not None:
            node.flops_wgrad = node.flops_wgrad / self.tp
        if node.flops is not None:
            node.flops = node.flops / self.tp
        
        # Weight is split by tp
        if node.weight_bytes is not None:
            node.weight_bytes = node.weight_bytes / self.tp
        
        # Memory: don't simply divide by tp
        # Original: (batch_seq * in + batch_seq * out + in * out) * dtype
        # After col split: (batch_seq * in + batch_seq * out/tp + in * out/tp) * dtype
        # Approximation: (in + out/tp + in*out/tp) / (in + out + in*out) * original
        # Since output and weight are split, but input is not, the ratio is ~1/tp for 
        # output-dominated terms but ~1 for input term.
        # For Linear with in≈out, memory ratio ≈ (1 + 1/tp + 1/tp) / 3 ≈ (tp + 2) / (3*tp)
        # Use a more accurate formula based on attrs
        in_f = node.attrs.get("in_features")
        out_f = node.attrs.get("out_features")
        batch_seq = node.attrs.get("batch_seq")
        
        if in_f is not None and out_f is not None and batch_seq is not None:
            dtype_bytes = 2  # Assume FP16
            # Calculate actual memory after TP split
            mem_fw = (batch_seq * in_f + batch_seq * out_f / self.tp + in_f * out_f / self.tp) * dtype_bytes
            mem_bw = (batch_seq * in_f + batch_seq * out_f / self.tp + in_f * out_f / self.tp * 2) * dtype_bytes
            node.memory_fw = mem_fw
            node.memory_bw = mem_bw
            if node.memory_agrad is not None:
                node.memory_agrad = (batch_seq * out_f / self.tp + in_f * out_f / self.tp + batch_seq * in_f) * dtype_bytes
            if node.memory_wgrad is not None:
                node.memory_wgrad = (batch_seq * in_f + batch_seq * out_f / self.tp + in_f * out_f / self.tp) * dtype_bytes
        else:
            # Fallback: use approximation (tp + 2) / (3 * tp) ≈ 0.4 for tp=8
            approx_ratio = (self.tp + 2) / (3 * self.tp)
            if node.memory_fw is not None:
                node.memory_fw = node.memory_fw * approx_ratio
            if node.memory_bw is not None:
                node.memory_bw = node.memory_bw * approx_ratio
            if node.memory_agrad is not None:
                node.memory_agrad = node.memory_agrad * approx_ratio
            if node.memory_wgrad is not None:
                node.memory_wgrad = node.memory_wgrad * approx_ratio
    
    def _adjust_row_parallel(self, node: OpNode) -> None:
        """Adjust node for row parallelism (split input dim).
        
        For row parallel Linear [in, out] -> [in/tp, out]:
        - FLOPs: divided by tp (proportional to in dim)
        - Weight: in/tp * out (divided by tp)
        - Memory: input (B*S*in/tp) + output (B*S*out) + weight (in/tp*out)
                  Not simply divided by tp because output is not split
        """
        node.shard = "tp_row"
        
        # FLOPs are proportional to input dim, so divide by tp
        if node.flops_fw is not None:
            node.flops_fw = node.flops_fw / self.tp
        if node.flops_bw is not None:
            node.flops_bw = node.flops_bw / self.tp
        if node.flops_agrad is not None:
            node.flops_agrad = node.flops_agrad / self.tp
        if node.flops_wgrad is not None:
            node.flops_wgrad = node.flops_wgrad / self.tp
        if node.flops is not None:
            node.flops = node.flops / self.tp
        
        # Weight is split by tp
        if node.weight_bytes is not None:
            node.weight_bytes = node.weight_bytes / self.tp
        
        # Memory: don't simply divide by tp
        # Original: (batch_seq * in + batch_seq * out + in * out) * dtype
        # After row split: (batch_seq * in/tp + batch_seq * out + in/tp * out) * dtype
        in_f = node.attrs.get("in_features")
        out_f = node.attrs.get("out_features")
        batch_seq = node.attrs.get("batch_seq")
        
        if in_f is not None and out_f is not None and batch_seq is not None:
            dtype_bytes = 2  # Assume FP16
            # Calculate actual memory after TP split
            mem_fw = (batch_seq * in_f / self.tp + batch_seq * out_f + in_f / self.tp * out_f) * dtype_bytes
            mem_bw = (batch_seq * in_f / self.tp + batch_seq * out_f + in_f / self.tp * out_f * 2) * dtype_bytes
            node.memory_fw = mem_fw
            node.memory_bw = mem_bw
            if node.memory_agrad is not None:
                node.memory_agrad = (batch_seq * out_f + in_f / self.tp * out_f + batch_seq * in_f / self.tp) * dtype_bytes
            if node.memory_wgrad is not None:
                node.memory_wgrad = (batch_seq * in_f / self.tp + batch_seq * out_f + in_f / self.tp * out_f) * dtype_bytes
        else:
            # Fallback: use approximation
            approx_ratio = (self.tp + 2) / (3 * self.tp)
            if node.memory_fw is not None:
                node.memory_fw = node.memory_fw * approx_ratio
            if node.memory_bw is not None:
                node.memory_bw = node.memory_bw * approx_ratio
            if node.memory_agrad is not None:
                node.memory_agrad = node.memory_agrad * approx_ratio
            if node.memory_wgrad is not None:
                node.memory_wgrad = node.memory_wgrad * approx_ratio
    
    def _adjust_sequence_parallel(self, node: OpNode) -> None:
        """Adjust node for sequence parallelism."""
        node.shard = "seq_par"
        
        # Divide by TP for sequence dimension
        if node.flops_fw is not None:
            node.flops_fw = node.flops_fw / self.tp
        if node.flops_bw is not None:
            node.flops_bw = node.flops_bw / self.tp
    
    def _adjust_attention_parallel(self, node: OpNode) -> None:
        """Adjust Attention node for tensor parallelism.
        
        Attention is split across heads, so FLOPs and memory are divided by TP.
        """
        node.shard = "tp_col"
        
        # FLOPs are proportional to num_heads, divide by tp
        if node.flops_fw is not None:
            node.flops_fw = node.flops_fw / self.tp
        if node.flops_bw is not None:
            node.flops_bw = node.flops_bw / self.tp
        if node.flops_agrad is not None:
            node.flops_agrad = node.flops_agrad / self.tp
        if node.flops_wgrad is not None:
            node.flops_wgrad = node.flops_wgrad / self.tp
        if node.flops is not None:
            node.flops = node.flops / self.tp
        
        # Memory is also divided (Q, K, V, Scores are all split)
        if node.memory_fw is not None:
            node.memory_fw = node.memory_fw / self.tp
        if node.memory_bw is not None:
            node.memory_bw = node.memory_bw / self.tp
    
    def _adjust_elementwise_parallel(self, node: OpNode) -> None:
        """Adjust element-wise ops (SiLU, Mul) for tensor parallelism.
        
        These ops operate on split tensors from column parallel layers.
        """
        node.shard = "tp_col"
        
        # FLOPs and memory are proportional to tensor size, divide by tp
        if node.flops_fw is not None:
            node.flops_fw = node.flops_fw / self.tp
        if node.flops_bw is not None:
            node.flops_bw = node.flops_bw / self.tp
        if node.memory_fw is not None:
            node.memory_fw = node.memory_fw / self.tp
        if node.memory_bw is not None:
            node.memory_bw = node.memory_bw / self.tp
        
        # Update num_elements attribute if present
        num_elem = node.attrs.get("num_elements")
        if num_elem is not None:
            node.attrs["num_elements"] = num_elem / self.tp
    
    def _create_allreduce_node(self, source_node: OpNode, ir: GraphIR) -> Optional[OpNode]:
        """Create an AllReduce communication node."""
        comm_id = f"{source_node.id}_allreduce"
        
        # Get data size from source node's output
        data_size = source_node.attrs.get("out_features", Symbol("out_features"))
        batch_seq = source_node.attrs.get("batch_seq", Symbol("batch_seq"))
        
        comm_node = OpNode(
            id=comm_id,
            op_type="AllReduce",
            inputs=[source_node.id],
            outputs=[f"{comm_id}_out"],
            attrs={
                "tp": self.tp,
                "data_size": batch_seq * data_size,
            },
        )
        
        # Set communication bytes
        dtype_bytes = 2  # Assume float16
        comm_node.comm_bytes_fw = 2 * (self.tp - 1) / self.tp * batch_seq * data_size * dtype_bytes
        comm_node.comm_bytes_bw = comm_node.comm_bytes_fw
        comm_node.comm_bytes = comm_node.comm_bytes_fw + comm_node.comm_bytes_bw
        comm_node.flops_fw = 0
        comm_node.flops_bw = 0
        comm_node.flops = 0
        
        return comm_node
    
    def _create_reducescatter_node(self, source_node: OpNode, ir: GraphIR) -> Optional[OpNode]:
        """Create a ReduceScatter communication node."""
        comm_id = f"{source_node.id}_reducescatter"
        
        data_size = source_node.attrs.get("out_features", Symbol("out_features"))
        batch_seq = source_node.attrs.get("batch_seq", Symbol("batch_seq"))
        
        comm_node = OpNode(
            id=comm_id,
            op_type="ReduceScatter",
            inputs=[source_node.id],
            outputs=[f"{comm_id}_out"],
            attrs={
                "tp": self.tp,
                "data_size": batch_seq * data_size,
            },
        )
        
        dtype_bytes = 2  # Assume float16
        comm_bytes = (self.tp - 1) / self.tp * batch_seq * data_size * dtype_bytes
        comm_node.comm_bytes_fw = comm_bytes
        comm_node.comm_bytes_bw = comm_bytes
        comm_node.comm_bytes = comm_node.comm_bytes_fw + comm_node.comm_bytes_bw
        comm_node.flops_fw = 0
        comm_node.flops_bw = 0
        comm_node.flops = 0
        
        return comm_node
    
    def _create_allgather_node(self, source_node: OpNode, ir: GraphIR) -> Optional[OpNode]:
        """Create an AllGather communication node."""
        comm_id = f"{source_node.id}_allgather"
        
        data_size = source_node.attrs.get("out_features", Symbol("out_features"))
        batch_seq = source_node.attrs.get("batch_seq", Symbol("batch_seq"))
        
        comm_node = OpNode(
            id=comm_id,
            op_type="AllGather",
            inputs=[source_node.id],
            outputs=[f"{comm_id}_out"],
            attrs={
                "tp": self.tp,
                "data_size": batch_seq * data_size,
            },
        )
        
        dtype_bytes = 2  # Assume float16
        comm_bytes = (self.tp - 1) / self.tp * batch_seq * data_size * dtype_bytes
        comm_node.comm_bytes_fw = comm_bytes
        comm_node.comm_bytes_bw = comm_bytes
        comm_node.comm_bytes = comm_node.comm_bytes_fw + comm_node.comm_bytes_bw
        comm_node.flops_fw = 0
        comm_node.flops_bw = 0
        comm_node.flops = 0
        
        return comm_node
    
    def _apply_pipeline_parallel(self, ir: GraphIR) -> None:
        """Apply pipeline parallelism to the graph.
        
        Assigns each node to a pipeline stage based on layer index.
        """
        import math
        num_layers = ir.metadata.get("num_layers", 1)
        layers_per_stage = max(1, math.ceil(num_layers / self.pp))
        
        for node_id, node in ir.nodes.items():
            # Extract layer index from node ID (e.g., "layer5_q_proj" -> 5)
            layer_idx = self._extract_layer_index(node_id)
            if layer_idx is not None:
                stage = min(layer_idx // layers_per_stage, self.pp - 1)
                node.device = stage
            else:
                # Default to stage 0
                node.device = 0
    
    def _extract_layer_index(self, node_id: str) -> Optional[int]:
        """Extract layer index from node ID."""
        import re
        match = re.search(r"layer(\d+)", node_id)
        if match:
            return int(match.group(1))
        
        match = re.search(r"block(\d+)", node_id)
        if match:
            return int(match.group(1))
        
        return None

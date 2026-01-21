"""IRBuilder - Build GraphIR from LayerDef operations.

This module provides utilities for constructing GraphIR
from the existing LayerDef-based model definitions.
"""

from typing import Any, Dict, List, Optional, Union

from sympy import Expr, Symbol

from .graph import GraphIR, OpNode, TensorRef


class IRBuilder:
    """Builder for constructing GraphIR from operations.
    
    Example usage:
        builder = IRBuilder()
        
        # Add symbols
        builder.add_symbol("B", "batch_size")
        builder.add_symbol("S", "seq_len") 
        builder.add_symbol("H", "hidden")
        
        # Build a simple linear layer
        x = builder.add_input("x", shape=[B, S, H])
        linear = builder.add_op("linear_1", "Linear", inputs=[x], 
                               attrs={"in_features": H, "out_features": H})
        
        # Get the graph
        graph = builder.build()
    """
    
    def __init__(self):
        self._nodes: Dict[str, OpNode] = {}
        self._tensors: Dict[str, TensorRef] = {}
        self._edges: List[tuple] = []
        self._symbols: Dict[str, Symbol] = {}
        self._metadata: Dict[str, Any] = {}
        self._counter = 0
        self._input_ids: List[str] = []
        self._output_ids: List[str] = []
    
    def add_symbol(self, name: str, description: str = "") -> Symbol:
        """Add a symbolic variable.
        
        Args:
            name: Symbol name
            description: Optional description
            
        Returns:
            SymPy Symbol
        """
        if name not in self._symbols:
            self._symbols[name] = Symbol(name)
        return self._symbols[name]
    
    def get_symbol(self, name: str) -> Symbol:
        """Get or create a symbol."""
        return self.add_symbol(name)
    
    def add_input(
        self,
        name: str,
        shape: Optional[List[Union[int, Expr]]] = None,
        dtype: str = "float16",
    ) -> str:
        """Add an input tensor to the graph.
        
        Args:
            name: Tensor name
            shape: Tensor shape (can include symbols)
            dtype: Data type
            
        Returns:
            Tensor ID
        """
        tensor_id = f"input_{name}"
        self._tensors[tensor_id] = TensorRef(
            id=tensor_id,
            shape=shape,
            dtype=dtype,
            producer=None,
        )
        self._input_ids.append(tensor_id)
        return tensor_id
    
    def add_op(
        self,
        name: str,
        op_type: str,
        inputs: Optional[List[str]] = None,
        attrs: Optional[Dict[str, Any]] = None,
        output_shape: Optional[List[Union[int, Expr]]] = None,
        output_dtype: str = "float16",
    ) -> str:
        """Add an operation node.
        
        Args:
            name: Operation name (used as ID)
            op_type: Type of operation
            inputs: List of input tensor/node IDs
            attrs: Operation attributes
            output_shape: Shape of output tensor
            output_dtype: Data type of output
            
        Returns:
            Node ID
        """
        node_id = name
        inputs = inputs or []
        attrs = attrs or {}
        
        # Create output tensor
        output_id = f"{node_id}_out"
        self._tensors[output_id] = TensorRef(
            id=output_id,
            shape=output_shape,
            dtype=output_dtype,
            producer=node_id,
        )
        
        # Create node
        node = OpNode(
            id=node_id,
            op_type=op_type,
            inputs=inputs,
            outputs=[output_id],
            attrs=attrs,
        )
        self._nodes[node_id] = node
        
        # Add edges from inputs to this node
        for inp in inputs:
            # Find the producer node of this input
            if inp in self._tensors:
                producer = self._tensors[inp].producer
                if producer:
                    self._edges.append((producer, node_id))
            elif inp in self._nodes:
                self._edges.append((inp, node_id))
        
        return node_id
    
    def add_linear(
        self,
        name: str,
        inputs: List[str],
        in_features: Union[int, Expr],
        out_features: Union[int, Expr],
        bias: bool = True,
        shard: Optional[str] = None,
    ) -> str:
        """Add a Linear layer.
        
        Args:
            name: Layer name
            inputs: Input tensor IDs
            in_features: Input feature dimension
            out_features: Output feature dimension
            bias: Whether to use bias
            shard: Sharding strategy ("tp_col", "tp_row", None)
            
        Returns:
            Node ID
        """
        return self.add_op(
            name=name,
            op_type="Linear",
            inputs=inputs,
            attrs={
                "in_features": in_features,
                "out_features": out_features,
                "bias": bias,
                "shard": shard,
            },
        )
    
    def add_rmsnorm(
        self,
        name: str,
        inputs: List[str],
        normalized_shape: Union[int, Expr],
    ) -> str:
        """Add an RMSNorm layer."""
        return self.add_op(
            name=name,
            op_type="RMSNorm",
            inputs=inputs,
            attrs={"normalized_shape": normalized_shape},
        )
    
    def add_attention(
        self,
        name: str,
        inputs: List[str],
        num_heads: Union[int, Expr],
        head_dim: Union[int, Expr],
        seq_len: Union[int, Expr],
    ) -> str:
        """Add an Attention block (Q@K^T, softmax, @V)."""
        return self.add_op(
            name=name,
            op_type="Attention",
            inputs=inputs,
            attrs={
                "num_heads": num_heads,
                "head_dim": head_dim,
                "seq_len": seq_len,
            },
        )
    
    def add_elementwise(
        self,
        name: str,
        op_type: str,  # "Add", "Mul", "SiLU", etc.
        inputs: List[str],
    ) -> str:
        """Add an elementwise operation."""
        return self.add_op(
            name=name,
            op_type=op_type,
            inputs=inputs,
        )
    
    def add_comm(
        self,
        name: str,
        comm_type: str,  # "AllReduce", "AllGather", "ReduceScatter"
        inputs: List[str],
        num_peers: int = 1,
    ) -> str:
        """Add a communication operation."""
        return self.add_op(
            name=name,
            op_type=comm_type,
            inputs=inputs,
            attrs={"num_peers": num_peers},
        )
    
    def mark_output(self, node_id: str) -> None:
        """Mark a node as graph output."""
        if node_id in self._nodes:
            output_tensor = self._nodes[node_id].outputs[0] if self._nodes[node_id].outputs else node_id
            self._output_ids.append(output_tensor)
    
    def set_metadata(self, key: str, value: Any) -> "IRBuilder":
        """Set graph metadata."""
        self._metadata[key] = value
        return self
    
    def build(self) -> GraphIR:
        """Build and return the GraphIR.
        
        Returns:
            Constructed GraphIR
        """
        graph = GraphIR(
            nodes=self._nodes.copy(),
            tensors=self._tensors.copy(),
            edges=self._edges.copy(),
            symbols=self._symbols.copy(),
            metadata=self._metadata.copy(),
        )
        graph.metadata["inputs"] = self._input_ids.copy()
        graph.metadata["outputs"] = self._output_ids.copy()
        return graph
    
    def reset(self) -> "IRBuilder":
        """Reset the builder state."""
        self._nodes.clear()
        self._tensors.clear()
        self._edges.clear()
        self._symbols.clear()
        self._metadata.clear()
        self._counter = 0
        self._input_ids.clear()
        self._output_ids.clear()
        return self


def build_transformer_block(
    hidden: Union[int, Symbol],
    feedforward: Union[int, Symbol],
    num_heads: Union[int, Symbol],
    head_dim: Union[int, Symbol],
    seq_len: Union[int, Symbol],
    batch_size: Union[int, Symbol],
    block_id: int = 0,
    tp: int = 1,
) -> GraphIR:
    """Build a GraphIR for a single Transformer block.
    
    This creates a standard LLaMA-style block with:
    - RMSNorm -> Attention (Q, K, V projections + MHA) -> Add (residual)
    - RMSNorm -> FFN (gate, up, down with SwiGLU) -> Add (residual)
    
    Args:
        hidden: Hidden dimension
        feedforward: FFN intermediate dimension
        num_heads: Number of attention heads
        head_dim: Dimension per head
        seq_len: Sequence length
        batch_size: Batch size
        block_id: Block identifier
        tp: Tensor parallelism degree
        
    Returns:
        GraphIR representing the block
    """
    builder = IRBuilder()
    prefix = f"block{block_id}_"
    
    # Add symbols if they aren't already
    if isinstance(hidden, Symbol):
        builder._symbols["hidden"] = hidden
    if isinstance(feedforward, Symbol):
        builder._symbols["feedforward"] = feedforward
    if isinstance(seq_len, Symbol):
        builder._symbols["seq_len"] = seq_len
    if isinstance(batch_size, Symbol):
        builder._symbols["batch_size"] = batch_size
    
    # Input
    x = builder.add_input("x", shape=[batch_size, seq_len, hidden])
    
    # === Attention Block ===
    # RMSNorm
    attn_norm = builder.add_rmsnorm(f"{prefix}attn_norm", [x], hidden)
    
    # Q, K, V projections (Column Parallel in TP)
    qkv_out = hidden if tp == 1 else hidden // tp
    q_proj = builder.add_linear(f"{prefix}q_proj", [attn_norm], hidden, hidden, shard="tp_col" if tp > 1 else None)
    k_proj = builder.add_linear(f"{prefix}k_proj", [attn_norm], hidden, hidden, shard="tp_col" if tp > 1 else None)
    v_proj = builder.add_linear(f"{prefix}v_proj", [attn_norm], hidden, hidden, shard="tp_col" if tp > 1 else None)
    
    # Attention computation
    attn = builder.add_attention(f"{prefix}attention", [q_proj, k_proj, v_proj], num_heads, head_dim, seq_len)
    
    # Output projection (Row Parallel in TP)
    attn_out = builder.add_linear(f"{prefix}attn_out", [attn], hidden, hidden, shard="tp_row" if tp > 1 else None)
    
    # Residual connection
    attn_add = builder.add_elementwise(f"{prefix}attn_add", "Add", [x, attn_out])
    
    # === FFN Block ===
    # RMSNorm
    ffn_norm = builder.add_rmsnorm(f"{prefix}ffn_norm", [attn_add], hidden)
    
    # Gate and Up projections (Column Parallel)
    gate_proj = builder.add_linear(f"{prefix}gate_proj", [ffn_norm], hidden, feedforward, shard="tp_col" if tp > 1 else None)
    up_proj = builder.add_linear(f"{prefix}up_proj", [ffn_norm], hidden, feedforward, shard="tp_col" if tp > 1 else None)
    
    # SiLU activation on gate
    gate_silu = builder.add_elementwise(f"{prefix}gate_silu", "SiLU", [gate_proj])
    
    # Element-wise multiply (SwiGLU)
    swiglu = builder.add_elementwise(f"{prefix}swiglu_mul", "Mul", [gate_silu, up_proj])
    
    # Down projection (Row Parallel)
    down_proj = builder.add_linear(f"{prefix}down_proj", [swiglu], feedforward, hidden, shard="tp_row" if tp > 1 else None)
    
    # Residual connection
    ffn_add = builder.add_elementwise(f"{prefix}ffn_add", "Add", [attn_add, down_proj])
    
    builder.mark_output(ffn_add)
    builder.set_metadata("block_id", block_id)
    builder.set_metadata("tp", tp)
    
    return builder.build()


def build_transformer_model(
    num_layers: int,
    hidden: Union[int, Symbol],
    feedforward: Union[int, Symbol],
    num_heads: Union[int, Symbol],
    head_dim: Union[int, Symbol],
    seq_len: Union[int, Symbol],
    batch_size: Union[int, Symbol],
    tp: int = 1,
    pp: int = 1,
) -> GraphIR:
    """Build a GraphIR for a full Transformer model.
    
    Args:
        num_layers: Number of transformer layers
        hidden: Hidden dimension
        feedforward: FFN intermediate dimension
        num_heads: Number of attention heads
        head_dim: Dimension per head
        seq_len: Sequence length
        batch_size: Batch size
        tp: Tensor parallelism degree
        pp: Pipeline parallelism degree
        
    Returns:
        GraphIR representing the full model
    """
    builder = IRBuilder()
    
    # Setup symbols
    B = batch_size if isinstance(batch_size, Symbol) else builder.add_symbol("B")
    S = seq_len if isinstance(seq_len, Symbol) else builder.add_symbol("S")
    H = hidden if isinstance(hidden, Symbol) else builder.add_symbol("H")
    FF = feedforward if isinstance(feedforward, Symbol) else builder.add_symbol("FF")
    
    # Input embedding (simplified - just the input tensor)
    x = builder.add_input("input", shape=[B, S, H])
    
    # Layers per pipeline stage
    layers_per_stage = num_layers // pp
    
    prev_output = x
    for layer_idx in range(num_layers):
        stage = layer_idx // layers_per_stage if pp > 1 else 0
        prefix = f"layer{layer_idx}_"
        
        # Attention block
        attn_norm = builder.add_rmsnorm(f"{prefix}attn_norm", [prev_output], H)
        q_proj = builder.add_linear(f"{prefix}q_proj", [attn_norm], H, H, shard="tp_col" if tp > 1 else None)
        k_proj = builder.add_linear(f"{prefix}k_proj", [attn_norm], H, H, shard="tp_col" if tp > 1 else None)  
        v_proj = builder.add_linear(f"{prefix}v_proj", [attn_norm], H, H, shard="tp_col" if tp > 1 else None)
        attn = builder.add_attention(f"{prefix}attn", [q_proj, k_proj, v_proj], num_heads, head_dim, S)
        attn_out = builder.add_linear(f"{prefix}attn_out", [attn], H, H, shard="tp_row" if tp > 1 else None)
        
        # Add communication if TP > 1
        if tp > 1:
            attn_out = builder.add_comm(f"{prefix}attn_allreduce", "AllReduce", [attn_out], tp)
        
        attn_add = builder.add_elementwise(f"{prefix}attn_add", "Add", [prev_output, attn_out])
        
        # FFN block
        ffn_norm = builder.add_rmsnorm(f"{prefix}ffn_norm", [attn_add], H)
        gate_proj = builder.add_linear(f"{prefix}gate_proj", [ffn_norm], H, FF, shard="tp_col" if tp > 1 else None)
        up_proj = builder.add_linear(f"{prefix}up_proj", [ffn_norm], H, FF, shard="tp_col" if tp > 1 else None)
        gate_silu = builder.add_elementwise(f"{prefix}gate_silu", "SiLU", [gate_proj])
        swiglu = builder.add_elementwise(f"{prefix}swiglu_mul", "Mul", [gate_silu, up_proj])
        down_proj = builder.add_linear(f"{prefix}down_proj", [swiglu], FF, H, shard="tp_row" if tp > 1 else None)
        
        # Add communication if TP > 1
        if tp > 1:
            down_proj = builder.add_comm(f"{prefix}ffn_allreduce", "AllReduce", [down_proj], tp)
        
        ffn_add = builder.add_elementwise(f"{prefix}ffn_add", "Add", [attn_add, down_proj])
        
        # Set stage for PP
        builder._nodes[f"{prefix}attn_norm"].device = stage
        # ... (set device for all nodes in this layer)
        
        prev_output = ffn_add
    
    builder.mark_output(prev_output)
    builder.set_metadata("num_layers", num_layers)
    builder.set_metadata("tp", tp)
    builder.set_metadata("pp", pp)
    
    return builder.build()

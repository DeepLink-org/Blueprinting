"""IRBuilder - Build hierarchical GraphIR for neural network models.

This module provides a fluent API for constructing hierarchical computation graphs
with Module → Block → Op structure.
"""

from contextlib import contextmanager
from typing import Any, Dict, List, Union

from sympy import Expr, Symbol

from .graph import BlockNode, GraphIR, ModuleNode, OpNode, TensorRef


class IRBuilder:
    """Builder for constructing hierarchical GraphIR.

    Example usage:
        builder = IRBuilder("gpt2", "Transformer")

        # Model inputs
        builder.add_input("x", shape=[B, S, H])

        # Add a transformer layer
        with builder.block("layer0", "TransformerLayer"):
            with builder.block("attn", "Attention"):
                builder.linear("q_proj", "x", H, H, shard="tp_col")
                builder.linear("k_proj", "x", H, H, shard="tp_col")
                builder.linear("v_proj", "x", H, H, shard="tp_col")
                builder.attention("mha", ["q_proj_out", "k_proj_out", "v_proj_out"])
                builder.linear("out_proj", "mha_out", H, H, shard="tp_row")

            with builder.block("ffn", "FFN"):
                builder.linear("fc1", "attn_out", H, FF)
                builder.activation("gelu", "fc1_out", "GELU")
                builder.linear("fc2", "gelu_out", FF, H)

        graph = builder.build()
    """

    def __init__(self, model_name: str = "model", model_type: str = "Model"):
        """Initialize the builder.

        Args:
            model_name: Name of the model
            model_type: Type of model (e.g., "Transformer", "GPT")
        """
        self._root = ModuleNode(name=model_name, module_type=model_type)
        self._tensors: Dict[str, TensorRef] = {}
        self._symbols: Dict[str, Symbol] = {}
        self._metadata: Dict[str, Any] = {}

        # Stack of current containers for nested block building
        self._block_stack: List[Union[ModuleNode, BlockNode]] = [self._root]

        # Counter for unique names
        self._counter = 0

    # === Symbol management ===

    def add_symbol(self, name: str) -> Symbol:
        """Add or get a symbolic variable."""
        if name not in self._symbols:
            self._symbols[name] = Symbol(name)
        return self._symbols[name]

    def get_symbol(self, name: str) -> Symbol:
        """Get or create a symbol."""
        return self.add_symbol(name)

    # === Tensor management ===

    def add_input(self, name: str, shape: List = None, dtype: str = "float16") -> str:
        """Add a model input tensor.

        Args:
            name: Input tensor name
            shape: Tensor shape
            dtype: Data type

        Returns:
            Tensor name
        """
        tensor = TensorRef(name=name, shape=shape, dtype=dtype, producer=None)
        self._tensors[name] = tensor
        self._root.inputs.append(name)
        return name

    def add_output(self, name: str) -> None:
        """Mark a tensor as model output."""
        self._root.outputs.append(name)

    def _register_output_tensor(
        self,
        name: str,
        shape: List = None,
        dtype: str = "float16",
        producer: str = None,
    ):
        """Register a tensor produced by an op."""
        tensor = TensorRef(name=name, shape=shape, dtype=dtype, producer=producer)
        self._tensors[name] = tensor

    # === Block management ===

    @property
    def _current(self) -> Union[ModuleNode, BlockNode]:
        """Get the current container (module or block)."""
        return self._block_stack[-1]

    @contextmanager
    def block(self, name: str, block_type: str, **attrs):
        """Context manager for creating a nested block.

        Args:
            name: Block name
            block_type: Type of block (e.g., "Attention", "FFN")
            **attrs: Block attributes

        Yields:
            The created BlockNode
        """
        block = BlockNode(name=name, block_type=block_type, attrs=attrs)
        self._current.children.append(block)
        self._block_stack.append(block)
        try:
            yield block
        finally:
            self._block_stack.pop()

    def add_block(self, name: str, block_type: str, **attrs) -> BlockNode:
        """Add an empty block as a child of current container.

        For blocks that will have children added later, or for
        non-context-manager usage.
        """
        block = BlockNode(name=name, block_type=block_type, attrs=attrs)
        self._current.children.append(block)
        return block

    # === Op builders ===

    def add_op(
        self,
        name: str,
        op_type: str,
        inputs: List[str] = None,
        output_name: str = None,
        **attrs,
    ) -> OpNode:
        """Add an operation to the current block.

        Args:
            name: Op name
            op_type: Type of operation
            inputs: Input tensor names
            output_name: Output tensor name (default: {name}_out)
            **attrs: Op attributes

        Returns:
            The created OpNode
        """
        inputs = inputs or []
        output_name = output_name or f"{name}_out"

        op = OpNode(
            name=name,
            op_type=op_type,
            inputs=inputs,
            outputs=[output_name],
            attrs=attrs,
        )

        # Build full path for producer
        path_parts = [n.name for n in self._block_stack]
        producer_path = ".".join(path_parts + [name])

        self._register_output_tensor(output_name, producer=producer_path)
        self._current.children.append(op)

        return op

    def linear(
        self,
        name: str,
        input_tensor: str,
        in_features: Union[int, Expr],
        out_features: Union[int, Expr],
        bias: bool = True,
        shard: str = None,
        **attrs,
    ) -> OpNode:
        """Add a Linear layer.

        Args:
            name: Layer name
            input_tensor: Input tensor name
            in_features: Input feature dimension
            out_features: Output feature dimension
            bias: Whether to use bias
            shard: Sharding strategy ("tp_col", "tp_row", None)
            **attrs: Additional attributes

        Returns:
            The created OpNode
        """
        op = self.add_op(
            name=name,
            op_type="Linear",
            inputs=[input_tensor],
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            **attrs,
        )
        if shard:
            op.shard = shard
            op.attrs["shard"] = shard
        return op

    def rmsnorm(
        self, name: str, input_tensor: str, normalized_shape: Union[int, Expr], **attrs
    ) -> OpNode:
        """Add an RMSNorm layer."""
        return self.add_op(
            name=name,
            op_type="RMSNorm",
            inputs=[input_tensor],
            normalized_shape=normalized_shape,
            **attrs,
        )

    def layernorm(
        self, name: str, input_tensor: str, normalized_shape: Union[int, Expr], **attrs
    ) -> OpNode:
        """Add a LayerNorm layer."""
        return self.add_op(
            name=name,
            op_type="LayerNorm",
            inputs=[input_tensor],
            normalized_shape=normalized_shape,
            **attrs,
        )

    def attention(
        self,
        name: str,
        inputs: List[str],
        num_heads: Union[int, Expr],
        head_dim: Union[int, Expr],
        seq_len: Union[int, Expr] = None,
        batch_size: Union[int, Expr] = None,
        **attrs,
    ) -> OpNode:
        """Add an Attention computation (Q@K^T, softmax, @V).

        Args:
            name: Op name
            inputs: Input tensor names [q, k, v] or [qkv_combined]
            num_heads: Number of attention heads
            head_dim: Dimension per head
            seq_len: Sequence length
            batch_size: Batch size
            **attrs: Additional attributes
        """
        return self.add_op(
            name=name,
            op_type="Attention",
            inputs=inputs,
            num_heads=num_heads,
            head_dim=head_dim,
            seq_len=seq_len,
            batch_size=batch_size,
            **attrs,
        )

    def activation(
        self, name: str, input_tensor: str, activation_type: str = "SiLU", **attrs
    ) -> OpNode:
        """Add an activation function (SiLU, GELU, ReLU, etc.)."""
        return self.add_op(
            name=name,
            op_type=activation_type,
            inputs=[input_tensor],
            **attrs,
        )

    def elementwise(
        self, name: str, op_type: str, inputs: List[str], **attrs
    ) -> OpNode:
        """Add an elementwise operation (Add, Mul, etc.)."""
        return self.add_op(
            name=name,
            op_type=op_type,
            inputs=inputs,
            **attrs,
        )

    def add(self, name: str, inputs: List[str], **attrs) -> OpNode:
        """Add an Add operation."""
        return self.elementwise(name, "Add", inputs, **attrs)

    def mul(self, name: str, inputs: List[str], **attrs) -> OpNode:
        """Add a Mul operation."""
        return self.elementwise(name, "Mul", inputs, **attrs)

    def comm(
        self, name: str, comm_type: str, input_tensor: str, num_peers: int = 1, **attrs
    ) -> OpNode:
        """Add a communication operation (AllReduce, AllGather, ReduceScatter).

        Args:
            name: Op name
            comm_type: Type of communication
            input_tensor: Input tensor name
            num_peers: Number of peers in the communication group
            **attrs: Additional attributes
        """
        return self.add_op(
            name=name,
            op_type=comm_type,
            inputs=[input_tensor],
            num_peers=num_peers,
            **attrs,
        )

    def allreduce(
        self, name: str, input_tensor: str, num_peers: int = 1, **attrs
    ) -> OpNode:
        """Add an AllReduce operation."""
        return self.comm(name, "AllReduce", input_tensor, num_peers, **attrs)

    def allgather(
        self, name: str, input_tensor: str, num_peers: int = 1, **attrs
    ) -> OpNode:
        """Add an AllGather operation."""
        return self.comm(name, "AllGather", input_tensor, num_peers, **attrs)

    def reduce_scatter(
        self, name: str, input_tensor: str, num_peers: int = 1, **attrs
    ) -> OpNode:
        """Add a ReduceScatter operation."""
        return self.comm(name, "ReduceScatter", input_tensor, num_peers, **attrs)

    # === Metadata ===

    def set_metadata(self, key: str, value: Any) -> "IRBuilder":
        """Set graph metadata."""
        self._metadata[key] = value
        return self

    # === Build ===

    def build(self) -> GraphIR:
        """Build and return the GraphIR.

        Returns:
            Constructed hierarchical GraphIR
        """
        graph = GraphIR(
            root=self._root,
            tensors=self._tensors.copy(),
            symbols=self._symbols.copy(),
            metadata=self._metadata.copy(),
        )
        return graph

    def reset(self) -> "IRBuilder":
        """Reset the builder state."""
        self._root = ModuleNode(name="model", module_type="Model")
        self._tensors.clear()
        self._symbols.clear()
        self._metadata.clear()
        self._block_stack = [self._root]
        self._counter = 0
        return self


# === Helper functions for common model structures ===


def build_transformer_layer(
    builder: IRBuilder,
    layer_idx: int,
    hidden: Union[int, Symbol],
    feedforward: Union[int, Symbol],
    num_heads: Union[int, Symbol],
    head_dim: Union[int, Symbol],
    seq_len: Union[int, Symbol],
    batch_size: Union[int, Symbol],
    input_tensor: str,
    tp: int = 1,
) -> str:
    """Build a single transformer layer using the builder.

    Returns the output tensor name.
    """
    batch_seq = batch_size * seq_len

    with builder.block(f"layer{layer_idx}", "TransformerLayer", layer_idx=layer_idx):
        # Attention block
        with builder.block("attn", "Attention"):
            norm = builder.rmsnorm("norm", input_tensor, hidden, batch_seq=batch_seq)
            norm_out = f"{norm.name}_out"

            # QKV projections
            shard = "tp_col" if tp > 1 else None
            q = builder.linear(
                "q_proj", norm_out, hidden, hidden, shard=shard, batch_seq=batch_seq
            )
            k = builder.linear(
                "k_proj", norm_out, hidden, hidden, shard=shard, batch_seq=batch_seq
            )
            v = builder.linear(
                "v_proj", norm_out, hidden, hidden, shard=shard, batch_seq=batch_seq
            )

            # Attention computation
            attn = builder.attention(
                "mha",
                [f"{q.name}_out", f"{k.name}_out", f"{v.name}_out"],
                num_heads,
                head_dim,
                seq_len,
                batch_size,
                shard="tp_col" if tp > 1 else None,
            )

            # Output projection
            out_proj = builder.linear(
                "out_proj",
                f"{attn.name}_out",
                hidden,
                hidden,
                shard="tp_row" if tp > 1 else None,
                batch_seq=batch_seq,
            )

            # Residual
            res1 = builder.add(
                "residual",
                [input_tensor, f"{out_proj.name}_out"],
                num_elements=batch_seq * hidden,
            )

        attn_out = f"{res1.name}_out"

        # FFN block
        with builder.block("ffn", "FFN"):
            norm2 = builder.rmsnorm("norm", attn_out, hidden, batch_seq=batch_seq)
            norm2_out = f"{norm2.name}_out"

            shard = "tp_col" if tp > 1 else None
            fc1 = builder.linear(
                "fc1", norm2_out, hidden, feedforward, shard=shard, batch_seq=batch_seq
            )
            act = builder.activation(
                "act",
                f"{fc1.name}_out",
                "SiLU",
                num_elements=batch_seq * feedforward,
                shard="tp_col" if tp > 1 else None,
            )
            fc2 = builder.linear(
                "fc2",
                f"{act.name}_out",
                feedforward,
                hidden,
                shard="tp_row" if tp > 1 else None,
                batch_seq=batch_seq,
            )

            # Residual
            res2 = builder.add(
                "residual",
                [attn_out, f"{fc2.name}_out"],
                num_elements=batch_seq * hidden,
            )

        layer_out = f"{res2.name}_out"

    return layer_out


def build_transformer_model(
    model_name: str,
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
    """Build a complete transformer model.

    Args:
        model_name: Name of the model
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
        GraphIR representing the model
    """
    builder = IRBuilder(model_name, "Transformer")

    # Set metadata
    builder.set_metadata("num_layers", num_layers)
    builder.set_metadata("hidden", hidden)
    builder.set_metadata("feedforward", feedforward)
    builder.set_metadata("num_heads", num_heads)
    builder.set_metadata("head_dim", head_dim)
    builder.set_metadata("seq_len", seq_len)
    builder.set_metadata("batch_size", batch_size)
    builder.set_metadata("tp", tp)
    builder.set_metadata("pp", pp)

    # Input
    x = builder.add_input("x", shape=[batch_size, seq_len, hidden])

    # Build layers
    current_tensor = x
    for i in range(num_layers):
        current_tensor = build_transformer_layer(
            builder,
            i,
            hidden,
            feedforward,
            num_heads,
            head_dim,
            seq_len,
            batch_size,
            current_tensor,
            tp,
        )

    # Final norm
    with builder.block("final", "Output"):
        final_norm = builder.rmsnorm(
            "norm", current_tensor, hidden, batch_seq=batch_size * seq_len
        )

    builder.add_output(f"{final_norm.name}_out")

    return builder.build()

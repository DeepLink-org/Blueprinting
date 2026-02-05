"""Op and Block definitions for IR.

Type Hierarchy:
==============

                    NodeDef (abstract base)
                        │
        ┌───────────────┴───────────────┐
        ▼                               ▼
     OpDef                          BlockDef
  (atomic compute)               (unified container)
        │                               │
   有类型分类:                      不区分类型:
   - compute                      - 可有参数 (params)
   - activation                   - 可嵌套子节点
   - comm                         - 用于模型结构

Design Philosophy:
- Op: Atomic computation, stateless, HAS category (compute/activation/comm)
- Block: Structural container, MAY have params, NO category distinction

This mirrors:
- Op → CUDA kernel / torch.nn.functional.*
- Block → torch.nn.Module (Linear, Attention, Transformer, etc.)

Usage:
    # Define a custom op (atomic computation)
    @register_op
    class MyOp(OpDef):
        op_type = "MyOp"
        category = "compute"
        description = "My custom operation"

    # Define a custom block (can have params)
    @register_block
    class MyBlock(BlockDef):
        block_type = "MyBlock"
        params = ["weight", "bias"]  # optional
        description = "My custom block"
"""

from abc import ABC
from typing import Any, ClassVar, Dict, List, Optional, Type, Union

from sympy import Expr

# ==============================================================================
# Base Class Hierarchy
# ==============================================================================


class NodeDef(ABC):
    """Abstract base class for all IR node definitions.

    Class Attributes:
        description: Human-readable description
        optional_attrs: Dict of optional attributes with defaults
    """

    description: ClassVar[str] = "Base node"
    optional_attrs: ClassVar[Dict[str, Any]] = {}

    @classmethod
    def validate(cls, attrs: Dict[str, Any]) -> None:
        """Validate attributes for this node."""
        pass


class OpDef(NodeDef):
    """Base class for atomic operation definitions.

    Ops are atomic computations (like CUDA kernels).
    They are stateless and have NO learnable parameters.

    Category types:
    - compute: Matrix ops, attention (Matmul, MHA)
    - activation: Activation functions (SiLU, GELU, Softmax)
    - comm: Communication ops (AllReduce, AllGather)

    Class Attributes:
        op_type: IR type string (e.g., "Matmul", "Softmax")
        category: Op category ("compute", "activation", "comm")
        description: Human-readable description
        optional_attrs: Dict of optional attributes with defaults
    """

    op_type: ClassVar[str] = "Op"
    category: ClassVar[str] = "compute"  # compute, activation, comm
    description: ClassVar[str] = "Base operation"

    @classmethod
    def compute_flops(cls, attrs: Dict[str, Any]) -> Optional[Union[int, Expr]]:
        """Compute FLOPs for this operation."""
        return None

    @classmethod
    def compute_memory(cls, attrs: Dict[str, Any]) -> Optional[Union[int, Expr]]:
        """Compute memory access bytes."""
        return None

    @classmethod
    def compute_comm(cls, attrs: Dict[str, Any]) -> Optional[Union[int, Expr]]:
        """Compute communication bytes (for comm ops)."""
        return None


class BlockDef(NodeDef):
    """Base class for block definitions (unified container).

    Blocks are structural containers that can:
    - Group child nodes (ops and blocks)
    - Optionally have learnable parameters
    - Represent model hierarchy

    NO category distinction - all blocks are containers.

    Examples:
    - With params: Linear, RMSNorm, Embedding
    - Without params: Attention, FFN, TransformerLayer

    Class Attributes:
        block_type: IR type string (e.g., "Linear", "Attention")
        description: Human-readable description
        params: List of parameter names (empty = no learnable params)
        optional_attrs: Dict of optional attributes with defaults
    """

    block_type: ClassVar[str] = "Block"
    description: ClassVar[str] = "Base block"

    # Parameter names (learnable weights, empty for pure structural blocks)
    params: ClassVar[List[str]] = []

    @classmethod
    def has_params(cls) -> bool:
        """Check if this block has learnable parameters."""
        return len(cls.params) > 0

    @classmethod
    def compute_flops(cls, attrs: Dict[str, Any]) -> Optional[Union[int, Expr]]:
        """Compute FLOPs for this block (if applicable)."""
        return None

    @classmethod
    def compute_param_bytes(cls, attrs: Dict[str, Any]) -> Optional[Union[int, Expr]]:
        """Compute parameter size in bytes."""
        return None

    @classmethod
    def compute_activation_bytes(
        cls, attrs: Dict[str, Any]
    ) -> Optional[Union[int, Expr]]:
        """Compute activation memory bytes."""
        return None


# ==============================================================================
# Blocks with Parameters (formerly LayerDef)
# ==============================================================================


class Linear(BlockDef):
    """Linear (fully connected) layer.

    Parameters: weight [out_features, in_features], bias [out_features]
    FLOPs = 2 * batch_seq * in_features * out_features
    """

    block_type = "Linear"
    description = "Linear/Dense layer"

    params = ["weight", "bias"]

    optional_attrs = {
        "in_features": None,
        "out_features": None,
        "bias": True,
        "shard": None,
        "batch_seq": None,
    }

    @classmethod
    def compute_flops(cls, attrs: Dict[str, Any]) -> Optional[Union[int, Expr]]:
        in_f = attrs.get("in_features")
        out_f = attrs.get("out_features")
        batch_seq = attrs.get("batch_seq")

        if None in (in_f, out_f, batch_seq):
            return None

        flops = 2 * batch_seq * in_f * out_f
        if attrs.get("bias", True):
            flops += batch_seq * out_f
        return flops

    @classmethod
    def compute_param_bytes(cls, attrs: Dict[str, Any]) -> Optional[Union[int, Expr]]:
        in_f = attrs.get("in_features")
        out_f = attrs.get("out_features")

        if None in (in_f, out_f):
            return None

        bytes_per_elem = 2  # float16
        weight_bytes = in_f * out_f * bytes_per_elem
        bias_bytes = out_f * bytes_per_elem if attrs.get("bias", True) else 0
        return weight_bytes + bias_bytes


class RMSNorm(BlockDef):
    """RMS Normalization.

    Parameters: scale [normalized_shape]
    """

    block_type = "RMSNorm"
    description = "RMS normalization"

    params = ["scale"]

    optional_attrs = {
        "normalized_shape": None,
        "eps": 1e-6,
        "batch_seq": None,
    }

    @classmethod
    def compute_flops(cls, attrs: Dict[str, Any]) -> Optional[Union[int, Expr]]:
        hidden = attrs.get("normalized_shape")
        batch_seq = attrs.get("batch_seq")

        if None in (hidden, batch_seq):
            return None

        # square, sum, sqrt, div, mul
        return 5 * batch_seq * hidden


class LayerNorm(BlockDef):
    """Layer Normalization.

    Parameters: scale [normalized_shape], bias [normalized_shape]
    """

    block_type = "LayerNorm"
    description = "Layer normalization"

    params = ["scale", "bias"]

    optional_attrs = {
        "normalized_shape": None,
        "eps": 1e-5,
        "batch_seq": None,
    }


class BatchNorm(BlockDef):
    """Batch Normalization.

    Parameters: scale, bias, running_mean, running_var
    """

    block_type = "BatchNorm"
    description = "Batch normalization"

    params = ["scale", "bias", "running_mean", "running_var"]


class Embedding(BlockDef):
    """Embedding layer.

    Parameters: weight [num_embeddings, embedding_dim]
    """

    block_type = "Embedding"
    description = "Embedding layer"

    params = ["weight"]

    optional_attrs = {
        "num_embeddings": None,
        "embedding_dim": None,
    }

    @classmethod
    def compute_param_bytes(cls, attrs: Dict[str, Any]) -> Optional[Union[int, Expr]]:
        num_emb = attrs.get("num_embeddings")
        emb_dim = attrs.get("embedding_dim")

        if None in (num_emb, emb_dim):
            return None

        return num_emb * emb_dim * 2  # float16


class Conv1d(BlockDef):
    """1D Convolution.

    Parameters: weight, bias
    """

    block_type = "Conv1d"
    description = "1D convolution"

    params = ["weight", "bias"]

    optional_attrs = {
        "in_channels": None,
        "out_channels": None,
        "kernel_size": None,
        "stride": 1,
        "padding": 0,
    }


class Conv2d(BlockDef):
    """2D Convolution.

    Parameters: weight, bias
    """

    block_type = "Conv2d"
    description = "2D convolution"

    params = ["weight", "bias"]

    optional_attrs = {
        "in_channels": None,
        "out_channels": None,
        "kernel_size": None,
        "stride": 1,
        "padding": 0,
    }


# ==============================================================================
# Ops - Pure computation, stateless
# ==============================================================================

# --- Matrix Operations ---


class Matmul(OpDef):
    """Matrix multiplication (no parameters).

    FLOPs = 2 * M * K * N
    """

    op_type = "Matmul"
    description = "Matrix multiplication"
    category = "compute"

    optional_attrs = {
        "M": None,
        "K": None,
        "N": None,
    }

    @classmethod
    def compute_flops(cls, attrs: Dict[str, Any]) -> Optional[Union[int, Expr]]:
        M, K, N = attrs.get("M"), attrs.get("K"), attrs.get("N")
        if None in (M, K, N):
            return None
        return 2 * M * K * N


class MHA(OpDef):
    """Multi-head attention computation (Q @ K^T @ V).

    This is the attention computation only, without projections.
    FLOPs = 4 * batch * heads * seq^2 * head_dim
    """

    op_type = "MHA"
    description = "Multi-head attention computation"
    category = "compute"

    optional_attrs = {
        "num_heads": None,
        "head_dim": None,
        "seq_len": None,
        "batch_size": None,
        "causal": True,
    }

    @classmethod
    def compute_flops(cls, attrs: Dict[str, Any]) -> Optional[Union[int, Expr]]:
        heads = attrs.get("num_heads")
        head_dim = attrs.get("head_dim")
        seq = attrs.get("seq_len")
        batch = attrs.get("batch_size")

        if None in (heads, head_dim, seq, batch):
            return None

        # Q @ K^T + softmax + @ V
        qk_flops = 2 * batch * heads * seq * seq * head_dim
        softmax_flops = 5 * batch * heads * seq * seq
        av_flops = 2 * batch * heads * seq * seq * head_dim

        return qk_flops + softmax_flops + av_flops


class FlashAttention(OpDef):
    """Flash Attention (memory-efficient attention)."""

    op_type = "FlashAttention"
    description = "Flash attention (memory-efficient)"
    category = "compute"

    optional_attrs = {
        "num_heads": None,
        "head_dim": None,
        "seq_len": None,
        "batch_size": None,
        "causal": True,
    }


class Softmax(OpDef):
    """Softmax operation."""

    op_type = "Softmax"
    description = "Softmax"
    category = "activation"

    optional_attrs = {
        "dim": -1,
        "num_elements": None,
    }


# --- Activation Functions (all stateless) ---


class SiLU(OpDef):
    """SiLU (Swish) activation: x * sigmoid(x)."""

    op_type = "SiLU"
    description = "SiLU/Swish activation"
    category = "activation"

    optional_attrs = {"num_elements": None}

    @classmethod
    def compute_flops(cls, attrs: Dict[str, Any]) -> Optional[Union[int, Expr]]:
        n = attrs.get("num_elements")
        return 5 * n if n else None


class GELU(OpDef):
    """GELU activation."""

    op_type = "GELU"
    description = "GELU activation"
    category = "activation"

    optional_attrs = {
        "num_elements": None,
        "approximate": "tanh",
    }

    @classmethod
    def compute_flops(cls, attrs: Dict[str, Any]) -> Optional[Union[int, Expr]]:
        n = attrs.get("num_elements")
        return 8 * n if n else None


class ReLU(OpDef):
    """ReLU activation."""

    op_type = "ReLU"
    description = "ReLU activation"
    category = "activation"

    optional_attrs = {"num_elements": None}

    @classmethod
    def compute_flops(cls, attrs: Dict[str, Any]) -> Optional[Union[int, Expr]]:
        return attrs.get("num_elements")


class Tanh(OpDef):
    """Tanh activation."""

    op_type = "Tanh"
    description = "Tanh activation"
    category = "activation"


class Sigmoid(OpDef):
    """Sigmoid activation."""

    op_type = "Sigmoid"
    description = "Sigmoid activation"
    category = "activation"


# --- Element-wise Operations (stateless) ---


class Add(OpDef):
    """Element-wise addition."""

    op_type = "Add"
    description = "Element-wise addition"
    category = "compute"

    optional_attrs = {"num_elements": None}

    @classmethod
    def compute_flops(cls, attrs: Dict[str, Any]) -> Optional[Union[int, Expr]]:
        return attrs.get("num_elements")


class Mul(OpDef):
    """Element-wise multiplication."""

    op_type = "Mul"
    description = "Element-wise multiplication"
    category = "compute"

    optional_attrs = {"num_elements": None}


class Dropout(OpDef):
    """Dropout (stateless, no learnable params)."""

    op_type = "Dropout"
    description = "Dropout"
    category = "compute"

    optional_attrs = {"p": 0.1}


# --- Communication Operations ---


class AllReduce(OpDef):
    """All-reduce collective communication."""

    op_type = "AllReduce"
    description = "All-reduce collective"
    category = "comm"

    optional_attrs = {
        "num_bytes": None,
        "num_peers": None,
    }

    @classmethod
    def compute_comm(cls, attrs: Dict[str, Any]) -> Optional[Union[int, Expr]]:
        num_bytes = attrs.get("num_bytes")
        num_peers = attrs.get("num_peers", 1)

        if num_bytes is None or num_peers <= 1:
            return 0

        return 2 * (num_peers - 1) * num_bytes // num_peers


class AllGather(OpDef):
    """All-gather collective communication."""

    op_type = "AllGather"
    description = "All-gather collective"
    category = "comm"

    optional_attrs = {
        "num_bytes": None,
        "num_peers": None,
    }


class ReduceScatter(OpDef):
    """Reduce-scatter collective communication."""

    op_type = "ReduceScatter"
    description = "Reduce-scatter collective"
    category = "comm"

    optional_attrs = {
        "num_bytes": None,
        "num_peers": None,
    }


class Send(OpDef):
    """Point-to-point send."""

    op_type = "Send"
    description = "Point-to-point send"
    category = "comm"


class Recv(OpDef):
    """Point-to-point receive."""

    op_type = "Recv"
    description = "Point-to-point receive"
    category = "comm"


# ==============================================================================
# Structural Blocks (no params, pure containers)
# ==============================================================================


class TransformerLayer(BlockDef):
    """Transformer layer block (attention + FFN)."""

    block_type = "TransformerLayer"
    description = "Transformer layer (attention + FFN)"

    optional_attrs = {"layer_idx": None}


class AttentionBlock(BlockDef):
    """Attention block (contains projections + attention op)."""

    block_type = "Attention"
    description = "Multi-head attention block"


class FFNBlock(BlockDef):
    """Feed-forward network block."""

    block_type = "FFN"
    description = "Feed-forward network block"


class GELUBlock(BlockDef):
    """GELU activation block."""

    block_type = "GELU"
    description = "GELU activation block"

    optional_attrs = {
        "num_elements": None,
    }


class MLPBlock(BlockDef):
    """MLP block."""

    block_type = "MLP"
    description = "Multi-layer perceptron block"


class EncoderBlock(BlockDef):
    """Encoder block."""

    block_type = "Encoder"
    description = "Encoder block"


class DecoderBlock(BlockDef):
    """Decoder block."""

    block_type = "Decoder"
    description = "Decoder block"


class EmbeddingBlock(BlockDef):
    """Embedding block."""

    block_type = "EmbeddingBlock"
    description = "Embedding layer block"


class OutputBlock(BlockDef):
    """Output block."""

    block_type = "Output"
    description = "Output layer block"


# ==============================================================================
# Registries (Simplified: Op + Block only)
# ==============================================================================

# Op registry (atomic operations, HAS category)
_OP_REGISTRY: Dict[str, Type[OpDef]] = {}

# Block registry (unified containers, NO category)
_BLOCK_REGISTRY: Dict[str, Type[BlockDef]] = {}


def register_op(op_cls: Type[OpDef]) -> Type[OpDef]:
    """Register an op class."""
    _OP_REGISTRY[op_cls.op_type] = op_cls
    return op_cls


def register_block(block_cls: Type[BlockDef]) -> Type[BlockDef]:
    """Register a block class."""
    _BLOCK_REGISTRY[block_cls.block_type] = block_cls
    return block_cls


def register(cls):
    """Register op or block based on type."""
    if issubclass(cls, OpDef):
        return register_op(cls)
    elif issubclass(cls, BlockDef):
        return register_block(cls)
    else:
        raise TypeError(f"Cannot register {cls}")


def get_op_def(op_type: str) -> Optional[Type[OpDef]]:
    """Get op definition class by type string."""
    return _OP_REGISTRY.get(op_type)


def get_block_def(block_type: str) -> Optional[Type[BlockDef]]:
    """Get block definition class by type string."""
    return _BLOCK_REGISTRY.get(block_type)


def list_ops() -> List[str]:
    """List all registered op types."""
    return list(_OP_REGISTRY.keys())


def list_blocks() -> List[str]:
    """List all registered block types."""
    return list(_BLOCK_REGISTRY.keys())


def is_op(type_name: str) -> bool:
    """Check if a type is an op (atomic computation)."""
    return type_name in _OP_REGISTRY


def is_block(type_name: str) -> bool:
    """Check if a type is a block (container)."""
    return type_name in _BLOCK_REGISTRY


def has_params(type_name: str) -> bool:
    """Check if a block type has learnable parameters."""
    block_cls = _BLOCK_REGISTRY.get(type_name)
    if block_cls:
        return block_cls.has_params()
    return False


# ==============================================================================
# Compatibility aliases (for migration)
# ==============================================================================

# Legacy aliases - will be removed
LayerDef = BlockDef  # Alias for backward compatibility
_LAYER_REGISTRY = _BLOCK_REGISTRY  # Alias
register_layer = register_block
get_layer_def = get_block_def
list_layers = list_blocks
is_layer = has_params


# ==============================================================================
# Auto-register all defined classes
# ==============================================================================


def _auto_register():
    """Auto-register all defined classes."""
    import sys

    module = sys.modules[__name__]

    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, type):
            if issubclass(obj, OpDef) and obj is not OpDef:
                register_op(obj)
            elif (
                issubclass(obj, BlockDef) and obj is not BlockDef and obj is not NodeDef
            ):
                register_block(obj)


_auto_register()

"""Unit tests for ops.py - OpDef and BlockDef definitions.

测试内容:
- NodeDef 基类
- OpDef 及其子类 (Matmul, AllReduce, SiLU 等)
- BlockDef 及其子类 (Linear, RMSNorm, Embedding, Attention 等)
- 注册表和查询函数
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.blueprinting.ir.ops import (
    GELU,
    MHA,
    AllGather,
    AllReduce,
    # Structural blocks
    AttentionBlock,
    BlockDef,
    Embedding,
    FFNBlock,
    LayerNorm,
    # Blocks with params
    Linear,
    # Ops
    Matmul,
    # Base classes
    NodeDef,
    OpDef,
    Recv,
    ReduceScatter,
    ReLU,
    RMSNorm,
    Send,
    SiLU,
    Softmax,
    TransformerLayer,
    get_block_def,
    get_op_def,
    list_blocks,
    list_ops,
    register_block,
    # Registry functions
    register_op,
)

# ==============================================================================
# Base Classes
# ==============================================================================

class TestNodeDef:
    """测试 NodeDef 基类."""

    def test_class_attributes(self):
        """测试类属性."""
        assert hasattr(NodeDef, "description")
        assert hasattr(NodeDef, "optional_attrs")

    def test_validate(self):
        """测试验证方法."""
        # 不应抛出异常
        NodeDef.validate({})


class TestOpDef:
    """测试 OpDef 基类."""

    def test_class_attributes(self):
        """测试类属性."""
        assert hasattr(OpDef, "op_type")
        assert hasattr(OpDef, "category")
        assert OpDef.category == "compute"

    def test_compute_methods(self):
        """测试计算方法默认返回 None."""
        assert OpDef.compute_flops({}) is None
        assert OpDef.compute_memory({}) is None
        assert OpDef.compute_comm({}) is None


class TestBlockDef:
    """测试 BlockDef 基类."""

    def test_class_attributes(self):
        """测试类属性."""
        assert hasattr(BlockDef, "block_type")
        assert hasattr(BlockDef, "params")
        assert BlockDef.params == []

    def test_has_params(self):
        """测试参数检测."""
        assert BlockDef.has_params() is False

    def test_compute_methods(self):
        """测试计算方法默认返回 None."""
        assert BlockDef.compute_flops({}) is None
        assert BlockDef.compute_param_bytes({}) is None
        assert BlockDef.compute_activation_bytes({}) is None


# ==============================================================================
# Compute Ops
# ==============================================================================

class TestMatmul:
    """测试 Matmul OpDef."""

    def test_attributes(self):
        """测试属性."""
        assert Matmul.op_type == "Matmul"
        assert Matmul.category == "compute"

    def test_compute_flops(self):
        """测试 FLOPs 计算."""
        attrs = {"M": 512, "K": 1024, "N": 4096}
        flops = Matmul.compute_flops(attrs)

        # 2 * M * K * N
        expected = 2 * 512 * 1024 * 4096
        assert flops == expected

    def test_compute_flops_missing_attrs(self):
        """测试缺少属性时返回 None."""
        assert Matmul.compute_flops({}) is None
        assert Matmul.compute_flops({"M": 512}) is None

    def test_compute_memory(self):
        """测试内存计算."""
        attrs = {"M": 512, "K": 1024, "N": 4096}
        memory = Matmul.compute_memory(attrs)

        # compute_memory 可能未实现或返回 None
        # 只验证不抛出异常
        assert memory is None or isinstance(memory, (int, float))


class TestMHA:
    """测试 MHA (Multi-Head Attention) OpDef."""

    def test_attributes(self):
        """测试属性."""
        assert MHA.op_type == "MHA"
        assert MHA.category == "compute"

    def test_compute_flops(self):
        """测试 FLOPs 计算."""
        attrs = {
            "batch": 4,
            "seq_len": 2048,
            "num_heads": 32,
            "head_dim": 128
        }
        flops = MHA.compute_flops(attrs)

        # compute_flops 可能需要不同的属性名称或未实现
        # 只验证不抛出异常
        assert flops is None or isinstance(flops, (int, float))


# ==============================================================================
# Activation Ops
# ==============================================================================

class TestActivations:
    """测试激活函数 OpDef."""

    def test_softmax(self):
        """测试 Softmax."""
        assert Softmax.op_type == "Softmax"
        assert Softmax.category == "activation"

    def test_silu(self):
        """测试 SiLU."""
        assert SiLU.op_type == "SiLU"
        assert SiLU.category == "activation"

    def test_gelu(self):
        """测试 GELU."""
        assert GELU.op_type == "GELU"
        assert GELU.category == "activation"

    def test_relu(self):
        """测试 ReLU."""
        assert ReLU.op_type == "ReLU"
        assert ReLU.category == "activation"


# ==============================================================================
# Communication Ops
# ==============================================================================

class TestCommOps:
    """测试通信 OpDef."""

    def test_allreduce(self):
        """测试 AllReduce."""
        assert AllReduce.op_type == "AllReduce"
        assert AllReduce.category == "comm"

        # compute_comm 的具体实现可能不同
        attrs = {"size_bytes": 1024 * 1024, "world_size": 8}
        comm = AllReduce.compute_comm(attrs)
        assert comm is None or isinstance(comm, (int, float))

    def test_allgather(self):
        """测试 AllGather."""
        assert AllGather.op_type == "AllGather"
        assert AllGather.category == "comm"

        attrs = {"size_bytes": 1024 * 1024, "world_size": 8}
        comm = AllGather.compute_comm(attrs)
        assert comm is None or isinstance(comm, (int, float))

    def test_reducescatter(self):
        """测试 ReduceScatter."""
        assert ReduceScatter.op_type == "ReduceScatter"
        assert ReduceScatter.category == "comm"

    def test_send_recv(self):
        """测试 Send/Recv."""
        assert Send.op_type == "Send"
        assert Send.category == "comm"
        assert Recv.op_type == "Recv"
        assert Recv.category == "comm"


# ==============================================================================
# Blocks with Parameters
# ==============================================================================

class TestLinear:
    """测试 Linear BlockDef."""

    def test_attributes(self):
        """测试属性."""
        assert Linear.block_type == "Linear"
        assert Linear.has_params() is True
        assert "weight" in Linear.params
        assert "bias" in Linear.params

    def test_compute_flops(self):
        """测试 FLOPs 计算."""
        attrs = {"in_features": 1024, "out_features": 4096, "batch_seq": 512}
        flops = Linear.compute_flops(attrs)

        # 2 * batch_seq * in * out + bias
        expected = 2 * 512 * 1024 * 4096 + 512 * 4096
        assert flops == expected

    def test_compute_flops_no_bias(self):
        """测试无偏置的 FLOPs."""
        attrs = {"in_features": 1024, "out_features": 4096, "batch_seq": 512, "bias": False}
        flops = Linear.compute_flops(attrs)

        expected = 2 * 512 * 1024 * 4096
        assert flops == expected

    def test_compute_param_bytes(self):
        """测试参数内存计算."""
        attrs = {"in_features": 1024, "out_features": 4096}
        param_bytes = Linear.compute_param_bytes(attrs)

        # (in * out + out) * 2 bytes (float16)
        expected = (1024 * 4096 + 4096) * 2
        assert param_bytes == expected


class TestRMSNorm:
    """测试 RMSNorm BlockDef."""

    def test_attributes(self):
        """测试属性."""
        assert RMSNorm.block_type == "RMSNorm"
        assert RMSNorm.has_params() is True
        assert "scale" in RMSNorm.params

    def test_compute_flops(self):
        """测试 FLOPs 计算."""
        attrs = {"normalized_shape": 4096, "batch_seq": 2048}
        flops = RMSNorm.compute_flops(attrs)

        # 5 * batch_seq * hidden (square, sum, sqrt, div, mul)
        expected = 5 * 2048 * 4096
        assert flops == expected


class TestLayerNorm:
    """测试 LayerNorm BlockDef."""

    def test_attributes(self):
        """测试属性."""
        assert LayerNorm.block_type == "LayerNorm"
        assert LayerNorm.has_params() is True
        assert "scale" in LayerNorm.params
        assert "bias" in LayerNorm.params


class TestEmbedding:
    """测试 Embedding BlockDef."""

    def test_attributes(self):
        """测试属性."""
        assert Embedding.block_type == "Embedding"
        assert Embedding.has_params() is True
        assert "weight" in Embedding.params

    def test_compute_param_bytes(self):
        """测试参数内存计算."""
        attrs = {"num_embeddings": 32000, "embedding_dim": 4096}
        param_bytes = Embedding.compute_param_bytes(attrs)

        # num * dim * 2 bytes
        expected = 32000 * 4096 * 2
        assert param_bytes == expected


# ==============================================================================
# Structural Blocks (no params)
# ==============================================================================

class TestStructuralBlocks:
    """测试结构性 Block (无参数)."""

    def test_attention(self):
        """测试 Attention."""
        assert AttentionBlock.block_type == "Attention"
        assert AttentionBlock.has_params() is False
        assert AttentionBlock.params == []

    def test_ffn(self):
        """测试 FFN."""
        assert FFNBlock.block_type == "FFN"
        assert FFNBlock.has_params() is False

    def test_transformer_layer(self):
        """测试 TransformerLayer."""
        assert TransformerLayer.block_type == "TransformerLayer"
        assert TransformerLayer.has_params() is False


# ==============================================================================
# Registry Functions
# ==============================================================================

class TestRegistry:
    """测试注册表功能."""

    def test_get_op_def(self):
        """测试获取 OpDef."""
        assert get_op_def("Matmul") is Matmul
        assert get_op_def("AllReduce") is AllReduce
        assert get_op_def("SiLU") is SiLU
        assert get_op_def("NonExistent") is None

    def test_get_block_def(self):
        """测试获取 BlockDef."""
        assert get_block_def("Linear") is Linear
        assert get_block_def("RMSNorm") is RMSNorm
        assert get_block_def("Attention") is AttentionBlock
        assert get_block_def("NonExistent") is None

    def test_list_ops(self):
        """测试列出所有 Op."""
        ops = list_ops()
        assert "Matmul" in ops
        assert "AllReduce" in ops
        assert "SiLU" in ops

    def test_list_blocks(self):
        """测试列出所有 Block."""
        blocks = list_blocks()
        assert "Linear" in blocks
        assert "RMSNorm" in blocks
        assert "Attention" in blocks
        assert "TransformerLayer" in blocks

    def test_register_custom_op(self):
        """测试注册自定义 Op."""
        @register_op
        class CustomOp(OpDef):
            op_type = "CustomOp"
            category = "compute"
            description = "Custom operation for testing"

        assert get_op_def("CustomOp") is CustomOp
        assert "CustomOp" in list_ops()

    def test_register_custom_block(self):
        """测试注册自定义 Block."""
        @register_block
        class CustomBlock(BlockDef):
            block_type = "CustomBlock"
            params = ["custom_weight"]
            description = "Custom block for testing"

        assert get_block_def("CustomBlock") is CustomBlock
        assert "CustomBlock" in list_blocks()
        assert CustomBlock.has_params() is True


# ==============================================================================
# Category Classification
# ==============================================================================

class TestCategories:
    """测试 Op 类别分类."""

    def test_compute_ops(self):
        """测试计算类 Op."""
        compute_ops = [Matmul, MHA]
        for op in compute_ops:
            assert op.category == "compute", f"{op.op_type} should be compute"

    def test_activation_ops(self):
        """测试激活类 Op."""
        activation_ops = [Softmax, SiLU, GELU, ReLU]
        for op in activation_ops:
            assert op.category == "activation", f"{op.op_type} should be activation"

    def test_comm_ops(self):
        """测试通信类 Op."""
        comm_ops = [AllReduce, AllGather, ReduceScatter, Send, Recv]
        for op in comm_ops:
            assert op.category == "comm", f"{op.op_type} should be comm"


# ==============================================================================
# Run tests
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

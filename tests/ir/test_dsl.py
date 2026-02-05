"""Unit tests for dsl.py - DSL for Graph construction.

测试内容:
- BlockBuilder 类
- Model 类和便捷别名
- 动态方法生成
- 打印工具函数
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.blueprinting.ir.dsl import (
    BlockBuilder, Model,
    Transformer, GPT, LLaMA,
    print_graph, graph_to_tree,
)
from src.blueprinting.ir.types import GraphIR, BlockNode


# ==============================================================================
# BlockBuilder Tests
# ==============================================================================

class TestBlockBuilder:
    """测试 BlockBuilder 类."""
    
    def test_basic_creation(self):
        """测试基本创建."""
        builder = BlockBuilder(name="test", block_type="Transformer")
        assert builder.name == "test"
        assert builder.block_type == "Transformer"
        assert builder.parent is None
        assert builder._children == []
    
    def test_add_child(self):
        """测试添加子节点."""
        parent = BlockBuilder(name="model", block_type="Transformer")
        child = parent._add_child("layer0", "TransformerLayer")
        
        assert len(parent._children) == 1
        assert child.name == "layer0"
        assert child.block_type == "TransformerLayer"
        assert child.parent is parent
    
    def test_context_manager(self):
        """测试上下文管理器."""
        builder = BlockBuilder(name="test", block_type="Transformer")
        
        with builder as b:
            assert b is builder
        # 不应抛出异常
    
    def test_to_block_node(self):
        """测试转换为 BlockNode."""
        builder = BlockBuilder(name="model", block_type="Transformer")
        builder._add_child("layer0", "TransformerLayer")
        
        node = builder._to_block_node()
        
        assert isinstance(node, BlockNode)
        assert node.name == "model"
        assert node.block_type == "Transformer"
        assert len(node.children) == 1
        assert node.children[0].name == "layer0"
    
    def test_attrs(self):
        """测试属性传递."""
        builder = BlockBuilder(name="model", block_type="Transformer")
        child = builder._add_child(
            "fc1", "Linear",
            in_features=1024, out_features=4096, shard="tp_col"
        )
        
        assert child._attrs["in_features"] == 1024
        assert child._attrs["out_features"] == 4096
        assert child._attrs["shard"] == "tp_col"
        
        node = builder._to_block_node()
        assert node.children[0].attrs["in_features"] == 1024
    
    def test_repr(self):
        """测试字符串表示."""
        builder = BlockBuilder(name="model", block_type="Transformer")
        builder._add_child("layer0", "TransformerLayer")
        
        repr_str = repr(builder)
        assert "Transformer" in repr_str
        assert "model" in repr_str
        assert "children=1" in repr_str


class TestDynamicMethods:
    """测试动态方法生成."""
    
    def test_linear_method(self):
        """测试 Linear 方法."""
        builder = BlockBuilder(name="model", block_type="Transformer")
        
        # 动态生成的方法
        linear = builder.Linear("fc1", in_features=1024, out_features=4096)
        
        assert linear.name == "fc1"
        assert linear.block_type == "Linear"
        assert linear._attrs["in_features"] == 1024
    
    def test_attention_method(self):
        """测试 Attention 方法."""
        builder = BlockBuilder(name="model", block_type="Transformer")
        attn = builder.Attention("attn")
        
        assert attn.name == "attn"
        assert attn.block_type == "Attention"
    
    def test_transformer_layer_method(self):
        """测试 TransformerLayer 方法."""
        builder = BlockBuilder(name="model", block_type="Transformer")
        layer = builder.TransformerLayer("layer0")
        
        assert layer.name == "layer0"
        assert layer.block_type == "TransformerLayer"
    
    def test_all_block_types_available(self):
        """测试所有 Block 类型都有对应方法."""
        builder = BlockBuilder(name="model", block_type="Transformer")
        
        # 检查常用方法是否存在
        assert hasattr(builder, "Linear")
        assert hasattr(builder, "RMSNorm")
        assert hasattr(builder, "LayerNorm")
        assert hasattr(builder, "Embedding")
        assert hasattr(builder, "Attention")
        assert hasattr(builder, "FFN")
        assert hasattr(builder, "TransformerLayer")


# ==============================================================================
# Model Tests
# ==============================================================================

class TestModel:
    """测试 Model 类."""
    
    def test_basic_creation(self):
        """测试基本创建."""
        model = Model("gpt2")
        assert model.name == "gpt2"
        assert model.block_type == "Transformer"
    
    def test_custom_module_type(self):
        """测试自定义模块类型."""
        model = Model("custom", module_type="CustomModule")
        assert model.block_type == "CustomModule"
    
    def test_metadata(self):
        """测试元数据设置."""
        model = Model("gpt2")
        model.metadata(batch_size=4, seq_len=2048, hidden=4096)
        
        assert model._metadata["batch_size"] == 4
        assert model._metadata["seq_len"] == 2048
        assert model._metadata["hidden"] == 4096
    
    def test_metadata_chaining(self):
        """测试元数据链式调用."""
        model = Model("gpt2")
        result = model.metadata(batch_size=4).metadata(seq_len=2048)
        
        assert result is model
        assert model._metadata["batch_size"] == 4
        assert model._metadata["seq_len"] == 2048
    
    def test_build_empty(self):
        """测试构建空模型."""
        model = Model("empty")
        graph = model.build()
        
        assert isinstance(graph, GraphIR)
        assert graph.name == "empty"
        assert graph.root is not None
        assert graph.root.name == "empty"
    
    def test_build_with_children(self):
        """测试构建带子节点的模型."""
        with Model("gpt2") as m:
            m.Linear("fc1", in_features=1024, out_features=4096)
            m.Linear("fc2", in_features=4096, out_features=1024)
        
        graph = m.build()
        
        assert graph.root is not None
        assert len(graph.root.children) == 2
        assert graph.root.children[0].name == "fc1"
        assert graph.root.children[1].name == "fc2"
    
    def test_build_with_metadata(self):
        """测试构建带元数据的模型."""
        with Model("gpt2") as m:
            m.metadata(batch_size=4, seq_len=2048)
        
        graph = m.build()
        
        assert graph.metadata["batch_size"] == 4
        assert graph.metadata["seq_len"] == 2048


class TestModelAliases:
    """测试模型便捷别名."""
    
    def test_transformer(self):
        """测试 Transformer 别名."""
        model = Transformer("gpt2")
        assert isinstance(model, Model)
        assert model.block_type == "Transformer"
    
    def test_gpt(self):
        """测试 GPT 别名."""
        model = GPT("gpt2")
        assert isinstance(model, Model)
        assert model.block_type == "GPT"
    
    def test_llama(self):
        """测试 LLaMA 别名."""
        model = LLaMA("llama_7b")
        assert isinstance(model, Model)
        assert model.block_type == "LLaMA"


# ==============================================================================
# DSL Usage Patterns
# ==============================================================================

class TestDSLUsagePatterns:
    """测试 DSL 使用模式."""
    
    def test_nested_context(self):
        """测试嵌套上下文."""
        with Transformer("model") as m:
            with m.TransformerLayer("layer0") as layer:
                with layer.Attention("attn") as attn:
                    attn.Linear("q_proj", in_features=1024, out_features=1024)
                    attn.Linear("k_proj", in_features=1024, out_features=1024)
                    attn.Linear("v_proj", in_features=1024, out_features=1024)
        
        graph = m.build()
        
        # 验证层次结构
        assert graph.root.name == "model"
        assert len(graph.root.children) == 1
        
        layer = graph.root.children[0]
        assert layer.name == "layer0"
        assert len(layer.children) == 1
        
        attn = layer.children[0]
        assert attn.name == "attn"
        assert len(attn.children) == 3
    
    def test_full_transformer_structure(self):
        """测试完整的 Transformer 结构."""
        with Transformer("llama") as m:
            m.metadata(batch_size=1, seq_len=2048, hidden=4096)
            
            m.Embedding("embed", num_embeddings=32000, embedding_dim=4096)
            
            for i in range(2):
                with m.TransformerLayer(f"layer{i}") as layer:
                    with layer.Attention("attn") as attn:
                        attn.RMSNorm("norm", normalized_shape=4096)
                        attn.Linear("q_proj", in_features=4096, out_features=4096)
                        attn.Linear("k_proj", in_features=4096, out_features=4096)
                        attn.Linear("v_proj", in_features=4096, out_features=4096)
                        attn.Linear("o_proj", in_features=4096, out_features=4096)
                    
                    with layer.FFN("ffn") as ffn:
                        ffn.RMSNorm("norm", normalized_shape=4096)
                        ffn.Linear("gate", in_features=4096, out_features=11008)
                        ffn.Linear("up", in_features=4096, out_features=11008)
                        ffn.Linear("down", in_features=11008, out_features=4096)
            
            m.RMSNorm("final_norm", normalized_shape=4096)
            m.Linear("lm_head", in_features=4096, out_features=32000)
        
        graph = m.build()
        
        # 验证基本结构
        assert graph.name == "llama"
        assert graph.metadata["hidden"] == 4096
        
        # 统计
        total_blocks = graph.count_blocks()
        params_blocks = graph.count_params_blocks()
        
        # 1 root + 1 embed + 2*(1 layer + 1 attn + 4 linear + 1 norm + 1 ffn + 3 linear + 1 norm) + 1 norm + 1 head
        # = 1 + 1 + 2*(1 + 1 + 4 + 1 + 1 + 3 + 1) + 1 + 1 = 1 + 1 + 24 + 2 = 28
        assert total_blocks == 28
        
        # 有参数的: embed, 2*(4 linear + norm + 3 linear + norm), final_norm, lm_head
        # = 1 + 2*(4 + 1 + 3 + 1) + 1 + 1 = 1 + 18 + 2 = 21
        assert params_blocks == 21


# ==============================================================================
# Print Tools Tests
# ==============================================================================

class TestPrintTools:
    """测试打印工具."""
    
    def test_print_graph(self):
        """测试 print_graph 函数."""
        with Model("test") as m:
            m.Linear("fc1", in_features=1024, out_features=4096)
            m.Linear("fc2", in_features=4096, out_features=1024)
        
        graph = m.build()
        output = print_graph(graph)
        
        assert "test" in output
        assert "Linear" in output
        assert "fc1" in output
        assert "fc2" in output
    
    def test_print_graph_verbose(self):
        """测试 verbose 模式."""
        with Model("test") as m:
            m.metadata(batch_size=4, seq_len=2048)
            m.Linear("fc1", in_features=1024, out_features=4096)
        
        graph = m.build()
        output = print_graph(graph, verbose=True)
        
        assert "metadata" in output
        assert "batch_size" in output
    
    def test_print_graph_shows_params(self):
        """测试显示参数信息."""
        with Model("test") as m:
            m.Linear("fc1", in_features=1024, out_features=4096)
        
        graph = m.build()
        output = print_graph(graph)
        
        assert "params" in output
        assert "weight" in output
    
    def test_graph_to_tree(self):
        """测试 graph_to_tree 函数."""
        with Model("test") as m:
            with m.TransformerLayer("layer0") as layer:
                layer.Attention("attn")
                layer.FFN("ffn")
        
        graph = m.build()
        tree = graph_to_tree(graph)
        
        assert "test" in tree
        assert "TransformerLayer" in tree
        assert "layer0" in tree
        assert "Attention" in tree
        assert "FFN" in tree
        
        # 检查树形结构字符
        assert "├──" in tree or "└──" in tree
    
    def test_graph_to_tree_with_attrs(self):
        """测试树形结构显示属性."""
        with Model("test") as m:
            m.Linear("fc1", in_features=1024, out_features=4096, shard="tp_col")
        
        graph = m.build()
        tree = graph_to_tree(graph)
        
        assert "in_features=1024" in tree
        assert "out_features=4096" in tree


# ==============================================================================
# Edge Cases
# ==============================================================================

class TestEdgeCases:
    """测试边界情况."""
    
    def test_empty_model(self):
        """测试空模型."""
        model = Model("empty")
        graph = model.build()
        
        assert graph.count_blocks() == 1  # 只有根节点
        assert graph.count_params_blocks() == 0
    
    def test_deeply_nested(self):
        """测试深层嵌套."""
        with Model("deep") as m:
            current = m
            for i in range(10):
                current = current.TransformerLayer(f"layer{i}")
        
        graph = m.build()
        assert graph.count_blocks() == 11  # 1 root + 10 layers
    
    def test_wide_model(self):
        """测试宽模型."""
        with Model("wide") as m:
            for i in range(100):
                m.Linear(f"fc{i}", in_features=1024, out_features=1024)
        
        graph = m.build()
        assert len(graph.root.children) == 100
    
    def test_special_names(self):
        """测试特殊名称."""
        with Model("test_model") as m:
            m.Linear("fc-1", in_features=1024, out_features=1024)
            m.Linear("fc.2", in_features=1024, out_features=1024)
            m.Linear("fc_3", in_features=1024, out_features=1024)
        
        graph = m.build()
        names = [c.name for c in graph.root.children]
        
        assert "fc-1" in names
        assert "fc.2" in names
        assert "fc_3" in names


# ==============================================================================
# Run tests
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

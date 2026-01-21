"""Tests for IRBuilder."""

import pytest
from sympy import Symbol, Expr

from blueprinting.ir.builder import (
    IRBuilder,
    build_transformer_block,
    build_transformer_model,
)
from blueprinting.ir.graph import GraphIR


class TestIRBuilder:
    """Tests for IRBuilder class."""
    
    def test_empty_builder(self):
        """Test empty builder."""
        builder = IRBuilder()
        graph = builder.build()
        
        assert isinstance(graph, GraphIR)
        assert len(graph.nodes) == 0
    
    def test_add_symbol(self):
        """Test symbol creation."""
        builder = IRBuilder()
        
        sym = builder.add_symbol("B", "batch size")
        
        assert sym == Symbol("B")
        assert "B" in builder._symbols
    
    def test_add_input(self):
        """Test input tensor creation."""
        builder = IRBuilder()
        B, S, H = Symbol("B"), Symbol("S"), Symbol("H")
        
        input_id = builder.add_input("x", shape=[B, S, H], dtype="float16")
        
        assert input_id == "input_x"
        assert "input_x" in builder._tensors
        assert builder._tensors["input_x"].shape == [B, S, H]
    
    def test_add_op(self):
        """Test operation creation."""
        builder = IRBuilder()
        
        # Add input first
        x = builder.add_input("x", shape=[4, 8, 16])
        
        # Add operation
        op_id = builder.add_op(
            name="linear_1",
            op_type="Linear",
            inputs=[x],
            attrs={"in_features": 16, "out_features": 32}
        )
        
        assert op_id == "linear_1"
        assert "linear_1" in builder._nodes
        assert builder._nodes["linear_1"].op_type == "Linear"
        assert builder._nodes["linear_1"].attrs["in_features"] == 16
    
    def test_add_linear(self):
        """Test Linear layer helper."""
        builder = IRBuilder()
        x = builder.add_input("x", shape=[4, 8, 16])
        
        linear_id = builder.add_linear(
            name="proj",
            inputs=[x],
            in_features=16,
            out_features=32,
            bias=True
        )
        
        assert linear_id == "proj"
        node = builder._nodes["proj"]
        assert node.op_type == "Linear"
        assert node.attrs["in_features"] == 16
        assert node.attrs["out_features"] == 32
        assert node.attrs["bias"] is True
    
    def test_add_rmsnorm(self):
        """Test RMSNorm layer helper."""
        builder = IRBuilder()
        x = builder.add_input("x", shape=[4, 8, 256])
        
        norm_id = builder.add_rmsnorm(
            name="norm",
            inputs=[x],
            normalized_shape=256
        )
        
        assert norm_id == "norm"
        node = builder._nodes["norm"]
        assert node.op_type == "RMSNorm"
        assert node.attrs["normalized_shape"] == 256
    
    def test_add_elementwise(self):
        """Test elementwise operations."""
        builder = IRBuilder()
        x = builder.add_input("x")
        y = builder.add_input("y")
        
        add_id = builder.add_elementwise("add", "Add", [x, y])
        mul_id = builder.add_elementwise("mul", "Mul", [x, y])
        silu_id = builder.add_elementwise("silu", "SiLU", [x])
        
        assert builder._nodes["add"].op_type == "Add"
        assert builder._nodes["mul"].op_type == "Mul"
        assert builder._nodes["silu"].op_type == "SiLU"
    
    def test_add_comm(self):
        """Test communication operations."""
        builder = IRBuilder()
        x = builder.add_input("x")
        
        ar_id = builder.add_comm("ar", "AllReduce", [x], num_peers=8)
        
        node = builder._nodes["ar"]
        assert node.op_type == "AllReduce"
        assert node.attrs["num_peers"] == 8
    
    def test_build_with_edges(self):
        """Test that edges are created correctly."""
        builder = IRBuilder()
        
        x = builder.add_input("x", shape=[4, 8, 16])
        l1 = builder.add_linear("l1", [x], 16, 32)
        l2 = builder.add_linear("l2", [l1], 32, 64)
        
        graph = builder.build()
        
        # Check edges exist
        assert len(graph.edges) >= 1
    
    def test_reset(self):
        """Test builder reset."""
        builder = IRBuilder()
        builder.add_input("x")
        builder.add_op("op1", "Linear", [])
        
        builder.reset()
        
        assert len(builder._nodes) == 0
        assert len(builder._tensors) == 0


class TestBuildTransformerBlock:
    """Tests for build_transformer_block function."""
    
    def test_basic_block(self):
        """Test basic transformer block construction."""
        H = Symbol("H")
        FF = Symbol("FF")
        
        graph = build_transformer_block(
            hidden=H,
            feedforward=FF,
            num_heads=32,
            head_dim=128,
            seq_len=2048,
            batch_size=4,
            block_id=0,
            tp=1
        )
        
        assert isinstance(graph, GraphIR)
        assert len(graph.nodes) > 0
        
        # Check expected operations exist
        op_types = {n.op_type for n in graph.nodes.values()}
        assert "Linear" in op_types
        assert "RMSNorm" in op_types
        assert "Add" in op_types
        assert "SiLU" in op_types
        assert "Mul" in op_types
        assert "Attention" in op_types
    
    def test_block_with_tp(self):
        """Test transformer block with tensor parallelism."""
        graph = build_transformer_block(
            hidden=4096,
            feedforward=11008,
            num_heads=32,
            head_dim=128,
            seq_len=2048,
            batch_size=4,
            block_id=0,
            tp=8
        )
        
        # Check that shard attributes are set
        sharded_nodes = [n for n in graph.nodes.values() 
                        if n.attrs.get("shard") is not None]
        assert len(sharded_nodes) > 0
    
    def test_block_metadata(self):
        """Test block metadata."""
        graph = build_transformer_block(
            hidden=256,
            feedforward=512,
            num_heads=8,
            head_dim=32,
            seq_len=128,
            batch_size=2,
            block_id=5,
            tp=4
        )
        
        assert graph.metadata["block_id"] == 5
        assert graph.metadata["tp"] == 4


class TestBuildTransformerModel:
    """Tests for build_transformer_model function."""
    
    def test_basic_model(self):
        """Test basic transformer model construction."""
        graph = build_transformer_model(
            num_layers=2,
            hidden=256,
            feedforward=512,
            num_heads=8,
            head_dim=32,
            seq_len=128,
            batch_size=2,
            tp=1,
            pp=1
        )
        
        assert isinstance(graph, GraphIR)
        assert len(graph.nodes) > 0
        assert graph.metadata["num_layers"] == 2
    
    def test_model_with_parallelism(self):
        """Test model with TP and PP."""
        graph = build_transformer_model(
            num_layers=4,
            hidden=256,
            feedforward=512,
            num_heads=8,
            head_dim=32,
            seq_len=128,
            batch_size=2,
            tp=2,
            pp=2
        )
        
        assert graph.metadata["tp"] == 2
        assert graph.metadata["pp"] == 2
        
        # With TP > 1, should have communication nodes
        comm_nodes = [n for n in graph.nodes.values() 
                     if n.op_type == "AllReduce"]
        assert len(comm_nodes) > 0
    
    def test_model_with_symbols(self):
        """Test model with symbolic dimensions."""
        B = Symbol("B")
        S = Symbol("S")
        H = Symbol("H")
        
        graph = build_transformer_model(
            num_layers=2,
            hidden=H,
            feedforward=4*H,
            num_heads=32,
            head_dim=H/32,
            seq_len=S,
            batch_size=B,
            tp=1,
            pp=1
        )
        
        # Graph should be built with symbolic dimensions
        assert len(graph.nodes) > 0
        # At least FF should be tracked as a symbol (4*H)
        assert len(graph.symbols) > 0 or any(
            isinstance(n.attrs.get("batch_seq"), Expr) 
            for n in graph.nodes.values()
        )

"""Tests for GraphIR and OpNode."""

import pytest
from sympy import Symbol, Expr

from blueprinting.ir.graph import GraphIR, OpNode, TensorRef


class TestTensorRef:
    """Tests for TensorRef class."""
    
    def test_basic_creation(self):
        """Test basic TensorRef creation."""
        tensor = TensorRef(id="t1", shape=[4, 8, 16], dtype="float16")
        assert tensor.id == "t1"
        assert tensor.shape == [4, 8, 16]
        assert tensor.dtype == "float16"
    
    def test_nbytes_numeric(self):
        """Test nbytes calculation with numeric shape."""
        tensor = TensorRef(id="t1", shape=[4, 8, 16], dtype="float16")
        # 4 * 8 * 16 * 2 bytes = 1024
        assert tensor.nbytes == 1024
    
    def test_nbytes_symbolic(self):
        """Test nbytes calculation with symbolic shape."""
        B, S, H = Symbol("B"), Symbol("S"), Symbol("H")
        tensor = TensorRef(id="t1", shape=[B, S, H], dtype="float16")
        # B * S * H * 2
        expected = B * S * H * 2
        assert tensor.nbytes == expected
    
    def test_nbytes_different_dtypes(self):
        """Test nbytes for different data types."""
        shape = [2, 4]
        
        assert TensorRef("t", shape, "float8").nbytes == 8
        assert TensorRef("t", shape, "float16").nbytes == 16
        assert TensorRef("t", shape, "bfloat16").nbytes == 16
        assert TensorRef("t", shape, "float32").nbytes == 32
        assert TensorRef("t", shape, "float64").nbytes == 64


class TestOpNode:
    """Tests for OpNode class."""
    
    def test_basic_creation(self):
        """Test basic OpNode creation."""
        node = OpNode(
            id="linear_1",
            op_type="Linear",
            inputs=["input"],
            outputs=["output"],
            attrs={"in_features": 256, "out_features": 512}
        )
        assert node.id == "linear_1"
        assert node.op_type == "Linear"
        assert node.inputs == ["input"]
        assert node.attrs["in_features"] == 256
    
    def test_hash_and_equality(self):
        """Test OpNode hashing and equality."""
        node1 = OpNode(id="n1", op_type="Linear")
        node2 = OpNode(id="n1", op_type="RMSNorm")  # Same ID
        node3 = OpNode(id="n2", op_type="Linear")  # Different ID
        
        assert node1 == node2  # Same ID
        assert node1 != node3  # Different ID
        assert hash(node1) == hash(node2)
    
    def test_workload_attributes(self):
        """Test workload attribute assignment."""
        node = OpNode(id="n1", op_type="Linear")
        
        # Initially None
        assert node.flops is None
        assert node.memory_fw is None
        
        # Can assign
        node.flops = 1000
        node.memory_fw = 2000
        assert node.flops == 1000
        assert node.memory_fw == 2000


class TestGraphIR:
    """Tests for GraphIR class."""
    
    def test_empty_graph(self):
        """Test empty graph creation."""
        graph = GraphIR()
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0
        assert len(graph.symbols) == 0
    
    def test_add_node(self):
        """Test adding nodes to graph."""
        graph = GraphIR()
        node = OpNode(id="n1", op_type="Linear")
        
        graph.add_node(node)
        
        assert "n1" in graph.nodes
        assert graph.get_node("n1") == node
    
    def test_add_edge(self):
        """Test adding edges to graph."""
        graph = GraphIR()
        graph.add_node(OpNode(id="n1", op_type="Linear"))
        graph.add_node(OpNode(id="n2", op_type="RMSNorm"))
        
        graph.add_edge("n1", "n2")
        
        assert ("n1", "n2") in graph.edges
    
    def test_get_symbol(self):
        """Test symbol management."""
        graph = GraphIR()
        
        # First call creates symbol
        sym1 = graph.get_symbol("B")
        assert sym1 == Symbol("B")
        assert "B" in graph.symbols
        
        # Second call returns same symbol
        sym2 = graph.get_symbol("B")
        assert sym1 is sym2
    
    def test_topological_sort_linear(self):
        """Test topological sort on linear graph."""
        graph = GraphIR()
        graph.add_node(OpNode(id="n1", op_type="Input"))
        graph.add_node(OpNode(id="n2", op_type="Linear"))
        graph.add_node(OpNode(id="n3", op_type="Output"))
        graph.add_edge("n1", "n2")
        graph.add_edge("n2", "n3")
        
        order = graph.topological_sort()
        
        # n1 must come before n2, n2 before n3
        assert order.index("n1") < order.index("n2")
        assert order.index("n2") < order.index("n3")
    
    def test_topological_sort_diamond(self):
        """Test topological sort on diamond graph."""
        graph = GraphIR()
        #     n1
        #    /  \
        #   n2  n3
        #    \  /
        #     n4
        graph.add_node(OpNode(id="n1", op_type="Input"))
        graph.add_node(OpNode(id="n2", op_type="Linear"))
        graph.add_node(OpNode(id="n3", op_type="Linear"))
        graph.add_node(OpNode(id="n4", op_type="Add"))
        graph.add_edge("n1", "n2")
        graph.add_edge("n1", "n3")
        graph.add_edge("n2", "n4")
        graph.add_edge("n3", "n4")
        
        order = graph.topological_sort()
        
        # n1 must come first, n4 must come last
        assert order[0] == "n1"
        assert order[-1] == "n4"
    
    def test_get_predecessors_successors(self):
        """Test predecessor and successor queries."""
        graph = GraphIR()
        graph.add_node(OpNode(id="n1", op_type="Input"))
        graph.add_node(OpNode(id="n2", op_type="Linear"))
        graph.add_node(OpNode(id="n3", op_type="Linear"))
        graph.add_edge("n1", "n2")
        graph.add_edge("n1", "n3")
        
        assert graph.get_predecessors("n2") == ["n1"]
        assert graph.get_predecessors("n1") == []
        assert set(graph.get_successors("n1")) == {"n2", "n3"}
    
    def test_total_flops(self):
        """Test total FLOPs calculation."""
        graph = GraphIR()
        
        node1 = OpNode(id="n1", op_type="Linear")
        node1.flops = 1000
        
        node2 = OpNode(id="n2", op_type="Linear")
        node2.flops = 2000
        
        graph.add_node(node1)
        graph.add_node(node2)
        
        assert graph.total_flops() == 3000
    
    def test_copy(self):
        """Test graph copying."""
        graph = GraphIR()
        graph.add_node(OpNode(id="n1", op_type="Linear"))
        graph.metadata["key"] = "value"
        
        copy = graph.copy()
        
        assert "n1" in copy.nodes
        assert copy.metadata["key"] == "value"
        
        # Modify copy shouldn't affect original
        copy.metadata["key"] = "modified"
        assert graph.metadata["key"] == "value"

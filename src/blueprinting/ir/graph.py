"""GraphIR - Computation graph representation for IR compiler.

This module defines the core IR data structures for representing
neural network computation graphs with symbolic expressions.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import Counter

from sympy import Expr, Symbol


@dataclass
class TensorRef:
    """Reference to a tensor in the graph.
    
    Attributes:
        id: Unique identifier for the tensor
        shape: Symbolic shape expression
        dtype: Data type string (e.g., "float16", "float32")
        producer: ID of the op that produces this tensor
    """
    id: str
    shape: Optional[List[Union[int, Expr]]] = None
    dtype: str = "float16"
    producer: Optional[str] = None
    
    @property
    def nbytes(self) -> Expr:
        """Calculate number of bytes for this tensor."""
        from functools import reduce
        from operator import mul
        
        dtype_sizes = {
            "float8": 1,
            "float16": 2,
            "bfloat16": 2,
            "float32": 4,
            "float64": 8,
        }
        if self.shape is None:
            return 0
        return reduce(mul, self.shape, 1) * dtype_sizes.get(self.dtype, 2)


@dataclass
class OpNode:
    """A node in the computation graph representing an operation.
    
    Attributes:
        id: Unique identifier for this operation
        op_type: Type of operation (e.g., "Linear", "RMSNorm", "AllReduce")
        inputs: List of input tensor/node IDs
        attrs: Operation-specific attributes
        
    Workload information (filled by WorkloadPass):
        flops: FLOPs for this operation (forward + backward)
        flops_fw: Forward FLOPs
        flops_bw: Backward FLOPs
        memory_fw: Memory accessed during forward
        memory_bw: Memory accessed during backward
        comm_bytes: Communication bytes (forward + backward)
        weight_bytes: Weight tensor bytes
        activation_bytes: Activation tensor bytes
        
    Parallel information (filled by ParallelPass):
        shard: Sharding strategy ("tp_col", "tp_row", "replicated", etc.)
        device: Device assignment
    """
    id: str
    op_type: str
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    attrs: Dict[str, Any] = field(default_factory=dict)
    
    # Workload information (filled by WorkloadPass)
    flops: Optional[Expr] = None
    flops_fw: Optional[Expr] = None
    flops_bw: Optional[Expr] = None
    flops_agrad: Optional[Expr] = None  # Activation gradient FLOPs (dX = dY @ W^T)
    flops_wgrad: Optional[Expr] = None  # Weight gradient FLOPs (dW = X^T @ dY)
    memory_fw: Optional[Expr] = None
    memory_bw: Optional[Expr] = None
    memory_agrad: Optional[Expr] = None  # Memory for agrad computation
    memory_wgrad: Optional[Expr] = None  # Memory for wgrad computation
    comm_bytes: Optional[Expr] = None
    comm_bytes_fw: Optional[Expr] = None
    comm_bytes_bw: Optional[Expr] = None
    weight_bytes: Optional[Expr] = None
    activation_bytes: Optional[Expr] = None
    
    # Parallel information (filled by ParallelPass)
    shard: Optional[str] = None
    device: Optional[int] = None
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if isinstance(other, OpNode):
            return self.id == other.id
        return False


@dataclass
class GraphIR:
    """Intermediate representation of a computation graph.
    
    Attributes:
        nodes: Dictionary mapping node ID to OpNode
        tensors: Dictionary mapping tensor ID to TensorRef
        edges: List of (source, destination) edges
        symbols: Symbol table mapping names to SymPy Symbols
        metadata: Additional graph-level metadata
    """
    nodes: Dict[str, OpNode] = field(default_factory=dict)
    tensors: Dict[str, TensorRef] = field(default_factory=dict)
    edges: List[Tuple[str, str]] = field(default_factory=list)
    symbols: Dict[str, Symbol] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_node(self, node: OpNode) -> "GraphIR":
        """Add a node to the graph."""
        self.nodes[node.id] = node
        return self
    
    def add_tensor(self, tensor: TensorRef) -> "GraphIR":
        """Add a tensor reference to the graph."""
        self.tensors[tensor.id] = tensor
        return self
    
    def add_edge(self, src: str, dst: str) -> "GraphIR":
        """Add an edge from src to dst."""
        self.edges.append((src, dst))
        return self
    
    def get_node(self, node_id: str) -> Optional[OpNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)
    
    def get_tensor(self, tensor_id: str) -> Optional[TensorRef]:
        """Get a tensor by ID."""
        return self.tensors.get(tensor_id)
    
    def get_symbol(self, name: str) -> Symbol:
        """Get or create a symbol by name."""
        if name not in self.symbols:
            self.symbols[name] = Symbol(name)
        return self.symbols[name]
    
    def topological_sort(self) -> List[str]:
        """Return nodes in topological order."""
        # Build adjacency list
        adj: Dict[str, List[str]] = {nid: [] for nid in self.nodes}
        in_degree: Dict[str, int] = {nid: 0 for nid in self.nodes}
        
        for src, dst in self.edges:
            if src in adj and dst in self.nodes:
                adj[src].append(dst)
                in_degree[dst] += 1
        
        # Kahn's algorithm
        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        result = []
        
        while queue:
            node = queue.pop(0)
            result.append(node)
            for neighbor in adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return result
    
    def get_predecessors(self, node_id: str) -> List[str]:
        """Get all predecessor node IDs."""
        return [src for src, dst in self.edges if dst == node_id]
    
    def get_successors(self, node_id: str) -> List[str]:
        """Get all successor node IDs."""
        return [dst for src, dst in self.edges if src == node_id]
    
    def total_flops(self) -> Expr:
        """Sum of all node FLOPs."""
        total = 0
        for node in self.nodes.values():
            if node.flops is not None:
                total = total + node.flops
        return total
    
    def total_weight_bytes(self) -> Expr:
        """Sum of all weight bytes."""
        total = 0
        for node in self.nodes.values():
            if node.weight_bytes is not None:
                total = total + node.weight_bytes
        return total
    
    def copy(self) -> "GraphIR":
        """Create a shallow copy of the graph."""
        import copy
        return GraphIR(
            nodes=copy.copy(self.nodes),
            tensors=copy.copy(self.tensors),
            edges=copy.copy(self.edges),
            symbols=copy.copy(self.symbols),
            metadata=copy.copy(self.metadata),
        )
    
    def __repr__(self) -> str:
        parts = [
            f"nodes={len(self.nodes)}",
            f"edges={len(self.edges)}",
            f"tensors={len(self.tensors)}",
        ]
        op_types = Counter(node.op_type for node in self.nodes.values())
        if op_types:
            top = ", ".join(f"{k}:{v}" for k, v in op_types.most_common(5))
            parts.append(f"op_types={{ {top} }}")
        meta_keys = list(self.metadata.keys())
        if meta_keys:
            sample = ", ".join(meta_keys[:6])
            suffix = "..." if len(meta_keys) > 6 else ""
            parts.append(f"metadata_keys=[{sample}{suffix}]")
        return f"GraphIR({', '.join(parts)})"

    def summary(self) -> str:
        """Return a readable multi-line summary of the graph."""
        op_types = Counter(node.op_type for node in self.nodes.values())
        top_ops = "\n".join(
            f"  - {k}: {v}" for k, v in op_types.most_common(8)
        ) if op_types else "  (none)"
        meta_keys = list(self.metadata.keys())
        meta_line = ", ".join(meta_keys) if meta_keys else "(none)"
        return (
            "GraphIR Summary\n"
            f"- nodes: {len(self.nodes)}\n"
            f"- edges: {len(self.edges)}\n"
            f"- tensors: {len(self.tensors)}\n"
            f"- op_types:\n{top_ops}\n"
            f"- metadata_keys: {meta_line}"
        )

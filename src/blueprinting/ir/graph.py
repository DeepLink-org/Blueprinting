"""GraphIR - Hierarchical computation graph representation for IR compiler.

This module defines a hierarchical IR structure similar to PyTorch's nn.Module,
with Module → Block → Op representing the computation tree.

Key design principles:
1. Pure tree hierarchy (no explicit edges)
2. Data flow expressed through tensor names (inputs/outputs binding)
3. Execution order determined by children ordering
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Iterator, Callable
from collections import Counter
from enum import Enum

from sympy import Expr, Symbol


class NodeType(Enum):
    """Types of nodes in the hierarchical graph."""
    MODULE = "module"  # Top-level container (e.g., Transformer model)
    BLOCK = "block"    # Named block (e.g., TransformerLayer, Attention, FFN)
    OP = "op"          # Leaf operation (e.g., Linear, RMSNorm, Softmax)


@dataclass
class TensorRef:
    """Reference to a tensor in the graph.
    
    Tensors are the data flow mechanism - ops consume and produce tensors by name.
    
    Attributes:
        name: Tensor name (used for binding inputs/outputs)
        shape: Symbolic shape expression
        dtype: Data type string (e.g., "float16", "float32")
        producer: Path to the op that produces this tensor (e.g., "layer0.attn.q_proj")
    """
    name: str
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
    """A leaf node representing an actual computation.
    
    Ops are the only nodes that do actual work. They consume input tensors
    and produce output tensors. Data dependencies are implicit through
    tensor name binding.
    
    Attributes:
        name: Op name (local within parent block)
        op_type: Type of operation (e.g., "Linear", "RMSNorm", "AllReduce")
        inputs: List of input tensor names
        outputs: List of output tensor names
        attrs: Operation-specific attributes
        
    Workload information (filled by WorkloadPass):
        flops_fw, flops_bw: Forward/backward FLOPs
        memory_fw, memory_bw: Memory accessed
        weight_bytes: Weight tensor bytes
        activation_bytes: Activation bytes
        
    Parallel information (filled by ParallelPass):
        shard: Sharding strategy
        device: Device assignment
    """
    name: str
    op_type: str
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    attrs: Dict[str, Any] = field(default_factory=dict)
    
    # Workload information
    flops: Optional[Expr] = None
    flops_fw: Optional[Expr] = None
    flops_bw: Optional[Expr] = None
    flops_agrad: Optional[Expr] = None
    flops_wgrad: Optional[Expr] = None
    memory_fw: Optional[Expr] = None
    memory_bw: Optional[Expr] = None
    memory_agrad: Optional[Expr] = None
    memory_wgrad: Optional[Expr] = None
    comm_bytes: Optional[Expr] = None
    comm_bytes_fw: Optional[Expr] = None
    comm_bytes_bw: Optional[Expr] = None
    weight_bytes: Optional[Expr] = None
    activation_bytes: Optional[Expr] = None
    
    # Parallel information
    shard: Optional[str] = None
    device: Optional[int] = None
    
    @property
    def node_type(self) -> NodeType:
        return NodeType.OP
    
    def __repr__(self) -> str:
        parts = [f"{self.op_type}({self.name!r})"]
        if self.inputs:
            parts.append(f"in={self.inputs}")
        if self.outputs:
            parts.append(f"out={self.outputs}")
        if self.shard:
            parts.append(f"shard={self.shard}")
        return f"Op({', '.join(parts)})"


@dataclass
class BlockNode:
    """A named block containing ops or sub-blocks.
    
    Blocks represent logical groupings like:
    - TransformerLayer (contains Attention + FFN)
    - Attention (contains Q/K/V projections, attention compute, output proj)
    - FFN (contains FC1, activation, FC2)
    
    Attributes:
        name: Block name (local within parent)
        block_type: Type of block (e.g., "Attention", "FFN", "TransformerLayer")
        children: Ordered list of child nodes (ops or sub-blocks)
        attrs: Block-specific attributes (e.g., hidden_size, num_heads)
    """
    name: str
    block_type: str
    children: List[Union["BlockNode", OpNode]] = field(default_factory=list)
    attrs: Dict[str, Any] = field(default_factory=dict)
    
    # Parallel information (applies to all children unless overridden)
    device: Optional[int] = None
    
    @property
    def node_type(self) -> NodeType:
        return NodeType.BLOCK
    
    def add_child(self, child: Union["BlockNode", OpNode]) -> "BlockNode":
        """Add a child node."""
        self.children.append(child)
        return self
    
    def add_op(self, name: str, op_type: str, inputs: List[str] = None, 
               outputs: List[str] = None, **attrs) -> OpNode:
        """Add an op as a child and return it."""
        op = OpNode(
            name=name,
            op_type=op_type,
            inputs=inputs or [],
            outputs=outputs or [f"{name}_out"],
            attrs=attrs,
        )
        self.children.append(op)
        return op
    
    def add_block(self, name: str, block_type: str, **attrs) -> "BlockNode":
        """Add a sub-block as a child and return it."""
        block = BlockNode(name=name, block_type=block_type, attrs=attrs)
        self.children.append(block)
        return block
    
    def iter_ops(self) -> Iterator[OpNode]:
        """Iterate over all ops in this block (recursively)."""
        for child in self.children:
            if isinstance(child, OpNode):
                yield child
            elif isinstance(child, BlockNode):
                yield from child.iter_ops()
    
    def iter_children(self) -> Iterator[Union["BlockNode", OpNode]]:
        """Iterate over direct children."""
        return iter(self.children)
    
    def find_child(self, name: str) -> Optional[Union["BlockNode", OpNode]]:
        """Find a direct child by name."""
        for child in self.children:
            if child.name == name:
                return child
        return None
    
    def __repr__(self) -> str:
        op_count = sum(1 for _ in self.iter_ops())
        return f"Block({self.block_type}({self.name!r}), children={len(self.children)}, ops={op_count})"


@dataclass  
class ModuleNode:
    """Root module containing the model structure.
    
    This is the top-level container representing the entire model.
    
    Attributes:
        name: Module name (usually the model name)
        module_type: Type of module (e.g., "Transformer", "GPT")
        children: Ordered list of child blocks (e.g., layers)
        inputs: Model input tensor names
        outputs: Model output tensor names
        attrs: Module-level attributes (e.g., num_layers, hidden_size)
    """
    name: str
    module_type: str
    children: List[BlockNode] = field(default_factory=list)
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    attrs: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def node_type(self) -> NodeType:
        return NodeType.MODULE
    
    def add_block(self, name: str, block_type: str, **attrs) -> BlockNode:
        """Add a block as a child and return it."""
        block = BlockNode(name=name, block_type=block_type, attrs=attrs)
        self.children.append(block)
        return block
    
    def iter_blocks(self) -> Iterator[BlockNode]:
        """Iterate over all blocks (recursively)."""
        def _iter(children):
            for child in children:
                yield child
                if isinstance(child, BlockNode):
                    yield from _iter(child.children)
        
        for child in self.children:
            yield child
            if hasattr(child, 'children'):
                for sub in child.children:
                    if isinstance(sub, BlockNode):
                        yield sub
                        yield from _iter(sub.children)
    
    def iter_ops(self) -> Iterator[OpNode]:
        """Iterate over all ops in the module (recursively)."""
        for child in self.children:
            yield from child.iter_ops()
    
    def __repr__(self) -> str:
        block_count = len(self.children)
        op_count = sum(1 for _ in self.iter_ops())
        return f"Module({self.module_type}({self.name!r}), blocks={block_count}, ops={op_count})"


@dataclass
class GraphIR:
    """Hierarchical intermediate representation of a computation graph.
    
    The graph is a tree with:
    - Root: ModuleNode (the model)
    - Branches: BlockNodes (layers, attention, ffn, etc.)
    - Leaves: OpNodes (linear, norm, activation, etc.)
    
    Data flow is expressed through tensor name binding, not explicit edges.
    Execution order is determined by children ordering within each block.
    
    Attributes:
        root: The root module
        tensors: Dictionary of all tensors by name
        symbols: Symbol table for symbolic values
        metadata: Graph-level metadata
    """
    root: Optional[ModuleNode] = None
    tensors: Dict[str, TensorRef] = field(default_factory=dict)
    symbols: Dict[str, Symbol] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # === Tensor management ===
    
    def add_tensor(self, name: str, shape: List = None, dtype: str = "float16",
                   producer: str = None) -> TensorRef:
        """Register a tensor."""
        tensor = TensorRef(name=name, shape=shape, dtype=dtype, producer=producer)
        self.tensors[name] = tensor
        return tensor
    
    def get_tensor(self, name: str) -> Optional[TensorRef]:
        """Get a tensor by name."""
        return self.tensors.get(name)
    
    def get_symbol(self, name: str) -> Symbol:
        """Get or create a symbol by name."""
        if name not in self.symbols:
            self.symbols[name] = Symbol(name)
        return self.symbols[name]
    
    # === Traversal ===
    
    def iter_ops(self) -> Iterator[OpNode]:
        """Iterate over all ops in the graph."""
        if self.root:
            yield from self.root.iter_ops()
    
    def iter_blocks(self) -> Iterator[BlockNode]:
        """Iterate over all blocks in the graph."""
        if self.root:
            yield from self.root.iter_blocks()
    
    def iter_ops_with_path(self) -> Iterator[tuple]:
        """Iterate over (path, op) pairs.
        
        Yields:
            (path, op) where path is like "layer0.attn.q_proj"
        """
        def _walk(node, path_parts):
            if isinstance(node, OpNode):
                yield (".".join(path_parts + [node.name]), node)
            elif isinstance(node, (BlockNode, ModuleNode)):
                for child in node.children:
                    yield from _walk(child, path_parts + [node.name])
        
        if self.root:
            yield from _walk(self.root, [])
    
    def find_by_path(self, path: str) -> Optional[Union[ModuleNode, BlockNode, OpNode]]:
        """Find a node by its path (e.g., 'model.layer0.attn.q_proj')."""
        parts = path.split(".")
        if not self.root or not parts:
            return None
        
        if parts[0] != self.root.name:
            return None
        
        current: Union[ModuleNode, BlockNode, OpNode] = self.root
        for part in parts[1:]:
            if isinstance(current, OpNode):
                return None  # Ops have no children
            found = None
            for child in current.children:
                if child.name == part:
                    found = child
                    break
            if found is None:
                return None
            current = found
        
        return current
    
    # === Aggregate computations ===
    
    def total_flops(self) -> Expr:
        """Sum of all op FLOPs."""
        total = 0
        for op in self.iter_ops():
            if op.flops is not None:
                total = total + op.flops
        return total
    
    def total_weight_bytes(self) -> Expr:
        """Sum of all weight bytes."""
        total = 0
        for op in self.iter_ops():
            if op.weight_bytes is not None:
                total = total + op.weight_bytes
        return total
    
    def count_ops(self) -> int:
        """Count total ops."""
        return sum(1 for _ in self.iter_ops())
    
    def count_blocks(self) -> int:
        """Count total blocks."""
        return sum(1 for _ in self.iter_blocks())
    
    # === Copy ===
    
    def copy(self) -> "GraphIR":
        """Create a deep copy of the graph."""
        import copy
        return GraphIR(
            root=copy.deepcopy(self.root),
            tensors=copy.deepcopy(self.tensors),
            symbols=copy.copy(self.symbols),
            metadata=copy.deepcopy(self.metadata),
        )
    
    # === Display ===
    
    def __repr__(self) -> str:
        if not self.root:
            return "GraphIR(empty)"
        op_count = self.count_ops()
        block_count = self.count_blocks()
        op_types = Counter(op.op_type for op in self.iter_ops())
        top_types = ", ".join(f"{k}:{v}" for k, v in op_types.most_common(4))
        return f"GraphIR({self.root.module_type}, blocks={block_count}, ops={op_count}, types={{ {top_types} }})"
    
    def summary(self) -> str:
        """Return a readable multi-line summary."""
        if not self.root:
            return "GraphIR(empty)"
        
        lines = [
            f"GraphIR: {self.root.module_type}({self.root.name})",
            f"├─ blocks: {self.count_blocks()}",
            f"├─ ops: {self.count_ops()}",
            f"├─ tensors: {len(self.tensors)}",
        ]
        
        # Op type breakdown
        op_types = Counter(op.op_type for op in self.iter_ops())
        if op_types:
            lines.append("├─ op_types:")
            for k, v in op_types.most_common(8):
                lines.append(f"│   {k}: {v}")
        
        # Top-level structure
        lines.append("├─ structure:")
        for child in self.root.children[:8]:
            child_ops = sum(1 for _ in child.iter_ops())
            lines.append(f"│   {child.block_type}({child.name}): {child_ops} ops")
        if len(self.root.children) > 8:
            lines.append(f"│   ... ({len(self.root.children) - 8} more)")
        
        # Metadata
        meta = ", ".join(list(self.metadata.keys())[:6])
        lines.append(f"└─ metadata: {meta or '(none)'}")
        
        return "\n".join(lines)
    
    def tree(self, max_depth: int = 3) -> str:
        """Return a tree representation of the graph structure."""
        if not self.root:
            return "(empty)"
        
        lines = []
        
        def _tree(node, depth, prefix=""):
            if depth > max_depth:
                return
            
            if isinstance(node, ModuleNode):
                lines.append(f"{prefix}■ {node.module_type}({node.name})")
                for i, child in enumerate(node.children):
                    is_last = (i == len(node.children) - 1)
                    child_prefix = prefix + ("└─ " if is_last else "├─ ")
                    next_prefix = prefix + ("   " if is_last else "│  ")
                    _tree(child, depth + 1, child_prefix)
                    if i < len(node.children) - 1:
                        lines[-1] = lines[-1]  # Keep formatting
            
            elif isinstance(node, BlockNode):
                op_count = sum(1 for _ in node.iter_ops())
                lines.append(f"{prefix}□ {node.block_type}({node.name}) [{op_count} ops]")
                if depth < max_depth:
                    for i, child in enumerate(node.children):
                        is_last = (i == len(node.children) - 1)
                        child_prefix = prefix + ("   └─ " if is_last else "   ├─ ")
                        next_prefix = prefix + ("      " if is_last else "   │  ")
                        _tree(child, depth + 1, child_prefix)
            
            elif isinstance(node, OpNode):
                shard_info = f" [{node.shard}]" if node.shard else ""
                lines.append(f"{prefix}◆ {node.op_type}({node.name}){shard_info}")
        
        _tree(self.root, 0)
        return "\n".join(lines)


# === Legacy compatibility layer ===
# These allow gradual migration from the old flat GraphIR

class LegacyGraphIRAdapter:
    """Adapter to provide legacy GraphIR interface on top of new hierarchical IR.
    
    This allows existing passes to work with minimal changes during migration.
    """
    
    def __init__(self, graph: GraphIR):
        self._graph = graph
        self._ops_cache: Optional[Dict[str, OpNode]] = None
        self._edges_cache: Optional[List[tuple]] = None
    
    @property
    def nodes(self) -> Dict[str, OpNode]:
        """Legacy: flat dict of all ops by their path."""
        if self._ops_cache is None:
            self._ops_cache = {}
            for path, op in self._graph.iter_ops_with_path():
                self._ops_cache[path] = op
        return self._ops_cache
    
    @property
    def edges(self) -> List[tuple]:
        """Legacy: inferred edges from tensor name binding."""
        if self._edges_cache is None:
            self._edges_cache = self._infer_edges()
        return self._edges_cache
    
    def _infer_edges(self) -> List[tuple]:
        """Infer dependency edges from tensor name binding."""
        edges = []
        # Build producer map: tensor_name -> op_path
        producers = {}
        for path, op in self._graph.iter_ops_with_path():
            for out in op.outputs:
                producers[out] = path
        
        # Build edges: for each op's input, find its producer
        for path, op in self._graph.iter_ops_with_path():
            for inp in op.inputs:
                if inp in producers:
                    edges.append((producers[inp], path))
        
        return edges
    
    @property
    def tensors(self) -> Dict[str, TensorRef]:
        return self._graph.tensors
    
    @property
    def symbols(self) -> Dict[str, Symbol]:
        return self._graph.symbols
    
    @property
    def metadata(self) -> Dict[str, Any]:
        return self._graph.metadata
    
    def get_node(self, node_id: str) -> Optional[OpNode]:
        return self.nodes.get(node_id)
    
    def topological_sort(self) -> List[str]:
        """Return op paths in topological order."""
        nodes = self.nodes
        edges = self.edges
        
        # Build adjacency
        adj = {nid: [] for nid in nodes}
        in_degree = {nid: 0 for nid in nodes}
        
        for src, dst in edges:
            if src in adj and dst in nodes:
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

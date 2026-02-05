"""DSL - Block-based Graph construction.

基于 types.py 中定义的 BlockNode 和 GraphIR 构建模型图。

使用方式:
=========

```python
from blueprinting.ir.dsl import Model

# 构建 GPT-2 模型
with Model("gpt2") as m:
    with m.Layer("layer0") as layer:
        with layer.Attention("attn") as attn:
            attn.RMSNorm("input_norm")
            attn.Linear("q_proj", shard="tp_col")
            attn.Linear("k_proj", shard="tp_col")
            attn.Linear("v_proj", shard="tp_col")
            attn.Linear("o_proj", shard="tp_row")

        with layer.FFN("ffn") as ffn:
            ffn.RMSNorm("input_norm")
            ffn.Linear("gate_proj", shard="tp_col")
            ffn.Linear("up_proj", shard="tp_col")
            ffn.Linear("down_proj", shard="tp_row")

# 构建 Graph IR
graph = m.build()
print(graph)
```

设计原则:
=========
1. Graph IR 只包含 Block，不包含 Op
2. Block 通过 BlockDef.__call__ 方法在 Schedule 阶段展开成 Op
3. DSL 使用 with 语句构建层次结构
4. 通过 ops.py 中的 BlockDef 注册表自动生成方法
5. BlockNode 直接关联 BlockDef，可访问参数定义和计算方法
"""

from __future__ import annotations

from typing import Any

from .ops import list_blocks
from .types import BlockNode, GraphIR


class BlockBuilder:
    """Block 构建器 - DSL 的核心类.

    每个 BlockBuilder 对应一个 BlockNode，
    支持通过方法调用添加子 Block。

    方法由 ops.py 中的 BlockDef 注册表自动生成。
    """

    def __init__(
        self, name: str, block_type: str, parent: BlockBuilder | None = None
    ):
        self.name = name
        self.block_type = block_type
        self.parent = parent
        self._children: list[BlockBuilder] = []
        self._attrs: dict[str, Any] = {}

    def __enter__(self) -> BlockBuilder:
        return self

    def __exit__(self, *args):
        pass

    def _add_child(self, name: str, block_type: str, **attrs) -> BlockBuilder:
        """添加子 Block."""
        child = BlockBuilder(name, block_type, parent=self)
        child._attrs = attrs
        self._children.append(child)
        return child

    def _to_block_node(self) -> BlockNode:
        """转换为 BlockNode."""
        node = BlockNode(
            name=self.name,
            block_type=self.block_type,
            attrs=self._attrs.copy(),
        )
        for child in self._children:
            node.children.append(child._to_block_node())
        return node

    def __repr__(self) -> str:
        return f"BlockBuilder({self.block_type}({self.name!r}), children={len(self._children)})"


# ==============================================================================
# 动态方法生成
# ==============================================================================


def _create_block_method(block_type: str):
    """为指定的 block_type 创建方法."""

    def method(self: BlockBuilder, name: str, **attrs) -> BlockBuilder:
        return self._add_child(name, block_type, **attrs)

    method.__name__ = block_type
    method.__doc__ = f"添加 {block_type} Block."
    return method


def _register_block_methods(cls):
    """为 BlockBuilder 注册所有 Block 方法."""
    for block_type in list_blocks():
        if not hasattr(cls, block_type):
            setattr(cls, block_type, _create_block_method(block_type))
    return cls


# 注册方法
_register_block_methods(BlockBuilder)


# ==============================================================================
# Model 入口
# ==============================================================================


class Model(BlockBuilder):
    """模型构建器 - DSL 入口.

    Usage:
        with Model("gpt2") as m:
            m.metadata(batch_size=4, seq_len=2048, hidden=4096)
            with m.Layer("layer0") as layer:
                ...

        graph = m.build()
    """

    def __init__(self, name: str, module_type: str = "Transformer"):
        super().__init__(name, module_type, parent=None)
        self._metadata: dict[str, Any] = {}

    def metadata(self, **kwargs) -> Model:
        """设置模型元数据."""
        self._metadata.update(kwargs)
        return self

    def build(self) -> GraphIR:
        """构建 GraphIR."""
        root = self._to_block_node()
        return GraphIR(
            name=self.name,
            root=root,
            metadata=self._metadata.copy(),
        )


# ==============================================================================
# 便捷别名
# ==============================================================================


def Transformer(name: str) -> Model:
    """创建 Transformer 模型."""
    return Model(name, "Transformer")


def GPT(name: str) -> Model:
    """创建 GPT 模型."""
    return Model(name, "GPT")


def LLaMA(name: str) -> Model:
    """创建 LLaMA 模型."""
    return Model(name, "LLaMA")


# ==============================================================================
# 打印工具
# ==============================================================================


def print_graph(graph: GraphIR, verbose: bool = False) -> str:
    """打印 Graph IR 结构."""
    lines = []
    lines.append(f"GraphIR: {graph.name}")
    if graph.metadata and verbose:
        lines.append(f"  metadata: {graph.metadata}")
    lines.append("")

    def print_block(block: BlockNode, level: int = 0):
        prefix = "  " * level
        attrs_str = ""
        if block.attrs:
            attrs_str = (
                f" {{{', '.join(f'{k}={v!r}' for k, v in block.attrs.items())}}}"
            )

        # 显示参数信息
        params_str = ""
        if block.has_params:
            params_str = f" [params: {block.params}]"

        lines.append(
            f"{prefix}{block.block_type}({block.name!r}){attrs_str}{params_str}"
        )
        for child in block.children:
            print_block(child, level + 1)

    if graph.root:
        print_block(graph.root)

    return "\n".join(lines)


def graph_to_tree(graph: GraphIR) -> str:
    """将 Graph IR 转换为 ASCII 树形表示."""
    lines = []

    def tree_block(block: BlockNode, prefix: str = "", is_last: bool = True):
        connector = "└── " if is_last else "├── "
        attrs_str = ""
        if block.attrs:
            key_attrs = {
                k: v
                for k, v in block.attrs.items()
                if k in ("shard", "in_features", "out_features")
            }
            if key_attrs:
                attrs_str = f" [{', '.join(f'{k}={v}' for k, v in key_attrs.items())}]"

        lines.append(f"{prefix}{connector}{block.block_type}({block.name}){attrs_str}")

        child_prefix = prefix + ("    " if is_last else "│   ")
        for i, child in enumerate(block.children):
            tree_block(child, child_prefix, i == len(block.children) - 1)

    if graph.root:
        lines.append(f"{graph.root.block_type}({graph.root.name})")
        for i, child in enumerate(graph.root.children):
            tree_block(child, "", i == len(graph.root.children) - 1)

    return "\n".join(lines)


# ==============================================================================
# 便捷构建函数
# ==============================================================================


def build_transformer_model(
    model_name: str,
    num_layers: int,
    hidden: int,
    feedforward: int,
    num_heads: int,
    head_dim: int,
    seq_len: int = 2048,
    batch_size: int = 1,
    tp: int = 1,
    pp: int = 1,
    dp: int = 1,
    gradient_checkpointing: bool = False,
    activation: str = "GELU",
) -> GraphIR:
    """构建 Transformer 模型的 GraphIR.

    Args:
        model_name: 模型名称
        num_layers: 层数
        hidden: 隐藏维度
        feedforward: FFN 中间维度
        num_heads: 注意力头数
        head_dim: 每个头的维度
        seq_len: 序列长度
        batch_size: 微批次大小
        tp: 张量并行度
        pp: 流水线并行度
        dp: 数据并行度
        gradient_checkpointing: 是否使用梯度检查点
        activation: 激活函数 ("GELU", "SiLU")

    Returns:
        GraphIR
    """
    batch_seq = batch_size * seq_len

    with Transformer(model_name) as m:
        m.metadata(
            model_name=model_name,
            num_layers=num_layers,
            hidden=hidden,
            feedforward=feedforward,
            num_heads=num_heads,
            head_dim=head_dim,
            batch_size=batch_size,
            seq_len=seq_len,
            tp=tp,
            pp=pp,
            dp=dp,
            batch_seq=batch_seq,
            gradient_checkpointing=gradient_checkpointing,
        )

        for i in range(num_layers):
            with m.TransformerLayer(f"layer{i}") as layer:
                # Attention block
                with layer.Attention("attn") as attn:
                    attn.RMSNorm("norm", normalized_shape=hidden)
                    attn.Linear("q_proj", in_features=hidden, out_features=hidden, shard="tp_col" if tp > 1 else None)
                    attn.Linear("k_proj", in_features=hidden, out_features=hidden, shard="tp_col" if tp > 1 else None)
                    attn.Linear("v_proj", in_features=hidden, out_features=hidden, shard="tp_col" if tp > 1 else None)
                    attn.Linear("out_proj", in_features=hidden, out_features=hidden, shard="tp_row" if tp > 1 else None)

                # FFN block
                with layer.FFN("ffn") as ffn:
                    ffn.RMSNorm("norm", normalized_shape=hidden)
                    ffn.Linear("up_proj", in_features=hidden, out_features=feedforward, shard="tp_col" if tp > 1 else None)
                    # 激活函数
                    if activation == "SiLU":
                        ffn.SiLU("act")
                    else:
                        ffn.GELU("act")
                    ffn.Linear("down_proj", in_features=feedforward, out_features=hidden, shard="tp_row" if tp > 1 else None)

    return m.build()


# ==============================================================================
# 导出
# ==============================================================================

__all__ = [
    "Model",
    "Transformer",
    "GPT",
    "LLaMA",
    "BlockBuilder",
    "print_graph",
    "graph_to_tree",
    "build_transformer_model",
]

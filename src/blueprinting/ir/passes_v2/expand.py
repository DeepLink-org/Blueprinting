"""ExpandPass - Expand GraphIR (Block) to ScheduleIR (Op).

这个 Pass 实现了纯粹的 Block → Op 展开:
1. 遍历 GraphIR 中的所有 Block
2. 根据 Block 类型生成对应的 Op 序列
3. 不计算 workload 和 duration（由 SchedulePass 负责）

职责分离:
- ExpandPass: 纯展开，生成 Op 结构
- SchedulePass: 计算 workload、duration、设备分配
"""

from typing import Any, Dict, List, Optional

from .base import Pass
from ..types import (
    GraphIR, BlockNode,
    ScheduleIR, OpNode, ScheduledOp,
)
from ..ops import get_block_def


class ExpandContext:
    """展开上下文 - 在 Block 展开时收集 Op.
    
    只负责创建 Op 结构，不计算 workload 和 duration。
    """
    
    def __init__(
        self,
        block: BlockNode,
        device: int = 0,
        stage: int = 0,
        metadata: Optional[Dict] = None,
        block_path: Optional[str] = None,
    ):
        self.block = block
        self.device = device
        self.stage = stage
        self.metadata = metadata or {}
        
        # Block 路径 (用于追踪 Op 来源)
        if block_path:
            self.block_path = block_path
        else:
            self.block_path = block.name
        
        self.ops: List[ScheduledOp] = []
        self._op_counter: int = 0
        
        # 当前正在展开的 Block (用于嵌套展开)
        self._current_block: BlockNode = block
        self._current_path: str = self.block_path
    
    def _set_current_block(self, block: BlockNode, path: str):
        """设置当前正在展开的 Block."""
        self._current_block = block
        self._current_path = path
    
    def _create_op(
        self,
        name: str,
        op_type: str,
        inputs: List[str] = None,
        outputs: List[str] = None,
        attrs: Dict[str, Any] = None,
    ) -> "TensorRef":
        """创建一个 Op（不计算 workload 和 duration）."""
        # source_block 使用当前正在展开的 Block 的路径
        source = f"{self._current_path}/{self._current_block.block_type}({self._current_block.name})"
        
        op = OpNode(
            name=name,
            op_type=op_type,
            inputs=inputs or [],
            outputs=outputs or [f"{name}_out"],
            attrs=attrs or {},
            source_block=source,
            # workload 字段由 SchedulePass 填充
            flops=None,
            memory_bytes=None,
            comm_bytes=None,
        )
        
        sched_op = ScheduledOp(
            op=op,
            device=self.device,
            stage=self.stage,
            stream="comm" if op_type in ("AllReduce", "AllGather", "ReduceScatter", "Send", "Recv") else "compute",
            # timing 由 SchedulePass 填充
            start=0,
            duration=0,
        )
        
        self.ops.append(sched_op)
        self._op_counter += 1
        
        return TensorRef(op.outputs[0])
    
    # ==== Op 生成方法 ====
    
    def Matmul(self, a, b, M=1, K=1, N=1, **attrs) -> "TensorRef":
        """生成矩阵乘法 Op."""
        return self._create_op(
            name=f"mm_{self._op_counter}",
            op_type="Matmul",
            attrs={"M": M, "K": K, "N": N, **attrs},
        )
    
    def Add(self, a, b, num_elements=1, **attrs) -> "TensorRef":
        """生成加法 Op."""
        return self._create_op(
            name=f"add_{self._op_counter}",
            op_type="Add",
            attrs={"num_elements": num_elements, **attrs},
        )
    
    def Mul(self, a, b, num_elements=1, **attrs) -> "TensorRef":
        """生成乘法 Op."""
        return self._create_op(
            name=f"mul_{self._op_counter}",
            op_type="Mul",
            attrs={"num_elements": num_elements, **attrs},
        )
    
    def Softmax(self, x, num_elements=1, **attrs) -> "TensorRef":
        """生成 Softmax Op."""
        return self._create_op(
            name=f"softmax_{self._op_counter}",
            op_type="Softmax",
            attrs={"num_elements": num_elements, **attrs},
        )
    
    def SiLU(self, x, num_elements=1, **attrs) -> "TensorRef":
        """生成 SiLU 激活 Op."""
        return self._create_op(
            name=f"silu_{self._op_counter}",
            op_type="SiLU",
            attrs={"num_elements": num_elements, **attrs},
        )
    
    def GELU(self, x, num_elements=1, **attrs) -> "TensorRef":
        """生成 GELU 激活 Op."""
        return self._create_op(
            name=f"gelu_{self._op_counter}",
            op_type="GELU",
            attrs={"num_elements": num_elements, **attrs},
        )
    
    def RMSNorm(self, x, normalized_shape=1, **attrs) -> "TensorRef":
        """生成 RMSNorm Op."""
        return self._create_op(
            name=f"rmsnorm_{self._op_counter}",
            op_type="RMSNorm",
            attrs={"normalized_shape": normalized_shape, **attrs},
        )
    
    def LayerNorm(self, x, normalized_shape=1, **attrs) -> "TensorRef":
        """生成 LayerNorm Op."""
        return self._create_op(
            name=f"layernorm_{self._op_counter}",
            op_type="LayerNorm",
            attrs={"normalized_shape": normalized_shape, **attrs},
        )
    
    def AllReduce(self, x, data_size=1, num_peers=8, **attrs) -> "TensorRef":
        """生成 AllReduce 通信 Op."""
        return self._create_op(
            name=f"allreduce_{self._op_counter}",
            op_type="AllReduce",
            attrs={"data_size": data_size, "num_peers": num_peers, **attrs},
        )
    
    def AllGather(self, x, data_size=1, num_peers=8, **attrs) -> "TensorRef":
        """生成 AllGather 通信 Op."""
        return self._create_op(
            name=f"allgather_{self._op_counter}",
            op_type="AllGather",
            attrs={"data_size": data_size, "num_peers": num_peers, **attrs},
        )
    
    def ReduceScatter(self, x, data_size=1, num_peers=8, **attrs) -> "TensorRef":
        """生成 ReduceScatter 通信 Op."""
        return self._create_op(
            name=f"reducescatter_{self._op_counter}",
            op_type="ReduceScatter",
            attrs={"data_size": data_size, "num_peers": num_peers, **attrs},
        )
    
    def Send(self, x, data_size=1, dst_rank=0, **attrs) -> "TensorRef":
        """生成 Send 通信 Op."""
        return self._create_op(
            name=f"send_{self._op_counter}",
            op_type="Send",
            attrs={"data_size": data_size, "dst_rank": dst_rank, **attrs},
        )
    
    def Recv(self, data_size=1, src_rank=0, **attrs) -> "TensorRef":
        """生成 Recv 通信 Op."""
        return self._create_op(
            name=f"recv_{self._op_counter}",
            op_type="Recv",
            attrs={"data_size": data_size, "src_rank": src_rank, **attrs},
        )


class TensorRef:
    """简单的张量引用."""
    def __init__(self, name: str):
        self.name = name
    
    def __repr__(self):
        return f"Tensor({self.name})"


# ==============================================================================
# 默认 Block 展开逻辑
# ==============================================================================

def _expand_linear(block: BlockNode, x: TensorRef, ctx: ExpandContext, path: str) -> TensorRef:
    """展开 Linear Block."""
    ctx._set_current_block(block, path)
    
    in_f = block.attrs.get("in_features", 4096)
    out_f = block.attrs.get("out_features", 4096)
    batch_seq = ctx.metadata.get("batch_size", 1) * ctx.metadata.get("seq_len", 2048)
    
    y = ctx.Matmul(x, None, M=batch_seq, K=in_f, N=out_f)
    
    # 如果有 shard，插入通信
    shard = block.attrs.get("shard")
    if shard == "tp_row":
        tp = ctx.metadata.get("tp", 8)
        data_size = batch_seq * out_f
        y = ctx.AllReduce(y, data_size=data_size, num_peers=tp)
    
    return y


def _expand_rmsnorm(block: BlockNode, x: TensorRef, ctx: ExpandContext, path: str) -> TensorRef:
    """展开 RMSNorm Block."""
    ctx._set_current_block(block, path)
    hidden = block.attrs.get("normalized_shape", ctx.metadata.get("hidden", 4096))
    return ctx.RMSNorm(x, normalized_shape=hidden)


def _expand_layernorm(block: BlockNode, x: TensorRef, ctx: ExpandContext, path: str) -> TensorRef:
    """展开 LayerNorm Block."""
    ctx._set_current_block(block, path)
    hidden = block.attrs.get("normalized_shape", ctx.metadata.get("hidden", 4096))
    return ctx.LayerNorm(x, normalized_shape=hidden)


def _expand_embedding(block: BlockNode, x: TensorRef, ctx: ExpandContext, path: str) -> TensorRef:
    """展开 Embedding Block."""
    ctx._set_current_block(block, path)
    # Embedding 不产生计算 Op (只是查表)
    return x


def _expand_container(block: BlockNode, x: TensorRef, ctx: ExpandContext, path: str) -> TensorRef:
    """展开容器 Block (Attention, FFN, Layer) - 递归展开子 Block."""
    for child in block.children:
        child_path = f"{path}/{block.block_type}({block.name})"
        x = expand_block(child, x, ctx, child_path)
    return x


# Block 类型到展开函数的映射
_EXPAND_FUNCS = {
    "Linear": _expand_linear,
    "RMSNorm": _expand_rmsnorm,
    "LayerNorm": _expand_layernorm,
    "Embedding": _expand_embedding,
    "Attention": _expand_container,
    "FFN": _expand_container,
    "MLP": _expand_container,
    "TransformerLayer": _expand_container,
    "Layer": _expand_container,
    "Transformer": _expand_container,
    "GPT": _expand_container,
    "LLaMA": _expand_container,
}


def expand_block(block: BlockNode, x: TensorRef, ctx: ExpandContext, path: str = "") -> TensorRef:
    """展开单个 Block 成 Op 序列."""
    expand_func = _EXPAND_FUNCS.get(block.block_type)
    
    if expand_func:
        return expand_func(block, x, ctx, path)
    else:
        # 默认：递归展开子 Block
        for child in block.children:
            child_path = f"{path}/{block.block_type}({block.name})"
            x = expand_block(child, x, ctx, child_path)
        return x


# ==============================================================================
# ExpandPass
# ==============================================================================

class ExpandPass(Pass):
    """将 GraphIR (Block) 展开成 ScheduleIR (Op).
    
    这个 Pass 只负责纯粹的展开:
    - 遍历 GraphIR 中的所有 Block
    - 根据 Block 类型生成对应的 Op 序列
    - 不计算 workload 和 duration（由 SchedulePass 负责）
    
    输入: GraphIR (Block 级别)
    输出: ScheduleIR (Op 级别，无 workload 和 timing)
    """
    
    def __init__(self, device: int = 0, stage: int = 0):
        self.device = device
        self.stage = stage
    
    def run(self, ir: GraphIR) -> ScheduleIR:
        schedule = ScheduleIR(
            num_devices=1,
            metadata=ir.metadata.copy(),
        )
        
        if ir.root:
            ctx = ExpandContext(
                block=ir.root,
                device=self.device,
                stage=self.stage,
                metadata=ir.metadata,
            )
            
            # 展开整个模型
            x = TensorRef("input")
            expand_block(ir.root, x, ctx)
            
            # 添加所有 Op 到 ScheduleIR
            for op in ctx.ops:
                schedule.add_op(op, stage=self.stage, device=self.device)
        
        return schedule

"""Inference Passes - Passes specialized for inference (prefill/decode).

推理场景与训练的核心区别:
1. 只有 forward pass，没有 backward 和 optimizer
2. Attention 计算在 generation 阶段使用 KV-cache，与训练完全不同
3. 内存模型不同：需要考虑 KV-cache 内存而非激活/梯度/优化器状态
4. 没有 PipelineSchedulePass（不需要 micro-batch 交错调度）

推理有两个阶段:
- Context (Prefill): 一次性处理所有输入 token，全量注意力
- Generation (Decode): 逐 token 生成，使用 KV-cache，memory-bandwidth bound

参数管理:
- InferenceParallelPass / InferenceExpandPass: @hp.param("parallel")
- InferenceSchedulePass: @hp.param("parallel")

IR Pipeline (推理):
    GraphIR (Block)
         │
         ↓ InferenceParallelPass (标记并行策略 + 推理模式)
         │
         ↓ InferenceExpandPass (Block → Op，区分 context/generation)
         │
    ScheduleIR (Op, 无 workload)
         │
         ↓ InferenceSchedulePass (计算 workload + timing + KV-cache 内存)
         │
    ScheduleIR (Op, 有 workload)
         │
         ↓ TimelinePass (Op → Event，复用训练的 TimelinePass)
         │
    TimelineIR (Event)
         │
         ↓ SimulatePass (纯观测，不区分训练/推理)
         │
    SimulationResult
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import hyperparameter as hp
from sympy import Expr

from blueprinting.core import SymMax

from ..types import BlockNode, GraphIR, MemoryPool, OpNode, Phase, ScheduledOp, ScheduleIR
from .base import Pass


# ============================================================================
# QuantConfig
# ============================================================================


@dataclass
class QuantConfig:
    """推理量化配置.

    控制推理中各组件的数据类型精度，影响内存占用和计算吞吐。

    参考 aiconfigurator 的量化模式:
    - GEMM weight: float16(2B), fp8(1B), int8(1B), int4(0.5B), nvfp4(0.5B)
    - KV-cache: float16(2B), fp8(1B), int8(1B)
    - 通信: half(2B), fp8(1B), int8(1B)

    量化如何影响性能:
    - weight_bytes: 降低权重内存 → Matmul 中读取权重的内存带宽时间减少
    - activation_bytes: 降低激活内存 → 所有 Op 的中间结果内存减少
    - kv_cache_bytes: 降低 KV-cache 内存 → 可以服务更长序列或更大 batch
    - comm_bytes: 降低通信数据量 → AllReduce 等通信 Op 时间减少
    - compute_multiplier: 量化 tensor core 通常有更高吞吐
      (例如 H100 FP8 = 2x FP16, B200 NVFP4 = 4x FP16)

    Attributes:
        weight_bytes: 权重存储精度 (bytes per element)
        activation_bytes: 激活/中间结果精度 (bytes per element)
        kv_cache_bytes: KV-cache 存储精度 (bytes per element)
        comm_bytes: 通信数据精度 (bytes per element)
        compute_multiplier: GEMM 计算吞吐倍数 (相对于 fp16 baseline)
    """

    weight_bytes: float = 2.0
    activation_bytes: float = 2.0
    kv_cache_bytes: float = 2.0
    comm_bytes: float = 2.0
    compute_multiplier: float = 1.0

    @staticmethod
    def fp16() -> "QuantConfig":
        """全 FP16 配置 (默认)."""
        return QuantConfig()

    @staticmethod
    def fp8() -> "QuantConfig":
        """FP8 量化: weight fp8, activation fp8, KV-cache fp8.

        H100/B100 上 FP8 tensor core 约为 FP16 的 2 倍吞吐。
        """
        return QuantConfig(
            weight_bytes=1.0,
            activation_bytes=1.0,
            kv_cache_bytes=1.0,
            comm_bytes=2.0,  # 通信通常保持 fp16
            compute_multiplier=2.0,
        )

    @staticmethod
    def w8a16() -> "QuantConfig":
        """INT8 weight-only 量化: weight int8, activation fp16.

        权重内存减半，但计算吞吐不变 (dequant to fp16 再计算)。
        """
        return QuantConfig(
            weight_bytes=1.0,
            activation_bytes=2.0,
            kv_cache_bytes=2.0,
            comm_bytes=2.0,
            compute_multiplier=1.0,
        )

    @staticmethod
    def w4a16() -> "QuantConfig":
        """INT4 weight-only 量化: weight int4, activation fp16.

        权重内存减为 1/4，但计算吞吐不变。
        对 generation 阶段的 memory-bandwidth bound GEMM 效果显著。
        """
        return QuantConfig(
            weight_bytes=0.5,
            activation_bytes=2.0,
            kv_cache_bytes=2.0,
            comm_bytes=2.0,
            compute_multiplier=1.0,
        )

    @staticmethod
    def nvfp4() -> "QuantConfig":
        """NVFP4 量化 (Blackwell): weight nvfp4, activation fp8.

        B200 上 NVFP4 tensor core 约为 FP16 的 4 倍吞吐。
        """
        return QuantConfig(
            weight_bytes=0.5,
            activation_bytes=1.0,
            kv_cache_bytes=1.0,
            comm_bytes=2.0,
            compute_multiplier=4.0,
        )

    @staticmethod
    def fp8_kv_fp8() -> "QuantConfig":
        """FP8 计算 + FP8 KV-cache (常见推理配置)."""
        return QuantConfig(
            weight_bytes=1.0,
            activation_bytes=1.0,
            kv_cache_bytes=1.0,
            comm_bytes=2.0,
            compute_multiplier=2.0,
        )

    @staticmethod
    def fp8_kv_int8() -> "QuantConfig":
        """FP8 计算 + INT8 KV-cache."""
        return QuantConfig(
            weight_bytes=1.0,
            activation_bytes=1.0,
            kv_cache_bytes=1.0,
            comm_bytes=2.0,
            compute_multiplier=2.0,
        )


# ============================================================================
# InferenceParallelPass
# ============================================================================


class InferenceParallelPass(Pass):
    """推理专用的并行策略 Pass.

    与训练 ParallelPass 类似，但:
    - 标记推理模式 (mode="inference") 和阶段 (phase="context"/"generation")
    - DP 含义不同: 推理中 DP 是请求级并行，不涉及梯度通信
    - TP/PP 策略复用训练的 shard 标记逻辑

    职责:
    - 标记 Block 的 shard 策略 (tp_col, tp_row)
    - 分配 PP stage
    - 更新 metadata (mode, phase, tp, pp)
    """

    @hp.param("parallel")
    def __init__(
        self,
        tp: int = 1,
        pp: int = 1,
        phase: str = "context",  # "context" 或 "generation"
        tp_comm_type: str = "ar",
    ):
        """初始化 InferenceParallelPass.

        Args:
            tp: Tensor Parallelism 度数
            pp: Pipeline Parallelism 度数
            phase: 推理阶段 ("context" = prefill, "generation" = decode)
            tp_comm_type: TP 通信类型 ("ar" = AllReduce, "rs_ag" = ReduceScatter + AllGather)
        """
        self.tp = tp
        self.pp = pp
        self.phase = phase
        self.tp_comm_type = tp_comm_type

    def run(self, ir: GraphIR) -> GraphIR:
        """执行推理并行策略标记."""
        # 更新 metadata
        ir.metadata["mode"] = "inference"
        ir.metadata["phase"] = self.phase
        ir.metadata["tp"] = self.tp
        ir.metadata["pp"] = self.pp
        ir.metadata["dp"] = 1  # 推理不使用训练的 DP
        ir.metadata["tp_comm_type"] = self.tp_comm_type

        if ir.root:
            # 应用 TP 策略 (复用训练的 shard 标记逻辑)
            if self.tp > 1:
                self._apply_tensor_parallel(ir.root)

            # 应用 PP 策略
            if self.pp > 1:
                self._apply_pipeline_parallel(ir)

        return ir

    def _apply_tensor_parallel(self, block: BlockNode) -> None:
        """递归应用 TP 策略到 Block 树."""
        self._mark_tp_shard(block)
        for child in block.children:
            self._apply_tensor_parallel(child)

    def _mark_tp_shard(self, block: BlockNode) -> None:
        """标记单个 Block 的 shard 策略 (与训练一致)."""
        block_type = block.block_type

        if block.attrs.get("shard"):
            return

        if block_type == "Linear":
            name = block.name.lower()

            # QKV projection: column parallel
            if any(
                x in name
                for x in ["qkv", "q_proj", "k_proj", "v_proj", "query", "key", "value"]
            ):
                block.attrs["shard"] = "tp_col"
            # Output projection: row parallel
            elif any(x in name for x in ["out", "o_proj", "output", "dense"]):
                block.attrs["shard"] = "tp_row"
            # FFN: up/gate column, down row
            elif any(x in name for x in ["up", "gate", "fc1", "w1", "w3"]):
                block.attrs["shard"] = "tp_col"
            elif any(x in name for x in ["down", "fc2", "w2"]):
                block.attrs["shard"] = "tp_row"

    def _apply_pipeline_parallel(self, ir: GraphIR) -> None:
        """应用 PP 策略，分配 stage."""
        if not ir.root:
            return

        num_layers = ir.metadata.get("num_layers")
        if num_layers is None:
            num_layers = self._count_layers(ir.root)

        if num_layers == 0:
            return

        if num_layers % self.pp != 0:
            raise ValueError(
                f"Pipeline parallelism requires num_layers to be divisible by pp. "
                f"Got num_layers={num_layers}, pp={self.pp}."
            )

        layers_per_stage = num_layers // self.pp
        self._assign_stages(ir.root, layers_per_stage)

    def _count_layers(self, root: BlockNode) -> int:
        """统计 TransformerLayer 的数量."""
        count = 0
        for child in root.children:
            if child.block_type in ("TransformerLayer", "Layer"):
                count += 1
            elif child.block_type in ("Transformer", "GPT", "LLaMA"):
                count += self._count_layers(child)
        return count

    def _assign_stages(self, block: BlockNode, layers_per_stage: int) -> None:
        """递归分配 stage."""
        for child in block.children:
            if child.block_type in ("TransformerLayer", "Layer"):
                layer_idx = self._extract_layer_index(child.name)
                if layer_idx is not None:
                    stage = min(layer_idx // layers_per_stage, self.pp - 1)
                    child.attrs["stage"] = stage
                    child.attrs["device"] = stage
            self._assign_stages(child, layers_per_stage)

    def _extract_layer_index(self, name: str) -> Optional[int]:
        """从名称提取层索引."""
        match = re.search(r"layer(\d+)", name, re.IGNORECASE)
        if match:
            return int(match.group(1))
        match = re.search(r"block(\d+)", name, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None


# ============================================================================
# InferenceExpandPass
# ============================================================================


class _InferenceTensorRef:
    """推理展开中的简单张量引用."""

    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return f"Tensor({self.name})"


class _InferenceExpandContext:
    """推理展开上下文.

    与训练的 ExpandContext 类似，但支持根据推理阶段
    (context/generation) 不同地展开 Attention。
    """

    def __init__(
        self,
        block: BlockNode,
        phase: str = "context",
        device: int = 0,
        stage: int = 0,
        metadata: Optional[Dict] = None,
        block_path: Optional[str] = None,
    ):
        self.block = block
        self.phase = phase  # "context" 或 "generation"
        self.device = device
        self.stage = stage
        self.metadata = metadata or {}

        if block_path:
            self.block_path = block_path
        else:
            self.block_path = block.name

        self.ops: List[ScheduledOp] = []
        self._op_counter: int = 0
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
        inputs: Optional[List[str]] = None,
        outputs: Optional[List[str]] = None,
        attrs: Optional[Dict[str, Any]] = None,
    ) -> _InferenceTensorRef:
        """创建一个 Op."""
        source = f"{self._current_path}/{self._current_block.block_type}({self._current_block.name})"

        op = OpNode(
            name=name,
            op_type=op_type,
            inputs=inputs or [],
            outputs=outputs or [f"{name}_out"],
            attrs=attrs or {},
            source_block=source,
            flops=None,
            memory_bytes=None,
            comm_bytes=None,
        )

        sched_op = ScheduledOp(
            op=op,
            device=self.device,
            stage=self.stage,
            stream=(
                "comm"
                if op_type in ("AllReduce", "AllGather", "ReduceScatter", "Send", "Recv")
                else "compute"
            ),
            phase=Phase.FORWARD,  # 推理只有 forward
            start=0,
            duration=0,
        )

        self.ops.append(sched_op)
        self._op_counter += 1

        return _InferenceTensorRef(op.outputs[0])

    # ==== Op 生成方法 ====

    def Matmul(self, a, b, M=1, K=1, N=1, **attrs) -> _InferenceTensorRef:
        """生成矩阵乘法 Op."""
        return self._create_op(
            name=f"mm_{self._op_counter}",
            op_type="Matmul",
            attrs={"M": M, "K": K, "N": N, **attrs},
        )

    def Softmax(self, x, num_elements=1, **attrs) -> _InferenceTensorRef:
        """生成 Softmax Op."""
        return self._create_op(
            name=f"softmax_{self._op_counter}",
            op_type="Softmax",
            attrs={"num_elements": num_elements, **attrs},
        )

    def SiLU(self, x, num_elements=1, **attrs) -> _InferenceTensorRef:
        """生成 SiLU 激活 Op."""
        return self._create_op(
            name=f"silu_{self._op_counter}",
            op_type="SiLU",
            attrs={"num_elements": num_elements, **attrs},
        )

    def GELU(self, x, num_elements=1, **attrs) -> _InferenceTensorRef:
        """生成 GELU 激活 Op."""
        return self._create_op(
            name=f"gelu_{self._op_counter}",
            op_type="GELU",
            attrs={"num_elements": num_elements, **attrs},
        )

    def RMSNorm(self, x, normalized_shape=1, **attrs) -> _InferenceTensorRef:
        """生成 RMSNorm Op."""
        return self._create_op(
            name=f"rmsnorm_{self._op_counter}",
            op_type="RMSNorm",
            attrs={"normalized_shape": normalized_shape, **attrs},
        )

    def LayerNorm(self, x, normalized_shape=1, **attrs) -> _InferenceTensorRef:
        """生成 LayerNorm Op."""
        return self._create_op(
            name=f"layernorm_{self._op_counter}",
            op_type="LayerNorm",
            attrs={"normalized_shape": normalized_shape, **attrs},
        )

    def AllReduce(self, x, data_size=1, num_peers=8, **attrs) -> _InferenceTensorRef:
        """生成 AllReduce 通信 Op."""
        return self._create_op(
            name=f"allreduce_{self._op_counter}",
            op_type="AllReduce",
            attrs={"data_size": data_size, "num_peers": num_peers, **attrs},
        )


# ---- 展开函数 ----


def _infer_expand_linear(
    block: BlockNode,
    x: _InferenceTensorRef,
    ctx: _InferenceExpandContext,
    path: str,
) -> _InferenceTensorRef:
    """展开 Linear Block (推理).

    与训练的展开逻辑相同: 根据 shard 策略调整 Matmul 维度,
    tp_row 时插入 AllReduce。
    """
    ctx._set_current_block(block, path)

    in_f = block.attrs.get("in_features", 4096)
    out_f = block.attrs.get("out_features", 4096)
    shard = block.attrs.get("shard")
    tp = max(1, ctx.metadata.get("tp", 1))

    # 推理的 M 维度:
    # context: batch_size * seq_len (处理所有输入 token)
    # generation: batch_size (每次只处理 1 个 token)
    if ctx.phase == "generation":
        M = ctx.metadata.get("batch_size", 1)
    else:
        M = ctx.metadata.get("batch_size", 1) * ctx.metadata.get("seq_len", 2048)

    K = in_f
    N = out_f
    if shard == "tp_col":
        N = out_f // tp
    elif shard == "tp_row":
        K = in_f // tp

    y = ctx.Matmul(x, None, M=M, K=K, N=N)

    if shard == "tp_row":
        data_size = M * out_f
        y = ctx.AllReduce(y, data_size=data_size, num_peers=tp)

    return y


def _infer_expand_rmsnorm(
    block: BlockNode,
    x: _InferenceTensorRef,
    ctx: _InferenceExpandContext,
    path: str,
) -> _InferenceTensorRef:
    """展开 RMSNorm Block (推理)."""
    ctx._set_current_block(block, path)
    hidden = block.attrs.get("normalized_shape", ctx.metadata.get("hidden", 4096))
    return ctx.RMSNorm(x, normalized_shape=hidden)


def _infer_expand_layernorm(
    block: BlockNode,
    x: _InferenceTensorRef,
    ctx: _InferenceExpandContext,
    path: str,
) -> _InferenceTensorRef:
    """展开 LayerNorm Block (推理)."""
    ctx._set_current_block(block, path)
    hidden = block.attrs.get("normalized_shape", ctx.metadata.get("hidden", 4096))
    return ctx.LayerNorm(x, normalized_shape=hidden)


def _infer_expand_gelu(
    block: BlockNode,
    x: _InferenceTensorRef,
    ctx: _InferenceExpandContext,
    path: str,
) -> _InferenceTensorRef:
    """展开 GELU Block (推理)."""
    ctx._set_current_block(block, path)
    feedforward = ctx.metadata.get("feedforward", ctx.metadata.get("hidden", 4096) * 4)
    tp = max(1, ctx.metadata.get("tp", 1))

    if ctx.phase == "generation":
        num_tokens = ctx.metadata.get("batch_size", 1)
    else:
        num_tokens = ctx.metadata.get("batch_size", 1) * ctx.metadata.get("seq_len", 2048)

    num_elements = block.attrs.get("num_elements", num_tokens * feedforward // tp)
    return ctx.GELU(x, num_elements=num_elements)


def _infer_expand_silu(
    block: BlockNode,
    x: _InferenceTensorRef,
    ctx: _InferenceExpandContext,
    path: str,
) -> _InferenceTensorRef:
    """展开 SiLU Block (推理)."""
    ctx._set_current_block(block, path)
    feedforward = ctx.metadata.get("feedforward", ctx.metadata.get("hidden", 4096) * 4)
    tp = max(1, ctx.metadata.get("tp", 1))

    if ctx.phase == "generation":
        num_tokens = ctx.metadata.get("batch_size", 1)
    else:
        num_tokens = ctx.metadata.get("batch_size", 1) * ctx.metadata.get("seq_len", 2048)

    num_elements = block.attrs.get("num_elements", num_tokens * feedforward // tp)
    return ctx.SiLU(x, num_elements=num_elements)


def _infer_expand_embedding(
    block: BlockNode,
    x: _InferenceTensorRef,
    ctx: _InferenceExpandContext,
    path: str,
) -> _InferenceTensorRef:
    """展开 Embedding Block (推理) — 查表操作，不产生计算 Op."""
    ctx._set_current_block(block, path)
    return x


def _infer_expand_container(
    block: BlockNode,
    x: _InferenceTensorRef,
    ctx: _InferenceExpandContext,
    path: str,
) -> _InferenceTensorRef:
    """展开容器 Block (Attention, FFN, Layer 等) — 递归展开子 Block."""
    for child in block.children:
        child_path = f"{path}/{block.block_type}({block.name})"
        x = _infer_expand_block(child, x, ctx, child_path)
    return x


def _infer_expand_attention(
    block: BlockNode,
    x: _InferenceTensorRef,
    ctx: _InferenceExpandContext,
    path: str,
) -> _InferenceTensorRef:
    """展开 Attention Block (推理).

    关键区别在于 context vs generation:

    Context (Prefill):
        - QK^T: (B*H_per_gpu, S, head_dim) @ (B*H_per_gpu, head_dim, S) -> (B*H_per_gpu, S, S)
        - Softmax: B * H_per_gpu * S * S
        - Score*V: (B*H_per_gpu, S, S) @ (B*H_per_gpu, S, head_dim) -> (B*H_per_gpu, S, head_dim)
        与训练相同。

    Generation (Decode):
        - 每次只处理 1 个新 token，但需要与所有历史 token 做注意力
        - QK^T: (B*H_per_gpu, 1, head_dim) @ (B*H_per_gpu, head_dim, kv_len) -> (B*H_per_gpu, 1, kv_len)
        - Softmax: B * H_per_gpu * kv_len
        - Score*V: (B*H_per_gpu, 1, kv_len) @ (B*H_per_gpu, kv_len, head_dim) -> (B*H_per_gpu, 1, head_dim)
        kv_len 是累计的上下文长度 (通常 = input_seq_len + generated_tokens)

    参考 aiconfigurator 中 ContextAttention vs GenerationAttention 的区分。
    """
    # 先递归展开子 Block (Q/K/V/O projections + norms)
    for child in block.children:
        child_path = f"{path}/{block.block_type}({block.name})"
        x = _infer_expand_block(child, x, ctx, child_path)

    # 添加 Attention 核心计算
    ctx._set_current_block(block, path)

    batch_size = ctx.metadata.get("batch_size", 1)
    seq_len = ctx.metadata.get("seq_len", 2048)
    hidden = ctx.metadata.get("hidden", 4096)
    num_heads = ctx.metadata.get("num_heads", hidden // 128)
    tp = max(1, ctx.metadata.get("tp", 1))

    heads_per_gpu = num_heads // tp
    head_dim = hidden // num_heads

    if ctx.phase == "generation":
        # Generation: 1 个新 token vs kv_len 个历史 token
        # kv_len = seq_len (近似: 输入长度 + 已生成长度)
        kv_len = ctx.metadata.get("kv_len", seq_len)

        # QK^T: (B*H, 1, head_dim) @ (B*H, head_dim, kv_len) -> (B*H, 1, kv_len)
        qk_M = batch_size * heads_per_gpu
        qk_K = head_dim
        qk_N = kv_len
        ctx.Matmul(x, None, M=qk_M, K=qk_K, N=qk_N)

        # Softmax
        ctx.Softmax(x, num_elements=batch_size * heads_per_gpu * kv_len)

        # Score*V: (B*H, 1, kv_len) @ (B*H, kv_len, head_dim) -> (B*H, 1, head_dim)
        sv_M = batch_size * heads_per_gpu
        sv_K = kv_len
        sv_N = head_dim
        ctx.Matmul(x, None, M=sv_M, K=sv_K, N=sv_N)
    else:
        # Context (Prefill): 全量注意力，与训练相同
        # QK^T: (B*H, S, head_dim) @ (B*H, head_dim, S) -> (B*H, S, S)
        qk_M = batch_size * heads_per_gpu * seq_len
        qk_K = head_dim
        qk_N = seq_len
        ctx.Matmul(x, None, M=qk_M, K=qk_K, N=qk_N)

        # Softmax
        ctx.Softmax(x, num_elements=batch_size * heads_per_gpu * seq_len * seq_len)

        # Score*V: (B*H, S, S) @ (B*H, S, head_dim) -> (B*H, S, head_dim)
        sv_M = batch_size * heads_per_gpu * seq_len
        sv_K = seq_len
        sv_N = head_dim
        ctx.Matmul(x, None, M=sv_M, K=sv_K, N=sv_N)

    return x


# Block 类型到推理展开函数的映射
_INFER_EXPAND_FUNCS = {
    "Linear": _infer_expand_linear,
    "RMSNorm": _infer_expand_rmsnorm,
    "LayerNorm": _infer_expand_layernorm,
    "GELU": _infer_expand_gelu,
    "SiLU": _infer_expand_silu,
    "Embedding": _infer_expand_embedding,
    "Attention": _infer_expand_attention,
    "FFN": _infer_expand_container,
    "MLP": _infer_expand_container,
    "TransformerLayer": _infer_expand_container,
    "Layer": _infer_expand_container,
    "Transformer": _infer_expand_container,
    "GPT": _infer_expand_container,
    "LLaMA": _infer_expand_container,
}


def _infer_expand_block(
    block: BlockNode,
    x: _InferenceTensorRef,
    ctx: _InferenceExpandContext,
    path: str = "",
) -> _InferenceTensorRef:
    """展开单个 Block 成 Op 序列 (推理)."""
    expand_func = _INFER_EXPAND_FUNCS.get(block.block_type)

    if expand_func:
        return expand_func(block, x, ctx, path)
    else:
        for child in block.children:
            child_path = f"{path}/{block.block_type}({block.name})"
            x = _infer_expand_block(child, x, ctx, child_path)
        return x


class InferenceExpandPass(Pass):
    """推理专用的 Block → Op 展开 Pass.

    与训练 ExpandPass 的区别:
    - 根据 phase (context/generation) 不同地展开 Attention
    - Context: 全量注意力，M = batch_size * seq_len
    - Generation: KV-cache 注意力，Linear 的 M = batch_size (单 token)
    - 只生成 Phase.FORWARD 的 Op

    输入: GraphIR (Block 级别)
    输出: ScheduleIR (Op 级别，无 workload 和 timing)
    """

    @hp.param("parallel")
    def __init__(
        self,
        phase: str = "context",
        device: int = 0,
        stage: int = 0,
    ):
        """初始化 InferenceExpandPass.

        Args:
            phase: 推理阶段 ("context" = prefill, "generation" = decode)
            device: 设备 ID
            stage: PP stage
        """
        self.phase = phase
        self.device = device
        self.stage = stage

    def run(self, ir: GraphIR) -> ScheduleIR:
        # 从 metadata 获取 phase（InferenceParallelPass 已设置）
        phase = ir.metadata.get("phase", self.phase)

        schedule = ScheduleIR(
            num_devices=1,
            metadata=ir.metadata.copy(),
        )

        pp = ir.metadata.get("pp", 1)
        num_layers = ir.metadata.get("num_layers", 1)
        layers_per_stage = num_layers // pp if pp > 1 else num_layers

        if ir.root:
            ctx = _InferenceExpandContext(
                block=ir.root,
                phase=phase,
                device=self.device,
                stage=self.stage,
                metadata=ir.metadata,
            )

            x = _InferenceTensorRef("input")
            _infer_expand_block(ir.root, x, ctx)

            for op in ctx.ops:
                stage = self._compute_stage(op, pp, layers_per_stage)
                op.stage = stage
                schedule.add_op(op, stage=stage, device=self.device)

        return schedule

    def _compute_stage(self, op: ScheduledOp, pp: int, layers_per_stage: int) -> int:
        """根据 Op 的 source_block 计算其所属的 stage."""
        if pp <= 1:
            return 0

        source = op.op.source_block if op.op else ""
        if not source:
            return 0

        match = re.search(r"layer(\d+)", source)
        if match:
            layer_idx = int(match.group(1))
            stage = layer_idx // layers_per_stage if layers_per_stage > 0 else 0
            return min(stage, pp - 1)

        return 0


# ============================================================================
# InferenceSchedulePass
# ============================================================================


def _symbolic_max(*args):
    """符号安全的 max 函数."""
    has_symbolic = any(isinstance(a, Expr) for a in args)
    if has_symbolic:
        return SymMax(*args)
    return max(args)


class InferenceSchedulePass(Pass):
    """推理专用的调度 Pass.

    与训练 SchedulePass 的区别:
    - 添加 KV-cache 内存估算 (写入 metadata)
    - Generation 阶段的 Matmul 通常是 memory-bandwidth bound (小 batch)
    - 不需要考虑梯度、优化器状态的内存

    支持两种 duration 计算模式:
    1. roofline 模型 (默认): 基于 peak_flops 和 memory_bandwidth 的分析估算
    2. 查表模式 (perf_db): 使用 PerfDatabase 实测数据查表，fallback 到 roofline

    当提供 perf_db 时，GEMM 和通信 Op 优先使用实测数据；
    Norm、Softmax、Activation 等操作仍使用 roofline 模型。

    输入: ScheduleIR (无 workload 和 timing)
    输出: ScheduleIR (有 workload 和 timing)
    """

    @hp.param("parallel")
    def __init__(
        self,
        # 硬件参数
        peak_tflops: float = 312.0,
        memory_bandwidth: float = 2.0e12,
        network_bandwidth: float = 400e9,
        network_efficiency: float = 0.65,
        network_latency: float = 10e-6,
        compute_efficiency: float = 0.95,
        # 通信参数
        all_reduce_offset: float = 1.0,
        # 处理模式
        processing_mode: str = "roofline",
        # 量化配置
        quant_config: Optional[QuantConfig] = None,
        # 性能数据库 (可选)
        perf_db: Optional[Any] = None,
    ):
        """初始化 InferenceSchedulePass.

        Args:
            peak_tflops: 计算峰值 (TFLOPS), fp16 baseline
            memory_bandwidth: 内存带宽 (bytes/s)
            network_bandwidth: 网络带宽 (bytes/s)
            network_efficiency: 网络效率
            network_latency: 网络延迟 (seconds)
            compute_efficiency: 计算效率
            all_reduce_offset: AllReduce 通信偏移量
            processing_mode: "roofline" 或 "no_overlap"
            quant_config: 量化配置 (默认 fp16)
            perf_db: PerfDatabase 实例 (可选)。提供时优先使用实测数据查表，
                     GEMM 和通信 Op 使用查表结果，其他 Op 仍用 roofline。
                     查表失败时自动 fallback 到 roofline 模型。
        """
        self.quant = quant_config or QuantConfig()
        self.peak_tflops = peak_tflops
        self.peak_flops = peak_tflops * 1e12 * compute_efficiency
        # GEMM 受益于量化 tensor core 加速 (e.g., FP8 = 2x, NVFP4 = 4x)
        self.gemm_peak_flops = self.peak_flops * self.quant.compute_multiplier
        self.memory_bandwidth = memory_bandwidth
        self.network_bandwidth = network_bandwidth
        self.network_efficiency = network_efficiency
        self.network_latency = network_latency
        self.compute_efficiency = compute_efficiency
        self.all_reduce_offset = all_reduce_offset
        self.processing_mode = processing_mode
        self.perf_db = perf_db

    def run(self, ir: ScheduleIR) -> ScheduleIR:
        """执行推理调度计算.

        职责:
        1. 计算每个 Op 的 workload + duration（roofline 或查表）
        2. 生成 memory_pools 内存注解（权重、激活、KV cache），
           下游 TimelinePass 机械翻译成 ALLOC/FREE 事件，无需了解推理语义。
        """
        metadata = ir.metadata

        batch = metadata.get("batch_size", metadata.get("batch", 1))
        seq = metadata.get("seq_len", metadata.get("seq", 2048))
        hidden = metadata.get("hidden", 4096)
        feedforward = metadata.get("feedforward", hidden * 4)
        phase = metadata.get("phase", "context")

        if phase == "generation":
            batch_seq = batch  # generation: 单 token
        else:
            batch_seq = batch * seq

        current_time: Union[float, Expr] = 0

        for op in ir.iter_ops():
            self._compute_workload(op.op, batch_seq, hidden, feedforward, metadata)
            duration = self._compute_duration(op.op)
            op.duration = duration
            op.start = current_time
            current_time = current_time + duration

        # ---- 生成 memory_pools 内存注解 ----
        forward_end = current_time  # 所有 Op 的结束时间

        self._emit_weight_pools(ir)
        self._emit_activation_pool(ir, forward_end)
        self._emit_kv_cache_pool(ir, forward_end)

        return ir

    # ---- memory_pools 生成 ----

    def _emit_weight_pools(self, ir: ScheduleIR) -> None:
        """生成权重内存池.

        遍历前向 Matmul Op，按 source_block 去重，用 quant.weight_bytes 计算权重大小。
        权重在模型加载时分配（alloc_time=0），推理期间不释放。
        """
        seen = set()
        pp = ir.metadata.get("pp", 1)
        total_weight = 0.0

        for op in ir.iter_ops():
            if op.op_type == "Matmul" and op.op:
                source = op.op.source_block or ""
                if source in seen:
                    continue
                seen.add(source)
                K = op.op.attrs.get("K", 0)
                N = op.op.attrs.get("N", 0)
                weight_bytes = K * N * self.quant.weight_bytes
                total_weight += weight_bytes

        # per-GPU：PP 分片后每个 GPU 只存部分层权重
        weight_per_gpu = total_weight / pp if pp > 0 else total_weight
        if weight_per_gpu > 0:
            ir.memory_pools.append(
                MemoryPool(
                    name="weight",
                    mem_type="weight",
                    size_bytes=weight_per_gpu,
                    alloc_time=0,
                    free_time=None,  # 推理期间不释放
                )
            )

    def _emit_activation_pool(self, ir: ScheduleIR, forward_end) -> None:
        """生成推理激活内存池.

        推理不需要为反向保留所有层激活，工作集约 1~2 层。
        前向结束时释放。
        """
        metadata = ir.metadata
        pp = metadata.get("pp", 1)
        num_layers = metadata.get("num_layers", 1)
        layers_per_stage = num_layers // pp if pp > 0 else num_layers

        # 从 Op 推导单层激活大小
        per_layer_activation = self._estimate_per_layer_activation(ir)
        # 推理工作集：1~2 层
        peak_activation = per_layer_activation * min(2, layers_per_stage)

        if peak_activation > 0:
            ir.memory_pools.append(
                MemoryPool(
                    name="activation",
                    mem_type="activation",
                    size_bytes=peak_activation,
                    alloc_time=0,
                    free_time=forward_end,
                    metadata={
                        "per_layer": per_layer_activation,
                        "layers_per_stage": layers_per_stage,
                        "model": "inference_working_set",
                    },
                )
            )

    def _emit_kv_cache_pool(self, ir: ScheduleIR, forward_end) -> None:
        """生成 KV-cache 内存池.

        KV-cache 存储每层的 K 和 V 张量:
        - 每层 KV-cache = 2 * batch_size * kv_len * num_kv_heads_per_gpu * head_dim * kv_cache_bytes
        - 总 KV-cache = per_layer * layers_per_stage (PP 分片)

        量化影响: KV-cache 使用 quant_config.kv_cache_bytes 精度存储。
        """
        metadata = ir.metadata
        batch_size = metadata.get("batch_size", 1)
        seq_len = metadata.get("seq_len", 2048)
        hidden = metadata.get("hidden", 4096)
        num_heads = metadata.get("num_heads", hidden // 128)
        num_layers = metadata.get("num_layers", 1)
        pp = metadata.get("pp", 1)
        tp = max(1, metadata.get("tp", 1))

        head_dim = hidden // num_heads
        num_kv_heads = metadata.get("num_kv_heads", num_heads)
        num_kv_heads_per_gpu = max(1, num_kv_heads // tp)
        layers_per_stage = num_layers // pp if pp > 0 else num_layers
        kv_len = metadata.get("kv_len", seq_len)

        per_layer_kv = (
            2 * batch_size * kv_len * num_kv_heads_per_gpu * head_dim * self.quant.kv_cache_bytes
        )
        total_kv_cache = per_layer_kv * layers_per_stage

        if total_kv_cache > 0:
            ir.memory_pools.append(
                MemoryPool(
                    name="kv_cache",
                    mem_type="kv_cache",
                    size_bytes=total_kv_cache,
                    alloc_time=0,
                    free_time=forward_end,
                    metadata={"per_layer": per_layer_kv},
                )
            )

        # 兼容：写入 metadata 供上层读
        metadata["kv_cache_per_layer_bytes"] = per_layer_kv
        metadata["kv_cache_bytes"] = total_kv_cache

    def _estimate_per_layer_activation(self, ir: ScheduleIR) -> float:
        """从 Op 推导单层激活大小（首层前向 Op 输出的累积）."""
        import re

        layer_ops: dict[int, list[ScheduledOp]] = {}
        for op in ir.iter_ops():
            source = op.op.source_block if op.op else ""
            layer_idx = 0
            if source:
                m = re.search(r"layer(\d+)", source, re.IGNORECASE)
                if m:
                    layer_idx = int(m.group(1))
            if layer_idx not in layer_ops:
                layer_ops[layer_idx] = []
            layer_ops[layer_idx].append(op)

        if not layer_ops:
            return 0.0

        first_layer = layer_ops[min(layer_ops.keys())]
        per_layer = 0.0
        for op in first_layer:
            if not op.op or not op.op.attrs:
                continue
            attrs = op.op.attrs
            if op.op_type == "Matmul":
                M = attrs.get("M", 0)
                N = attrs.get("N", 0)
                per_layer += M * N * self.quant.activation_bytes
            elif op.op_type in ("RMSNorm", "LayerNorm"):
                ns = attrs.get("normalized_shape", 0)
                metadata = ir.metadata
                batch_seq = metadata.get("batch_size", 1) * metadata.get("seq_len", 2048)
                phase = metadata.get("phase", "context")
                if phase == "generation":
                    batch_seq = metadata.get("batch_size", 1)
                per_layer += batch_seq * ns * self.quant.activation_bytes
            elif op.op_type in ("Softmax", "SiLU", "GELU", "ReLU"):
                ne = attrs.get("num_elements", 0)
                per_layer += ne * self.quant.activation_bytes
        return per_layer

    def _compute_workload(
        self,
        op: OpNode,
        batch_seq,
        hidden,
        feedforward,
        metadata: Dict,
    ) -> None:
        """计算单个 Op 的 workload (与训练 SchedulePass 相同)."""
        op_type = op.op_type
        attrs = op.attrs

        if op_type == "Matmul":
            self._compute_matmul(op, attrs)
        elif op_type in ("RMSNorm", "LayerNorm"):
            self._compute_norm(op, attrs, batch_seq)
        elif op_type == "Softmax":
            self._compute_softmax(op, attrs)
        elif op_type in ("SiLU", "GELU", "ReLU"):
            self._compute_activation(op, attrs, batch_seq, feedforward)
        elif op_type in ("Add", "Mul"):
            self._compute_elementwise(op, attrs, batch_seq, hidden)
        elif op_type in ("AllReduce", "AllGather", "ReduceScatter"):
            self._compute_collective(op, attrs)
        elif op_type in ("Send", "Recv"):
            self._compute_p2p(op, attrs)
        else:
            op.flops = 0
            op.memory_bytes = 0
            op.comm_bytes = 0

    def _compute_matmul(self, op: OpNode, attrs: Dict) -> None:
        """计算 Matmul workload.

        量化影响:
        - 读取输入激活: M * K * activation_bytes
        - 读取权重: K * N * weight_bytes (权重量化直接减少此项)
        - 写入输出激活: M * N * activation_bytes
        - 对于 weight-only 量化 (w4a16, w8a16): 权重内存大幅减少，
          这对 generation 阶段的 memory-bandwidth bound GEMM 效果特别显著
        """
        M = attrs.get("M", 1)
        K = attrs.get("K", 1)
        N = attrs.get("N", 1)

        op.flops = 2 * M * K * N
        # 分别计算: 输入激活(read) + 权重(read) + 输出激活(write)
        op.memory_bytes = (
            M * K * self.quant.activation_bytes  # 读取输入激活
            + K * N * self.quant.weight_bytes     # 读取权重
            + M * N * self.quant.activation_bytes  # 写入输出激活
        )
        op.comm_bytes = 0

    def _compute_norm(self, op: OpNode, attrs: Dict, batch_seq) -> None:
        """计算 RMSNorm/LayerNorm workload.

        Norm 操作在激活精度下执行，使用 activation_bytes。
        """
        normalized_shape = attrs.get("normalized_shape", 4096)
        num_elements = batch_seq * normalized_shape

        op.flops = 5 * num_elements
        op.memory_bytes = 2 * num_elements * self.quant.activation_bytes
        op.comm_bytes = 0

    def _compute_softmax(self, op: OpNode, attrs: Dict) -> None:
        """计算 Softmax workload.

        Softmax 在激活精度下执行。
        """
        num_elements = attrs.get("num_elements", 1)

        op.flops = 5 * num_elements
        op.memory_bytes = 2 * num_elements * self.quant.activation_bytes
        op.comm_bytes = 0

    def _compute_activation(
        self, op: OpNode, attrs: Dict, batch_seq, feedforward
    ) -> None:
        """计算激活函数 workload.

        激活函数在激活精度下执行。
        """
        num_elements = attrs.get("num_elements", batch_seq * feedforward)

        if op.op_type == "SiLU":
            op.flops = 4 * num_elements
        elif op.op_type == "GELU":
            op.flops = 8 * num_elements
        else:
            op.flops = num_elements

        op.memory_bytes = 2 * num_elements * self.quant.activation_bytes
        op.comm_bytes = 0

    def _compute_elementwise(self, op: OpNode, attrs: Dict, batch_seq, hidden) -> None:
        """计算 Add/Mul workload.

        逐元素操作在激活精度下执行。
        """
        num_elements = attrs.get("num_elements", batch_seq * hidden)

        op.flops = num_elements
        op.memory_bytes = 3 * num_elements * self.quant.activation_bytes
        op.comm_bytes = 0

    def _compute_collective(self, op: OpNode, attrs: Dict) -> None:
        """计算集合通信 workload.

        通信使用 comm_bytes 精度。量化通信 (如 fp8 AllReduce)
        可将通信量减半。参考 aiconfigurator CommQuantMode。
        """
        data_size = attrs.get("data_size", 1)
        num_peers = max(1, attrs.get("num_peers", 8))

        op.flops = 0
        op.memory_bytes = 0

        base_comm_bytes = data_size * self.quant.comm_bytes

        if op.op_type == "AllReduce":
            op.comm_bytes = base_comm_bytes * (1 + self.all_reduce_offset / num_peers)
        elif op.op_type == "AllGather":
            op.comm_bytes = base_comm_bytes * (num_peers - 1) / num_peers
        elif op.op_type == "ReduceScatter":
            op.comm_bytes = base_comm_bytes * (num_peers - 1) / num_peers
        else:
            op.comm_bytes = base_comm_bytes

    def _compute_p2p(self, op: OpNode, attrs: Dict) -> None:
        """计算点对点通信 workload."""
        data_size = attrs.get("data_size", 1)

        op.flops = 0
        op.memory_bytes = 0
        op.comm_bytes = data_size * self.quant.comm_bytes

    def _compute_duration(self, op: OpNode) -> Union[float, Expr]:
        """计算 Op 的执行时间.

        优先使用 PerfDatabase 查表 (如果可用)，否则使用 roofline 模型。

        查表策略:
        - Matmul: query_gemm(m, n, k, quant_mode) → latency (ms)
        - AllReduce: query_custom_allreduce(tp_size, size) → latency (ms)
        - AllGather/ReduceScatter: query_nccl(num_gpus, operation, size) → latency (ms)
        - 其他 Op (Norm, Softmax, Activation): 始终使用 roofline 模型

        Roofline 模型:
        - Matmul: 使用 gemm_peak_flops (= peak_flops * compute_multiplier)
        - 其他 Op: 使用 peak_flops (不受量化加速)
        - 内存时间: 通过不同的 bytes-per-element 体现
        """
        # 尝试使用 PerfDatabase 查表
        if self.perf_db is not None:
            db_result = self._query_perf_db(op)
            if db_result is not None:
                return db_result

        # Fallback: roofline 模型
        return self._compute_duration_roofline(op)

    def _query_perf_db(self, op: OpNode) -> Optional[Union[float, Expr]]:
        """尝试使用 PerfDatabase 查表获取 Op 延迟.

        Returns:
            latency (seconds) or None if 查表不适用/失败
        """
        from ..perf_database import GEMMQuantMode, CommQuantMode

        try:
            if op.op_type == "Matmul":
                attrs = op.attrs
                M = attrs.get("M", 1)
                K = attrs.get("K", 1)
                N = attrs.get("N", 1)
                # 跳过含符号表达式的 (无法查表)
                if any(isinstance(v, Expr) for v in (M, K, N)):
                    return None
                # 将 QuantConfig 映射到 GEMMQuantMode
                quant_mode = self._map_gemm_quant_mode()
                latency_ms = self.perf_db.query_gemm(
                    int(M), int(N), int(K), quant_mode
                )
                return latency_ms / 1000.0  # ms → seconds

            elif op.op_type == "AllReduce":
                attrs = op.attrs
                data_size = attrs.get("data_size", 1)
                num_peers = max(1, attrs.get("num_peers", 8))
                if isinstance(data_size, Expr):
                    return None
                latency_ms = self.perf_db.query_custom_allreduce(
                    quant_mode=CommQuantMode.half,
                    tp_size=int(num_peers),
                    size=int(data_size),
                )
                return latency_ms / 1000.0

            elif op.op_type in ("AllGather", "ReduceScatter"):
                attrs = op.attrs
                data_size = attrs.get("data_size", 1)
                num_peers = max(1, attrs.get("num_peers", 8))
                if isinstance(data_size, Expr):
                    return None
                operation = "all_gather" if op.op_type == "AllGather" else "reduce_scatter"
                latency_ms = self.perf_db.query_nccl(
                    dtype=CommQuantMode.half,
                    num_gpus=int(num_peers),
                    operation=operation,
                    message_size=int(data_size),
                )
                return latency_ms / 1000.0

        except Exception as e:
            import logging
            logging.getLogger(__name__).debug(f"PerfDB 查表失败 ({op.op_type}): {e}")

        return None

    def _map_gemm_quant_mode(self):
        """将 QuantConfig 映射到 PerfDatabase 的 GEMMQuantMode."""
        from ..perf_database import GEMMQuantMode

        w = self.quant.weight_bytes
        c = self.quant.compute_multiplier
        # 根据 weight_bytes 和 compute_multiplier 推断模式
        if w == 0.5 and c >= 4:
            return GEMMQuantMode.nvfp4
        elif w == 0.5:
            return GEMMQuantMode.int4_wo
        elif w == 1.0 and c >= 2:
            return GEMMQuantMode.fp8
        elif w == 1.0:
            return GEMMQuantMode.int8_wo
        else:
            return GEMMQuantMode.float16

    def _compute_duration_roofline(self, op: OpNode) -> Union[float, Expr]:
        """基于 roofline 模型计算 duration (原始逻辑)."""
        flops = op.flops or 0
        memory = op.memory_bytes or 0
        comm = op.comm_bytes or 0

        # Matmul 使用量化加速的 peak flops，其他 Op 使用 fp16 peak flops
        if op.op_type == "Matmul":
            effective_peak_flops = self.gemm_peak_flops
        else:
            effective_peak_flops = self.peak_flops

        compute_time = flops / effective_peak_flops if flops > 0 else 0
        memory_time = memory / self.memory_bandwidth if memory > 0 else 0

        if comm > 0:
            effective_bandwidth = self.network_bandwidth * self.network_efficiency
            comm_time = self.network_latency + comm / effective_bandwidth
        else:
            comm_time = 0

        if op.op_type in ("AllReduce", "AllGather", "ReduceScatter", "Send", "Recv"):
            duration = comm_time
        elif self.processing_mode == "no_overlap":
            duration = compute_time + memory_time
        else:
            duration = _symbolic_max(compute_time, memory_time)

        return duration

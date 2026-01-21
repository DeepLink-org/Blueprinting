#!/usr/bin/env python3
"""GPT-175B 模型训练仿真示例

本示例展示如何使用 blueprinting IR 编译器对 GPT-175B 模型进行
训练仿真分析，并与 Calculon 的结果进行对比。

编译流程：
    GraphIR → WorkloadPass → ParallelPass → SchedulePass → TimelinePass → EvaluatePass → SimulationResult

主要功能：
1. 从模型参数构建计算图 (GraphIR)
2. 计算各操作的计算量、内存访问量 (WorkloadPass)
3. 应用并行策略 (ParallelPass): TP=8, PP=8, DP=16
4. 生成执行调度 (SchedulePass)
5. 生成事件时间线 (TimelinePass)
6. 评估最终结果 (EvaluatePass)
7. 与 Calculon 结果对比
8. 输出 Chrome Trace 格式用于可视化

使用方法:
    python examples/gpt175b_compile.py
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, Union, Optional

from sympy import Symbol

# 添加 src 到 path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from blueprinting.ir import (
    # Core IR types
    GraphIR,
    OpNode,
    TensorRef,
    ScheduleIR,
    TimelineIR,
    SimulationResult,
    # Builder
    IRBuilder,
    # Compiler and Passes
    Compiler,
    WorkloadPass,
    ParallelPass,
    SchedulePass,
    TimelinePass,
    OverlapAnalysisPass,
    EvaluatePass,
    OptimizerPass,
    OptimizerConfig,
    # Utilities
    EventType,
    StreamType,
    get_cache_stats,
    clear_expr_cache,
)

# Calculon imports for comparison
import calculon
from calculon import System
from calculon.llm import Llm


# ============================================================================
# GPT-175B 模型参数
# ============================================================================

GPT_175B_CONFIG = {
    "name": "GPT-175B",
    "hidden": 12288,           # 隐藏层维度
    "feedforward": 49152,      # FFN中间层维度 (4 * hidden)
    "seq_len": 2048,           # 序列长度
    "num_heads": 96,           # 注意力头数
    "head_dim": 128,           # 每头维度 (hidden / num_heads)
    "num_layers": 96,          # Transformer层数
    "vocab_size": 50257,       # 词表大小
}

# H100 NVL8 系统配置
H100_NVL8_CONFIG = {
    "name": "H100-80G-NVL8",
    "peak_tflops": 1000,              # FP16 峰值 TFLOPS
    "memory_bandwidth_gbps": 3072,    # HBM3 带宽 GB/s
    "network_bandwidth_gbps": 450,    # NVLink 带宽 GB/s
    "memory_capacity_gb": 80,         # 显存容量 GB
    "num_gpus_per_node": 8,           # 每节点GPU数
}

# 训练配置
TRAINING_CONFIG = {
    "batch_size": 2048,               # Global batch size
    "micro_batch_size": 4,            # Micro batch size per GPU
    "tensor_parallel": 8,             # TP 度
    "pipeline_parallel": 8,           # PP 度 (需要多节点)
    "data_parallel": 16,              # DP 度
    "dtype": "float16",               # 数据类型
    "optimizer": "adam",              # 优化器
    "gradient_checkpointing": True,   # 梯度检查点
}


def build_gpt175b_graph(
    config: Dict[str, Any],
    batch_size: int,
    seq_len: int,
    tp: int = 1,
) -> GraphIR:
    """构建 GPT-175B 模型的 GraphIR。
    
    使用 IRBuilder 构建包含以下结构的计算图：
    - Input Embedding
    - N × Transformer Blocks:
        - RMSNorm → Attention (QKV + MHA + Out) → Residual Add
        - RMSNorm → FFN (Gate + Up + SiLU + Mul + Down) → Residual Add
    - Final RMSNorm → LM Head
    
    Args:
        config: 模型配置字典
        batch_size: 批大小
        seq_len: 序列长度
        tp: Tensor Parallelism degree
        
    Returns:
        GraphIR 表示的计算图
    """
    builder = IRBuilder()
    
    # 定义符号变量
    B = builder.add_symbol("B", "batch_size")
    S = builder.add_symbol("S", "seq_len")
    H = builder.add_symbol("H", "hidden")
    FF = builder.add_symbol("FF", "feedforward")
    
    hidden = config["hidden"]
    feedforward = config["feedforward"]
    num_heads = config["num_heads"]
    head_dim = config["head_dim"]
    num_layers = config["num_layers"]
    
    # 计算每GPU上的维度（考虑TP）
    hidden_per_tp = hidden // tp if tp > 1 else hidden
    ff_per_tp = feedforward // tp if tp > 1 else feedforward
    heads_per_tp = num_heads // tp if tp > 1 else num_heads
    
    # 设置元数据
    builder.set_metadata("model_name", "GPT-175B")
    builder.set_metadata("num_layers", num_layers)
    builder.set_metadata("hidden", hidden)
    builder.set_metadata("feedforward", feedforward)
    builder.set_metadata("batch_size", batch_size)
    builder.set_metadata("seq_len", seq_len)
    builder.set_metadata("tp", tp)
    
    # Input tensor
    batch_seq = batch_size * seq_len
    x = builder.add_input("input_ids", shape=[batch_size, seq_len, hidden])
    
    prev_output = x
    
    # 构建 Transformer 层
    for layer_idx in range(num_layers):
        prefix = f"layer{layer_idx}_"
        
        # =====================
        # Attention Block
        # =====================
        
        # Input LayerNorm / RMSNorm
        attn_norm = builder.add_rmsnorm(
            f"{prefix}attn_norm", 
            [prev_output], 
            normalized_shape=hidden
        )
        builder._nodes[attn_norm].attrs["batch_seq"] = batch_seq
        
        # Q, K, V projections (Column Parallel if TP > 1)
        # Each projection: [B*S, H] @ [H, H] -> [B*S, H]
        # ParallelPass will handle TP sharding
        q_proj = builder.add_linear(
            f"{prefix}q_proj",
            [attn_norm],
            in_features=hidden,
            out_features=hidden,
            shard="tp_col" if tp > 1 else None
        )
        builder._nodes[q_proj].attrs["batch_seq"] = batch_seq
        
        k_proj = builder.add_linear(
            f"{prefix}k_proj",
            [attn_norm],
            in_features=hidden,
            out_features=hidden,
            shard="tp_col" if tp > 1 else None
        )
        builder._nodes[k_proj].attrs["batch_seq"] = batch_seq
        
        v_proj = builder.add_linear(
            f"{prefix}v_proj",
            [attn_norm],
            in_features=hidden,
            out_features=hidden,
            shard="tp_col" if tp > 1 else None
        )
        builder._nodes[v_proj].attrs["batch_seq"] = batch_seq
        
        # Attention computation: Q @ K^T -> Softmax -> @ V
        # FLOPs: 2 * B * H * S * S * D (QK^T) + 2 * B * H * S * D * S (Score@V)
        attn = builder.add_attention(
            f"{prefix}attention",
            [q_proj, k_proj, v_proj],
            num_heads=heads_per_tp,
            head_dim=head_dim,
            seq_len=seq_len
        )
        builder._nodes[attn].attrs["batch_size"] = batch_size
        
        # Output projection (Row Parallel if TP > 1)
        attn_out = builder.add_linear(
            f"{prefix}attn_out",
            [attn],
            in_features=hidden,
            out_features=hidden,
            shard="tp_row" if tp > 1 else None
        )
        builder._nodes[attn_out].attrs["batch_seq"] = batch_seq
        
        # AllReduce for TP (if TP > 1)
        if tp > 1:
            attn_comm = builder.add_comm(
                f"{prefix}attn_allreduce",
                "AllReduce",
                [attn_out],
                num_peers=tp
            )
            builder._nodes[attn_comm].attrs["data_size"] = batch_seq * hidden
            attn_output_for_add = attn_comm
        else:
            attn_output_for_add = attn_out
        
        # Residual Add
        attn_add = builder.add_elementwise(
            f"{prefix}attn_add",
            "Add",
            [prev_output, attn_output_for_add]
        )
        builder._nodes[attn_add].attrs["num_elements"] = batch_seq * hidden
        
        # =====================
        # FFN Block (SwiGLU)
        # =====================
        
        # FFN LayerNorm / RMSNorm
        ffn_norm = builder.add_rmsnorm(
            f"{prefix}ffn_norm",
            [attn_add],
            normalized_shape=hidden
        )
        builder._nodes[ffn_norm].attrs["batch_seq"] = batch_seq
        
        # Gate projection (Column Parallel)
        gate_proj = builder.add_linear(
            f"{prefix}gate_proj",
            [ffn_norm],
            in_features=hidden,
            out_features=feedforward,
            shard="tp_col" if tp > 1 else None
        )
        builder._nodes[gate_proj].attrs["batch_seq"] = batch_seq
        
        # Up projection (Column Parallel)
        up_proj = builder.add_linear(
            f"{prefix}up_proj",
            [ffn_norm],
            in_features=hidden,
            out_features=feedforward,
            shard="tp_col" if tp > 1 else None
        )
        builder._nodes[up_proj].attrs["batch_seq"] = batch_seq
        
        # SiLU activation on gate
        gate_silu = builder.add_elementwise(
            f"{prefix}gate_silu",
            "SiLU",
            [gate_proj]
        )
        builder._nodes[gate_silu].attrs["num_elements"] = batch_seq * ff_per_tp
        
        # Element-wise multiply (SwiGLU)
        swiglu_mul = builder.add_elementwise(
            f"{prefix}swiglu_mul",
            "Mul",
            [gate_silu, up_proj]
        )
        builder._nodes[swiglu_mul].attrs["num_elements"] = batch_seq * ff_per_tp
        
        # Down projection (Row Parallel)
        down_proj = builder.add_linear(
            f"{prefix}down_proj",
            [swiglu_mul],
            in_features=feedforward,
            out_features=hidden,
            shard="tp_row" if tp > 1 else None
        )
        builder._nodes[down_proj].attrs["batch_seq"] = batch_seq
        
        # AllReduce for TP (if TP > 1)
        if tp > 1:
            ffn_comm = builder.add_comm(
                f"{prefix}ffn_allreduce",
                "AllReduce",
                [down_proj],
                num_peers=tp
            )
            builder._nodes[ffn_comm].attrs["data_size"] = batch_seq * hidden
            ffn_output_for_add = ffn_comm
        else:
            ffn_output_for_add = down_proj
        
        # Residual Add
        ffn_add = builder.add_elementwise(
            f"{prefix}ffn_add",
            "Add",
            [attn_add, ffn_output_for_add]
        )
        builder._nodes[ffn_add].attrs["num_elements"] = batch_seq * hidden
        
        prev_output = ffn_add
    
    # Final RMSNorm
    final_norm = builder.add_rmsnorm(
        "final_norm",
        [prev_output],
        normalized_shape=hidden
    )
    builder._nodes[final_norm].attrs["batch_seq"] = batch_seq
    
    # LM Head (output projection to vocab)
    # Note: Often tied with input embedding, but we model it separately
    lm_head = builder.add_linear(
        "lm_head",
        [final_norm],
        in_features=hidden,
        out_features=config["vocab_size"],
        shard="tp_col" if tp > 1 else None
    )
    builder._nodes[lm_head].attrs["batch_seq"] = batch_seq
    
    builder.mark_output(lm_head)
    
    return builder.build()


def create_compiler_pipeline(
    system_config: Union[Dict[str, Any], str, Path],
    tp: int = 1,
    pp: int = 1,
    dp: int = 1,
    batch_size: int = 1,
    seq_len: int = 2048,
    hidden: int = 4096,
    feedforward: int = 16384,
    num_layers: int = 32,
    training: bool = True,
    use_calculon_config: bool = True,
    calibration_override: Optional[Dict[str, float]] = None,
    gradient_checkpointing: bool = False,
    num_microbatches: int = 1,
) -> Compiler:
    """创建完整的编译器流水线。
    
    Pipeline: GraphIR → WorkloadPass → ParallelPass → SchedulePass 
              → OptimizerPass → TimelinePass → OverlapAnalysisPass → EvaluatePass
    
    Args:
        system_config: 硬件系统配置，支持:
            - Calculon JSON 配置文件路径
            - Calculon 格式配置字典
            - 简单配置字典 (peak_tflops, memory_bandwidth_gbps, etc.)
        tp, pp, dp: 并行度
        batch_size, seq_len, hidden, feedforward: 模型/训练参数
        num_layers: 层数
        training: 是否训练模式
        use_calculon_config: 是否使用 Calculon 的动态效率因子 (从配置文件)
        calibration_override: 覆盖配置文件的效率因子 (使用固定值)
        gradient_checkpointing: 是否启用梯度检查点
        num_microbatches: microbatch 数量
        
    Returns:
        配置好的 Compiler
    """
    from blueprinting.ir import SystemConfig
    
    # 加载系统配置
    if isinstance(system_config, (str, Path)):
        sys_cfg = SystemConfig(system_config)
    elif isinstance(system_config, dict):
        sys_cfg = SystemConfig(system_config)
    else:
        sys_cfg = system_config
    
    # 符号替换表
    subs = {
        "B": batch_size,
        "S": seq_len,
        "H": hidden,
        "FF": feedforward,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "hidden": hidden,
        "feedforward": feedforward,
        "batch_seq": batch_size * seq_len,
        "num_layers": num_layers,
    }
    
    # 校准参数设置
    # - use_calculon_config=True + calibration_override=None: 使用配置文件的动态效率
    # - use_calculon_config=False 或 calibration_override 有值: 使用固定效率
    calibration = None
    if not use_calculon_config or calibration_override:
        calibration = calibration_override or {
            "compute_efficiency": 0.50,  # 默认固定效率
            "memory_efficiency": 0.75,
            "network_efficiency": 0.65,
        }
    
    # 优化器配置
    optimizer_config = OptimizerConfig(
        optimizer_type="adam",
        dtype_bytes=2,  # FP16
        master_weights=True,  # 使用 FP32 主权重
        gradient_checkpointing=gradient_checkpointing,
        recompute_mode="attn_only" if gradient_checkpointing else "none",  # 与 Calculon 一致
        num_microbatches=num_microbatches,
    )
    
    # 创建编译器
    compiler = Compiler(sys_cfg.to_dict())
    
    # 添加 passes
    compiler.add_pass(WorkloadPass(dtype_bytes=2, count_add=True))
    compiler.add_pass(ParallelPass(tp=tp, pp=pp, dp=dp, sequence_parallel=False))
    compiler.add_pass(SchedulePass(
        system_config=sys_cfg,  # 直接传 SystemConfig 对象
        strategy="sequential",
        overlap_compute_comm=True,
        processing_mode=None,  # 使用配置文件的 processing_mode
        calibration=calibration,
        training=training,
    ))
    # 添加优化器 Pass (在 Schedule 之后，Timeline 之前)
    if training:
        compiler.add_pass(OptimizerPass(
            optimizer_config=optimizer_config,
            system_config=sys_cfg.to_dict(),
        ))
    compiler.add_pass(TimelinePass(
        model_overlap=True,
        memory_model="eager",
        include_optimizer=True
    ))
    compiler.add_pass(OverlapAnalysisPass(overlap_efficiency=0.9))
    compiler.add_pass(EvaluatePass(subs=subs, training=training, optimizer="adam"))
    
    return compiler


def calculate_model_metrics(config: Dict[str, Any], training_config: Dict[str, Any]) -> Dict[str, Any]:
    """计算模型的理论 FLOPs 和内存需求。
    
    基于论文: "Efficient Large-Scale Language Model Training on GPU Clusters"
    https://arxiv.org/abs/2104.04473
    
    Args:
        config: 模型配置
        training_config: 训练配置
        
    Returns:
        包含参数量、FLOPs、内存等指标的字典
    """
    hidden = config["hidden"]
    feedforward = config["feedforward"]
    num_layers = config["num_layers"]
    num_heads = config["num_heads"]
    head_dim = config["head_dim"]
    vocab_size = config["vocab_size"]
    seq_len = config["seq_len"]
    
    batch_size = training_config["batch_size"]
    micro_batch = training_config["micro_batch_size"]
    tp = training_config["tensor_parallel"]
    pp = training_config["pipeline_parallel"]
    dp = training_config["data_parallel"]
    
    # ========== 参数量计算 (与 Calculon 对齐) ==========
    # 参考 Megatron-LM 论文: https://cs.stanford.edu/~matei/papers/2021/sc_megatron_lm.pdf
    # Calculon 使用标准 GPT 架构，包含 biases
    
    # 每层参数 (与 Calculon 的 num_parameters() 对齐):
    # - MLP weights: 2 * H * FF (标准 2 层，非 SwiGLU)
    # - Attention weights: 4 * H * H (Q, K, V, O)
    # - MLP biases: H + FF
    # - Attention biases: 3 * H + H (Q, K, V biases + O bias)
    # - LayerNorm: 2 * 2 * H (2 layers, each with weight + bias)
    params_per_layer = (
        2 * hidden * feedforward +      # MLP weights (标准 GPT 2层)
        4 * hidden * hidden +           # Attention: Q, K, V, O
        hidden + feedforward +          # MLP biases
        4 * hidden +                    # Attention biases (Q,K,V,O)
        4 * hidden                      # LayerNorm (2 layers × 2 params)
    )
    
    # Embedding 参数 (Calculon 使用 vocab=51200 + positional)
    # Calculon 公式: (51200 + seq_size) * hidden
    embedding_params = (vocab_size + seq_len) * hidden
    
    total_params = num_layers * params_per_layer + embedding_params
    
    # ========== FLOPs 计算 (每个 token) ==========
    # 参考: https://arxiv.org/abs/2104.04473
    # 
    # 每层 Forward FLOPs:
    # - QKV projection: 3 * 2 * B * S * H * H
    # - Attention scores: 2 * B * H * S * S * D (D = head_dim)
    # - Softmax: 3 * B * H * S * S (approximate)
    # - Score @ V: 2 * B * H * S * S * D
    # - Out projection: 2 * B * S * H * H
    # - FFN: 2 * 3 * B * S * H * FF (gate + up + down)
    
    B = micro_batch
    S = seq_len
    H = hidden
    FF = feedforward
    D = head_dim
    n_heads = num_heads
    
    # Attention FLOPs per layer
    qkv_flops = 3 * 2 * B * S * H * H
    attn_score_flops = 2 * B * n_heads * S * S * D
    score_v_flops = 2 * B * n_heads * S * S * D
    out_proj_flops = 2 * B * S * H * H
    attention_flops = qkv_flops + attn_score_flops + score_v_flops + out_proj_flops
    
    # FFN FLOPs per layer (SwiGLU)
    ffn_flops = 3 * 2 * B * S * H * FF  # gate + up + down
    
    # Total forward FLOPs per layer
    flops_per_layer_fw = attention_flops + ffn_flops
    
    # 简化公式: 6 * B * S * H^2 * (1 + FF/H + S/H)
    # 对于标准 FFN (FF = 4H): 约 24 * B * S * H^2 (如果忽略attention score)
    
    # 总 Forward FLOPs
    total_forward_flops = num_layers * flops_per_layer_fw
    
    # Embedding + LM Head FLOPs
    embedding_flops = 0  # Embedding is lookup, not matmul
    lm_head_flops = 2 * B * S * H * vocab_size
    total_forward_flops += lm_head_flops
    
    # Backward 约为 Forward 的 2 倍
    total_backward_flops = 2 * total_forward_flops
    
    # 单次 iteration 总 FLOPs (micro_batch)
    total_flops_per_micro = total_forward_flops + total_backward_flops
    
    # 完整 batch 的 FLOPs
    num_microbatches = batch_size // (micro_batch * dp)
    total_flops_per_iteration = total_flops_per_micro * num_microbatches * dp
    
    # ========== 内存计算 (单个 GPU) ==========
    # 与 Calculon 对齐的内存模型
    dtype_bytes = 2  # FP16
    
    # 每层权重参数 (Transformer block) - 与 Calculon 对齐
    # Calculon 使用标准 GPT 架构:
    # - Attention: Q, K, V, O projections
    #   - Q: H × (heads × attn_size / TP)
    #   - K, V, O: 同上
    #   - 总计: 4 × H × H / TP (因为 heads × attn_size = H)
    # - FFN: 2 个线性层 (不是 SwiGLU 的 3 个)
    #   - Mlp1: H × (FF / TP)
    #   - Mlp2: (FF / TP) × H
    #   - 总计: 2 × H × FF / TP
    attn_params_per_layer = 4 * hidden * hidden / tp  # Q, K, V, O
    ffn_params_per_layer = 2 * hidden * feedforward / tp  # Mlp1, Mlp2 (Calculon 标准)
    params_per_layer_per_gpu = attn_params_per_layer + ffn_params_per_layer
    
    # Embedding 和 LM head 通常不被 TP 分片（或只分片一部分）
    # 这里简化：只考虑 transformer 层的权重
    layers_per_stage = num_layers // pp
    
    # 模型参数内存
    model_memory = params_per_layer_per_gpu * layers_per_stage * dtype_bytes
    
    # 每个 GPU 的参数数量（用于优化器计算）
    params_per_gpu = params_per_layer_per_gpu * layers_per_stage
    
    # 优化器状态 (Adam: master weights FP32 + m FP32 + v FP32)
    # Calculon: block_optimizer_space = 6 * block_weight_space (FP32 vs FP16)
    # 12 bytes / 2 bytes = 6x
    optimizer_memory = params_per_gpu * 12  # 12 bytes per param (3 * FP32)
    
    # 梯度内存 (与权重相同)
    gradient_memory = model_memory
    
    # 激活内存 (与 Calculon 的 act_space 对齐)
    # Calculon 计算: act_space = working + storage * (blocks * microbatches - 1)
    # 
    # Calculon 的激活空间包括:
    # - block_act_working_space: 用于计算的临时空间 (~13x B*S*H)
    #   包括 attention 中间结果、FFN 中间结果等
    # - block_act_storage_space: 需要保存用于反向传播的激活 (~5x B*S*H)
    
    # 基于 Calculon 的实际数据校准系数:
    # - working_space ≈ 13 * B * S * H * dtype_bytes (每层)
    # - storage_space ≈ 5 * B * S * H * dtype_bytes (每层, attn_only mode)
    act_working_per_layer = 13 * B * S * hidden * dtype_bytes
    
    if training_config.get("gradient_checkpointing", False):
        # attn_only recompute: 保存较少激活
        act_storage_per_layer = 5 * B * S * hidden * dtype_bytes
    else:
        # 完整保存: 保存更多激活
        act_storage_per_layer = 10 * B * S * hidden * dtype_bytes
    
    # 考虑 microbatch 数量对激活内存的影响 (1F1B schedule)
    num_microbatches = batch_size // (micro_batch * dp)
    pipeline_bubble_mb = min(pp, num_microbatches)  # 1F1B schedule 稳态保存 PP 个微批激活
    
    # Calculon 公式: working + storage * (blocks * microbatches - 1)
    activation_memory = act_working_per_layer + \
                        act_storage_per_layer * (layers_per_stage * pipeline_bubble_mb - 1)
    
    # 总内存
    total_memory_per_gpu = model_memory + optimizer_memory + gradient_memory + activation_memory
    
    return {
        "total_params": total_params,
        "params_per_layer": params_per_layer,
        "params_per_gpu": params_per_gpu,
        "num_layers": num_layers,
        "hidden": hidden,
        "total_forward_flops": total_forward_flops,
        "total_backward_flops": total_backward_flops,
        "total_flops_per_micro": total_flops_per_micro,
        "total_flops_per_iteration": total_flops_per_iteration,
        "model_memory_gb": model_memory / 1e9,
        "optimizer_memory_gb": optimizer_memory / 1e9,
        "gradient_memory_gb": gradient_memory / 1e9,
        "activation_memory_gb": activation_memory / 1e9,
        "total_memory_per_gpu_gb": total_memory_per_gpu / 1e9,
        "num_microbatches": num_microbatches,
        "layers_per_stage": layers_per_stage,
    }


def format_bytes(n: float) -> str:
    """格式化字节数为人类可读格式。"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(n) < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PB"


def format_flops(n: float) -> str:
    """格式化 FLOPs 为人类可读格式。"""
    for unit in ['FLOPs', 'KFLOPs', 'MFLOPs', 'GFLOPs', 'TFLOPs', 'PFLOPs']:
        if abs(n) < 1000.0:
            return f"{n:.2f} {unit}"
        n /= 1000.0
    return f"{n:.2f} EFLOPs"


def format_time(seconds: float) -> str:
    """格式化时间为人类可读格式。"""
    if seconds < 1e-6:
        return f"{seconds * 1e9:.2f} ns"
    elif seconds < 1e-3:
        return f"{seconds * 1e6:.2f} µs"
    elif seconds < 1:
        return f"{seconds * 1e3:.2f} ms"
    else:
        return f"{seconds:.2f} s"


def print_separator(title: str = "", char: str = "=", width: int = 70):
    """打印分隔线。"""
    if title:
        padding = (width - len(title) - 2) // 2
        print(f"\n{char * padding} {title} {char * padding}")
    else:
        print(char * width)


def analyze_graph(graph: GraphIR) -> Dict[str, Any]:
    """分析 GraphIR 统计信息。"""
    op_counts = {}
    for node in graph.nodes.values():
        op_type = node.op_type
        op_counts[op_type] = op_counts.get(op_type, 0) + 1
    
    return {
        "total_nodes": len(graph.nodes),
        "total_edges": len(graph.edges),
        "total_tensors": len(graph.tensors),
        "op_counts": op_counts,
    }


def print_result_summary(
    result: SimulationResult,
    timeline: TimelineIR,
    config: Dict[str, Any],
):
    """打印仿真结果摘要。"""
    print_separator("仿真结果摘要")
    
    print(f"\n{'='*60}")
    print(f"  模型: GPT-175B")
    print(f"  系统: H100-80G-NVL8")
    print(f"{'='*60}")
    
    # 并行配置
    print(f"\n【并行配置】")
    print(f"  Tensor Parallel (TP):   {config.get('tp', 1)}")
    print(f"  Pipeline Parallel (PP): {config.get('pp', 1)}")
    print(f"  Data Parallel (DP):     {config.get('dp', 1)}")
    print(f"  总GPU数:                {config.get('tp', 1) * config.get('pp', 1) * config.get('dp', 1)}")
    
    # 训练配置
    print(f"\n【训练配置】")
    print(f"  Global Batch Size:      {config.get('batch_size', 0)}")
    print(f"  Micro Batch Size:       {config.get('micro_batch', 0)}")
    print(f"  Sequence Length:        {config.get('seq_len', 0)}")
    
    # 内存分析
    print(f"\n【内存分析】")
    print(f"  峰值内存:               {format_bytes(result.peak_memory)}")
    if result.memory_breakdown:
        mb = result.memory_breakdown
        print(f"    - 权重:               {format_bytes(mb.weights)}")
        print(f"    - 激活:               {format_bytes(mb.activations)}")
        print(f"    - 梯度:               {format_bytes(mb.gradients)}")
        print(f"    - 优化器状态:         {format_bytes(mb.optimizer_states)}")
    print(f"  内存利用率:             {result.memory_utilization:.1%}")
    print(f"  内存是否足够:           {'✓ 是' if result.is_feasible() else '✗ 否 (OOM)'}")
    
    # 性能分析
    print(f"\n【性能分析】")
    print(f"  端到端时间:             {format_time(result.e2e_time)}")
    if result.time_breakdown:
        tb = result.time_breakdown
        print(f"    - 前向传播:           {format_time(tb.forward)}")
        print(f"    - 反向传播:           {format_time(tb.backward)}")
        print(f"    - 通信:               {format_time(tb.communication)}")
        print(f"    - 流水线气泡:         {format_time(tb.bubble)}")
    
    print(f"  总 FLOPs:               {format_flops(result.total_flops)}")
    print(f"  吞吐 FLOPs:             {format_flops(result.achieved_flops)}")
    print(f"  MFU (模型利用率):       {result.mfu:.1%}")
    
    # Timeline 分析
    print(f"\n【Timeline 分析】")
    print(f"  事件总数:               {len(timeline.events)}")
    
    # 分设备统计
    for dev in range(timeline.num_devices):
        compute = timeline.compute_time(dev)
        comm = timeline.comm_time(dev)
        overlap = timeline.overlap_ratio(dev)
        
        if isinstance(compute, (int, float)) and isinstance(comm, (int, float)):
            print(f"  Device {dev}:")
            print(f"    - 计算时间:           {format_time(compute)}")
            print(f"    - 通信时间:           {format_time(comm)}")
            print(f"    - 重叠率:             {overlap:.1%}")
    
    # 警告
    if result.warnings:
        print(f"\n【警告】")
        for warning in result.warnings:
            print(f"  ⚠ {warning}")
    
    print(f"\n{'='*60}\n")


def export_chrome_trace(timeline: TimelineIR, output_path: str):
    """导出 Chrome Trace 格式用于可视化。
    
    可以在 chrome://tracing 或 https://ui.perfetto.dev/ 中打开。
    """
    trace_events = timeline.to_trace_events()
    
    trace_data = {
        "traceEvents": trace_events,
        "metadata": {
            "model": "GPT-175B",
            "system": "H100-NVL8",
            **timeline.metadata
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(trace_data, f, indent=2, default=str)
    
    print(f"Chrome Trace 已导出到: {output_path}")
    print(f"可以在 chrome://tracing 或 https://ui.perfetto.dev/ 中查看")


# ============================================================================
# Calculon 对比
# ============================================================================

def run_calculon_simulation(
    model_config: Dict[str, Any],
    training_config: Dict[str, Any],
    system_file: str,
) -> Dict[str, Any]:
    """使用 Calculon 运行仿真并返回结果。
    
    Args:
        model_config: 模型配置
        training_config: 训练配置
        system_file: 系统配置文件路径
        
    Returns:
        Calculon 仿真结果字典
    """
    # 创建 Calculon Application 配置
    app_config = {
        "hidden": model_config["hidden"],
        "feedforward": model_config["feedforward"],
        "seq_size": model_config["seq_len"],
        "attn_heads": model_config["num_heads"],
        "attn_size": model_config["head_dim"],
        "num_blocks": model_config["num_layers"],
    }
    
    tp = training_config["tensor_parallel"]
    pp = training_config["pipeline_parallel"]
    dp = training_config["data_parallel"]
    
    # 创建 Calculon Execution 配置
    exe_config = {
        "num_procs": tp * pp * dp,
        "tensor_par": tp,
        "pipeline_par": pp,
        "data_par": dp,
        "tensor_par_net": 0,      # 使用第一个网络层级
        "pipeline_par_net": 0 if pp <= 8 else 1,  # 节点内/节点间
        "data_par_net": 1,        # 使用第二个网络层级
        "batch_size": training_config["batch_size"],
        "microbatch_size": training_config["micro_batch_size"],
        "datatype": "float16",
        "fused_activation": True,
        "attention_type": "multihead",
        "activation_recompute": "attn_only" if training_config.get("gradient_checkpointing", False) else "none",
        "pipeline_interleaving": 1,
        "optimizer_sharding": dp > 1,
        "tensor_par_comm_type": "ar",
        "tensor_par_overlap": "none",
        "seq_par_ag_redo": False,
        "data_par_overlap": dp > 1,
        "weight_offload": False,
        "activations_offload": False,
        "optimizer_offload": False,
        "training": True,
    }
    
    # 加载系统配置
    sys_json = calculon.io.read_json_file(system_file)
    
    # 创建 Calculon 对象
    app = Llm.Application(app_config)
    exe = Llm.Execution.from_json(exe_config)
    syst = System(sys_json)
    
    # 运行仿真
    import logging
    logger = logging.getLogger("calculon")
    
    try:
        model = Llm(app, logger)
        model.compile(syst, exe)
        model.run(syst)
        
        # 获取结果
        stats = model.get_stats_json(include_layers=False)
        
        return {
            "success": True,
            "stats": stats,
            "num_parameters": app.num_parameters(),
        }
    except Llm.Error as e:
        return {
            "success": False,
            "error": str(e),
        }


def compare_with_calculon(
    ir_metrics: Dict[str, Any],
    calculon_result: Dict[str, Any],
    training_config: Dict[str, Any],
):
    """对比 IR 编译器和 Calculon 的结果。
    
    Args:
        ir_metrics: IR 编译器计算的指标
        calculon_result: Calculon 仿真结果
        training_config: 训练配置
    """
    print_separator("IR 编译器 vs Calculon 对比")
    
    if not calculon_result.get("success", False):
        print(f"\n⚠ Calculon 仿真失败: {calculon_result.get('error', 'Unknown error')}")
        return
    
    stats = calculon_result["stats"]
    
    # 表头
    print(f"\n{'指标':<30} {'IR 编译器':<20} {'Calculon':<20} {'差异':<15}")
    print("-" * 85)
    
    # 参数量对比
    ir_params = ir_metrics["total_params"]
    calc_params = calculon_result["num_parameters"]
    diff_params = (ir_params - calc_params) / calc_params * 100 if calc_params > 0 else 0
    print(f"{'参数量':<30} {ir_params/1e9:>15.2f}B {calc_params/1e9:>15.2f}B {diff_params:>+12.1f}%")
    
    # 内存对比 (每 GPU)
    ir_weight_mem = ir_metrics["model_memory_gb"]
    calc_weight_mem = stats.get("weight_space", 0) / 1e9
    diff_weight = (ir_weight_mem - calc_weight_mem) / calc_weight_mem * 100 if calc_weight_mem > 0 else 0
    print(f"{'权重内存 (GB)':<30} {ir_weight_mem:>15.2f} {calc_weight_mem:>15.2f} {diff_weight:>+12.1f}%")
    
    ir_act_mem = ir_metrics["activation_memory_gb"]
    calc_act_mem = stats.get("act_space", 0) / 1e9
    diff_act = (ir_act_mem - calc_act_mem) / calc_act_mem * 100 if calc_act_mem > 0 else 0
    print(f"{'激活内存 (GB)':<30} {ir_act_mem:>15.2f} {calc_act_mem:>15.2f} {diff_act:>+12.1f}%")
    
    ir_opt_mem = ir_metrics["optimizer_memory_gb"]
    calc_opt_mem = stats.get("optimizer_space", 0) / 1e9
    diff_opt = (ir_opt_mem - calc_opt_mem) / calc_opt_mem * 100 if calc_opt_mem > 0 else 0
    print(f"{'优化器内存 (GB)':<30} {ir_opt_mem:>15.2f} {calc_opt_mem:>15.2f} {diff_opt:>+12.1f}%")
    
    ir_total_mem = ir_metrics["total_memory_per_gpu_gb"]
    calc_total_mem = stats.get("proc_mem_tier1_cap_req", 0) / 1e9
    diff_total = (ir_total_mem - calc_total_mem) / calc_total_mem * 100 if calc_total_mem > 0 else 0
    print(f"{'总内存需求 (GB)':<30} {ir_total_mem:>15.2f} {calc_total_mem:>15.2f} {diff_total:>+12.1f}%")
    
    print("-" * 85)
    
    # 时间对比
    # Calculon 返回的是单个 iteration 的时间（秒）
    calc_total_time = stats.get("total_time", 0)  # 秒
    calc_fw_time = stats.get("fw_time", 0)
    calc_bw_time = stats.get("bw_time", 0)
    calc_bubble_time = stats.get("bubble_time", 0)
    calc_optim_time = stats.get("optim_step_time", 0)
    calc_tp_comm_time = stats.get("tp_comm_exposed_time", 0)
    calc_dp_comm_time = stats.get("dp_comm_exposed_time", 0)
    
    tp = training_config.get("tensor_parallel", 1)
    pp = training_config.get("pipeline_parallel", 1)
    dp = training_config.get("data_parallel", 1)
    total_gpus = tp * pp * dp
    batch_size = training_config.get("batch_size", 1)
    micro_batch = training_config.get("micro_batch_size", 1)
    num_microbatches = ir_metrics.get("num_microbatches", batch_size // (micro_batch * dp))
    
    # ========== IR 编译器的时间 ==========
    # 优先使用 IR 编译器实际输出的时间
    ir_iteration_time = ir_metrics.get("ir_iteration_time", None)
    ir_micro_batch_time = ir_metrics.get("ir_micro_batch_time", None)
    ir_compute_time_actual = ir_metrics.get("ir_compute_time", None)
    ir_comm_time_actual = ir_metrics.get("ir_comm_time", None)
    
    # 如果 IR 没有实际时间输出，使用理论估计
    if ir_iteration_time is None:
        # 峰值 FLOPs (H100: 1000 TFLOPs for FP16)
        peak_flops_per_gpu = 1000 * 1e12  # H100 FP16
        
        # 理论计算时间 (100% 利用率)
        ir_theoretical_time = ir_metrics["total_flops_per_iteration"] / (peak_flops_per_gpu * total_gpus)
        
        # 估计各阶段时间 (基于典型比例)
        ir_fw_time = ir_theoretical_time * 0.33 / 0.5  # 假设 50% 利用率
        ir_bw_time = ir_theoretical_time * 0.67 / 0.5
        
        # Bubble 时间 (1F1B)
        bubble_ratio = (pp - 1) / num_microbatches if num_microbatches > 0 and pp > 1 else 0
        ir_bubble_time = (ir_fw_time + ir_bw_time) * bubble_ratio
        
        # 通信时间估计
        seq_len = 2048
        hidden = ir_metrics.get("hidden", 5120)
        num_layers = ir_metrics.get("num_layers", 40)
        nvlink_bw = 450 * 1e9 * 0.65
        activation_bytes = micro_batch * seq_len * hidden * 2
        ir_tp_comm_time = 2 * (tp - 1) / tp * activation_bytes / nvlink_bw * num_layers * 2 if tp > 1 else 0
        
        ir_total_time = ir_fw_time + ir_bw_time + max(ir_tp_comm_time, 0) + ir_bubble_time
        time_source = "理论估计"
    else:
        # 使用 IR 编译器实际输出的时间
        ir_total_time = ir_iteration_time
        
        # 从实际值计算 FW/BW/通信 时间
        if ir_compute_time_actual is not None and isinstance(ir_compute_time_actual, (int, float)):
            # FW:BW 大约是 1:2
            ir_fw_time = ir_compute_time_actual * 0.33 * num_microbatches
            ir_bw_time = ir_compute_time_actual * 0.67 * num_microbatches
        else:
            ir_fw_time = ir_total_time * 0.33
            ir_bw_time = ir_total_time * 0.67
        
        if ir_comm_time_actual is not None and isinstance(ir_comm_time_actual, (int, float)):
            ir_tp_comm_time = ir_comm_time_actual * num_microbatches
        else:
            ir_tp_comm_time = 0
        
        # Bubble 时间
        bubble_ratio = (pp - 1) / num_microbatches if num_microbatches > 0 and pp > 1 else 0
        ir_bubble_time = (ir_fw_time + ir_bw_time) * bubble_ratio
        
        time_source = "IR 编译器实际输出"
    
    print(f"\n【时间来源】IR: {time_source}, Calculon: 实际仿真")
    
    # 时间对比表
    print(f"\n{'时间指标':<30} {'IR 编译器':<20} {'Calculon':<20} {'差异':<15}")
    print("-" * 85)
    
    def safe_diff(ir_val, calc_val):
        if calc_val > 0:
            return (ir_val - calc_val) / calc_val * 100
        return 0
    
    diff_fw = safe_diff(ir_fw_time, calc_fw_time)
    print(f"{'前向时间 (ms)':<30} {ir_fw_time*1e3:>15.2f} {calc_fw_time*1e3:>15.2f} {diff_fw:>+12.1f}%")
    
    diff_bw = safe_diff(ir_bw_time, calc_bw_time)
    print(f"{'反向时间 (ms)':<30} {ir_bw_time*1e3:>15.2f} {calc_bw_time*1e3:>15.2f} {diff_bw:>+12.1f}%")
    
    diff_tp = safe_diff(ir_tp_comm_time, calc_tp_comm_time)
    print(f"{'TP 通信时间 (ms)':<30} {ir_tp_comm_time*1e3:>15.2f} {calc_tp_comm_time*1e3:>15.2f} {diff_tp:>+12.1f}%")
    
    diff_bubble = safe_diff(ir_bubble_time, calc_bubble_time)
    print(f"{'Bubble 时间 (ms)':<30} {ir_bubble_time*1e3:>15.2f} {calc_bubble_time*1e3:>15.2f} {diff_bubble:>+12.1f}%")
    
    diff_total_time = safe_diff(ir_total_time, calc_total_time)
    print(f"{'总时间 (ms)':<30} {ir_total_time*1e3:>15.2f} {calc_total_time*1e3:>15.2f} {diff_total_time:>+12.1f}%")
    
    print("-" * 85)
    
    # 效率指标对比
    calc_compute_eff = stats.get("compute_efficiency", 0) * 100
    calc_system_eff = stats.get("system_efficiency", 0) * 100
    calc_total_eff = stats.get("total_efficiency", 0) * 100
    
    # IR 效率计算 - 基于理论最小时间
    # 理论最小时间 = FLOPs / (峰值计算能力 * GPU数)
    peak_flops_per_gpu = 1000 * 1e12  # H100 FP16
    ir_theoretical_min_time = ir_metrics["total_flops_per_iteration"] / (peak_flops_per_gpu * total_gpus)
    
    # MFU = 理论最小时间 / 实际时间 = 硬件利用率
    ir_mfu = ir_theoretical_min_time / ir_total_time * 100 if ir_total_time > 0 else 0
    
    # 计算效率 = 只考虑计算时间的效率
    ir_pure_compute_time = ir_fw_time + ir_bw_time
    ir_compute_eff = ir_theoretical_min_time / ir_pure_compute_time * 100 if ir_pure_compute_time > 0 else 0
    
    # 系统效率 = 计算时间 / 总时间（不含通信和 bubble 的纯计算占比）
    ir_system_eff = ir_pure_compute_time / ir_total_time * 100 if ir_total_time > 0 else 0
    
    print(f"{'计算效率':<30} {ir_compute_eff:>14.1f}% {calc_compute_eff:>14.1f}% {safe_diff(ir_compute_eff, calc_compute_eff):>+12.1f}%")
    print(f"{'系统效率':<30} {ir_system_eff:>14.1f}% {calc_system_eff:>14.1f}% {safe_diff(ir_system_eff, calc_system_eff):>+12.1f}%")
    print(f"{'总效率 (MFU)':<30} {ir_mfu:>14.1f}% {calc_total_eff:>14.1f}% {safe_diff(ir_mfu, calc_total_eff):>+12.1f}%")
    
    print("-" * 85)
    
    # 吞吐量对比
    calc_sample_rate = stats.get("sample_rate", 0)
    calc_throughput = calc_sample_rate * 2048 if calc_sample_rate > 0 else 0
    
    # IR 吞吐量
    ir_sample_rate = batch_size / ir_total_time if ir_total_time > 0 else 0
    ir_throughput = ir_sample_rate * 2048
    
    diff_sample = safe_diff(ir_sample_rate, calc_sample_rate)
    print(f"{'样本吞吐 (samples/s)':<30} {ir_sample_rate:>15.2f} {calc_sample_rate:>15.2f} {diff_sample:>+12.1f}%")
    
    diff_throughput = safe_diff(ir_throughput, calc_throughput)
    print(f"{'Token 吞吐 (tokens/s)':<30} {ir_throughput:>15,.0f} {calc_throughput:>15,.0f} {diff_throughput:>+12.1f}%")
    
    print("\n" + "=" * 85)
    
    # 返回对比数据用于进一步分析
    return {
        "ir": ir_metrics,
        "calculon": stats,
        "diff_params_pct": diff_params,
        "diff_total_mem_pct": diff_total,
    }


def fine_grained_time_analysis(
    timeline_ir: TimelineIR,
    calculon_stats: Dict[str, Any],
    model_config: Dict[str, Any],
    training_config: Dict[str, Any],
):
    """细粒度时间分析，用于校准 IR 编译器的精度。
    
    对比单层 Transformer block 的各个阶段时间。
    """
    print_separator("细粒度时间分析 (单层 Block)")
    
    num_layers = model_config.get("num_layers", 40)
    num_microbatches = training_config.get("batch_size", 1) // training_config.get("micro_batch_size", 1)
    
    # ========== Calculon 单层时间分解 ==========
    print("\n【Calculon 单层时间分解】")
    
    # Block 级别时间 (单层，单个 micro batch)
    calc_block_fw_time = calculon_stats.get("block_fw_time", 0)  # 前向总时间
    calc_block_fw_flops_time = calculon_stats.get("block_fw_flops_time", 0)  # 前向计算时间
    calc_block_fw_mem_time = calculon_stats.get("block_fw_mem_time", 0)  # 前向内存时间
    
    calc_block_agrad_time = calculon_stats.get("block_agrad_time", 0)  # 激活梯度时间
    calc_block_agrad_flops_time = calculon_stats.get("block_agrad_flops_time", 0)
    calc_block_agrad_mem_time = calculon_stats.get("block_agrad_mem_time", 0)
    
    calc_block_wgrad_time = calculon_stats.get("block_wgrad_time", 0)  # 权重梯度时间
    calc_block_wgrad_flops_time = calculon_stats.get("block_wgrad_flops_time", 0)
    calc_block_wgrad_mem_time = calculon_stats.get("block_wgrad_mem_time", 0)
    
    calc_block_optim_time = calculon_stats.get("block_optim_time", 0)  # 优化器时间
    calc_block_re_time = calculon_stats.get("block_re_time", 0)  # 重计算时间
    
    # TP 通信时间 (单层)
    calc_fw_tp_time = calculon_stats.get("baseblock_fw_tp_time_exposed", 0)
    calc_agrad_tp_time = calculon_stats.get("baseblock_agrad_tp_time_exposed", 0)
    
    # FLOPs (单层)
    calc_block_fw_flops = calculon_stats.get("block_fw_flops", 0)
    calc_block_agrad_flops = calculon_stats.get("block_agrad_flops", 0)
    calc_block_wgrad_flops = calculon_stats.get("block_wgrad_flops", 0)
    
    # 内存访问 (单层)
    calc_block_fw_mem = calculon_stats.get("block_fw_mem_accessed", 0)
    calc_block_agrad_mem = calculon_stats.get("block_agrad_mem_accessed", 0)
    calc_block_wgrad_mem = calculon_stats.get("block_wgrad_mem_accessed", 0)
    
    print(f"  前向 (Forward):")
    print(f"    计算时间:    {calc_block_fw_flops_time*1e6:>10.2f} µs  (FLOPs: {calc_block_fw_flops/1e9:.2f} GFLOPs)")
    print(f"    内存时间:    {calc_block_fw_mem_time*1e6:>10.2f} µs  (访问: {calc_block_fw_mem/1e6:.2f} MB)")
    print(f"    总时间:      {calc_block_fw_time*1e6:>10.2f} µs  (计算受限: {calc_block_fw_flops_time < calc_block_fw_mem_time})")
    print(f"    TP 通信:     {calc_fw_tp_time*1e6:>10.2f} µs")
    
    print(f"  激活梯度 (Activation Gradient):")
    print(f"    计算时间:    {calc_block_agrad_flops_time*1e6:>10.2f} µs  (FLOPs: {calc_block_agrad_flops/1e9:.2f} GFLOPs)")
    print(f"    内存时间:    {calc_block_agrad_mem_time*1e6:>10.2f} µs  (访问: {calc_block_agrad_mem/1e6:.2f} MB)")
    print(f"    总时间:      {calc_block_agrad_time*1e6:>10.2f} µs")
    print(f"    TP 通信:     {calc_agrad_tp_time*1e6:>10.2f} µs")
    
    print(f"  权重梯度 (Weight Gradient):")
    print(f"    计算时间:    {calc_block_wgrad_flops_time*1e6:>10.2f} µs  (FLOPs: {calc_block_wgrad_flops/1e9:.2f} GFLOPs)")
    print(f"    内存时间:    {calc_block_wgrad_mem_time*1e6:>10.2f} µs  (访问: {calc_block_wgrad_mem/1e6:.2f} MB)")
    print(f"    总时间:      {calc_block_wgrad_time*1e6:>10.2f} µs")
    
    print(f"  优化器:        {calc_block_optim_time*1e6:>10.2f} µs")
    print(f"  重计算:        {calc_block_re_time*1e6:>10.2f} µs")
    
    calc_block_bw_time = calc_block_agrad_time + calc_block_wgrad_time
    calc_block_total = calc_block_fw_time + calc_block_bw_time + calc_block_optim_time
    print(f"  ------------------------------------")
    print(f"  单层总计:      {calc_block_total*1e6:>10.2f} µs (FW: {calc_block_fw_time/calc_block_total*100:.1f}%, BW: {calc_block_bw_time/calc_block_total*100:.1f}%, Optim: {calc_block_optim_time/calc_block_total*100:.1f}%)")
    
    # ========== 端到端时间分解（解释差异来源）==========
    calc_fw_time_total = calculon_stats.get("fw_time", 0)
    calc_bw_time_total = calculon_stats.get("bw_time", 0)
    calc_optim_time_total = calculon_stats.get("optim_step_time", 0)
    calc_recompute_time_total = calculon_stats.get("recompute_time", 0)
    calc_tp_comm_exposed = calculon_stats.get("tp_comm_exposed_time", 0)
    calc_total_time = calculon_stats.get("total_time", 0)
    
    print(f"\n【Calculon 端到端时间分解】(解释单层 vs 端到端差异)")
    print(f"  前向计算:      {calc_fw_time_total*1e3:>10.2f} ms ({calc_fw_time_total/calc_total_time*100:.1f}%)")
    print(f"  反向计算:      {calc_bw_time_total*1e3:>10.2f} ms ({calc_bw_time_total/calc_total_time*100:.1f}%)")
    print(f"  优化器步骤:    {calc_optim_time_total*1e3:>10.2f} ms ({calc_optim_time_total/calc_total_time*100:.1f}%)")
    print(f"  重计算时间:    {calc_recompute_time_total*1e3:>10.2f} ms ({calc_recompute_time_total/calc_total_time*100:.1f}%)")
    print(f"  TP通信暴露:    {calc_tp_comm_exposed*1e3:>10.2f} ms ({calc_tp_comm_exposed/calc_total_time*100:.1f}%)")
    print(f"  ------------------------------------")
    print(f"  总计:          {calc_total_time*1e3:>10.2f} ms")
    
    # 组成分析
    compute_part = (calc_fw_time_total + calc_bw_time_total) / calc_total_time * 100
    comm_part = calc_tp_comm_exposed / calc_total_time * 100
    overhead_part = (calc_optim_time_total + calc_recompute_time_total) / calc_total_time * 100
    print(f"  组成: 计算 {compute_part:.1f}% + 通信 {comm_part:.1f}% + 开销 {overhead_part:.1f}%")
    print(f"\n  ⚠ 注意: 单层对比只比较计算时间，端到端还包含通信暴露+重计算")
    
    # ========== IR 编译器单层时间分解 ==========
    print("\n【IR 编译器单层时间分解】")
    
    # 从 Timeline 提取每种操作的时间
    ir_times_by_op = {}
    ir_flops_by_op = {}
    ir_mem_by_op = {}
    
    # 跟踪事件开始时间以计算 duration
    # TimelineEvent 使用 time 属性，而不是 start/end
    compute_starts = {}  # resource_id -> (time, op_type, metadata)
    comm_starts = {}     # resource_id -> time
    
    for event in timeline_ir.events:
        if event.event_type == EventType.COMPUTE_START:
            compute_starts[event.resource_id] = (event.time, event.op_type, event.metadata)
        
        elif event.event_type == EventType.COMPUTE_END:
            if event.resource_id in compute_starts:
                start_time, op_type, metadata = compute_starts[event.resource_id]
                end_time = event.time
                
                # 计算 duration
                if isinstance(start_time, (int, float)) and isinstance(end_time, (int, float)):
                    duration = end_time - start_time
                    ir_times_by_op[op_type] = ir_times_by_op.get(op_type, 0) + duration
                    
                    # 获取 FLOPs 和内存
                    flops = metadata.get("flops_fw", 0) or metadata.get("flops", 0)
                    if isinstance(flops, (int, float)):
                        ir_flops_by_op[op_type] = ir_flops_by_op.get(op_type, 0) + flops
                    
                    mem = metadata.get("memory_fw", 0) or metadata.get("memory", 0)
                    if isinstance(mem, (int, float)):
                        ir_mem_by_op[op_type] = ir_mem_by_op.get(op_type, 0) + mem
        
        elif event.event_type == EventType.COMM_START:
            comm_starts[event.resource_id] = event.time
        
        elif event.event_type == EventType.COMM_END:
            if event.resource_id in comm_starts:
                start_time = comm_starts[event.resource_id]
                end_time = event.time
                
                if isinstance(start_time, (int, float)) and isinstance(end_time, (int, float)):
                    duration = end_time - start_time
                    ir_times_by_op["Communication"] = ir_times_by_op.get("Communication", 0) + duration
    
    # 计算 IR 的单层时间（分别统计计算、通信、优化器）
    ir_compute_ops = ["Linear", "Attention", "RMSNorm", "Add", "SiLU", "Mul", "Softmax"]
    ir_pure_compute = sum(ir_times_by_op.get(op, 0) for op in ir_compute_ops if isinstance(ir_times_by_op.get(op, 0), (int, float)))
    ir_comm_time = ir_times_by_op.get("Communication", 0) if isinstance(ir_times_by_op.get("Communication", 0), (int, float)) else 0
    ir_optim_time = ir_times_by_op.get("OptimizerStep", 0) if isinstance(ir_times_by_op.get("OptimizerStep", 0), (int, float)) else 0
    ir_total_all = ir_pure_compute + ir_comm_time + ir_optim_time
    
    # 单层时间（只算纯计算，不含通信和优化器，以便与 Calculon 对比）
    ir_compute_per_layer = ir_pure_compute / num_layers if num_layers > 0 else ir_pure_compute
    ir_per_layer_time = ir_total_all / num_layers if num_layers > 0 else ir_total_all  # 包含所有操作
    
    print(f"  操作类型时间分布 (所有层总计):")
    for op_type, time in sorted(ir_times_by_op.items(), key=lambda x: -x[1] if isinstance(x[1], (int, float)) else 0):
        if isinstance(time, (int, float)):
            flops = ir_flops_by_op.get(op_type, 0)
            mem = ir_mem_by_op.get(op_type, 0)
            print(f"    {op_type:<20} {time*1e3:>10.3f} ms  (FLOPs: {flops/1e9:.2f}G, Mem: {mem/1e6:.2f}MB)")
    
    print(f"  ------------------------------------")
    print(f"  IR 单层 (纯计算): {ir_compute_per_layer*1e6:>8.2f} µs")
    print(f"  IR 单层 (含全部): {ir_per_layer_time*1e6:>8.2f} µs (计算+通信+优化器)")
    
    # ========== 细粒度对比表格 ==========
    print("\n【单层时间对比 (校准分析)】")
    print(f"\n{'阶段':<25} {'IR (µs)':<15} {'Calculon (µs)':<15} {'差异':<12} {'说明':<30}")
    print("-" * 100)
    
    def safe_compare(ir_val, calc_val, name, note=""):
        diff = (ir_val - calc_val) / calc_val * 100 if calc_val > 0 else 0
        status = "✓" if abs(diff) < 20 else "⚠" if abs(diff) < 50 else "✗"
        print(f"{name:<25} {ir_val:>12.2f} {calc_val:>12.2f} {diff:>+10.1f}% {status} {note}")
        return diff
    
    # IR 各操作类型的单层时间
    linear_time = ir_times_by_op.get("Linear", 0)
    attention_time = ir_times_by_op.get("Attention", 0)
    norm_time = ir_times_by_op.get("RMSNorm", 0) + ir_times_by_op.get("LayerNorm", 0)
    activation_time = ir_times_by_op.get("SiLU", 0) + ir_times_by_op.get("GELU", 0)
    comm_time = ir_times_by_op.get("Communication", 0) + ir_times_by_op.get("AllReduce", 0)
    
    # 单层时间（除以层数）
    ir_linear_per_layer = linear_time / num_layers if num_layers > 0 else linear_time
    ir_attn_per_layer = attention_time / num_layers if num_layers > 0 else attention_time
    ir_norm_per_layer = norm_time / num_layers if num_layers > 0 else norm_time
    ir_comm_per_layer = comm_time / num_layers if num_layers > 0 else comm_time
    
    # 使用纯计算时间估算前向/反向 (约 1/3 vs 2/3 比例)
    # 这样与 Calculon 的 block_fw_time 和 block_bw_time 定义一致
    ir_fw_per_layer = ir_compute_per_layer * 0.33
    ir_bw_per_layer = ir_compute_per_layer * 0.67
    
    safe_compare(ir_fw_per_layer * 1e6, calc_block_fw_time * 1e6, "前向总时间", "IR 估计")
    safe_compare(ir_bw_per_layer * 1e6, calc_block_bw_time * 1e6, "反向总时间", "IR 估计")
    safe_compare(ir_linear_per_layer * 1e6, (calc_block_fw_flops_time + calc_block_agrad_flops_time + calc_block_wgrad_flops_time) * 1e6 * 0.8, "Linear 操作", "占计算 ~80%")
    safe_compare(ir_comm_per_layer * 1e6, (calc_fw_tp_time + calc_agrad_tp_time) * 1e6, "TP 通信", "AllReduce")
    
    print("-" * 100)
    
    # ========== 端到端时间完整对比 ==========
    print("\n【端到端时间完整对比】")
    
    ir_total_time = timeline_ir.makespan(0)
    calc_total_time_val = calculon_stats.get("total_time", 0)
    
    # IR 编译器的端到端时间分解
    # 注意：ir_times_by_op 是单个 micro batch (所有层) 的时间
    ir_compute_per_mb = ir_times_by_op.get("Linear", 0) + ir_times_by_op.get("Attention", 0) + \
                        ir_times_by_op.get("RMSNorm", 0) + ir_times_by_op.get("Add", 0) + \
                        ir_times_by_op.get("SiLU", 0) + ir_times_by_op.get("Mul", 0)
    ir_comm_per_mb = ir_times_by_op.get("Communication", 0)
    ir_optim_time = ir_times_by_op.get("OptimizerStep", 0)  # OptimizerStep runs ONCE, not per microbatch
    ir_recompute_time = ir_times_by_op.get("RecomputeStep", 0)  # RecomputeStep runs ONCE (total for all microbatches)
    
    # 扩展到完整 iteration (× num_microbatches)
    ir_compute_total = ir_compute_per_mb * num_microbatches if isinstance(ir_compute_per_mb, (int, float)) else 0
    ir_comm_total = ir_comm_per_mb * num_microbatches if isinstance(ir_comm_per_mb, (int, float)) else 0
    # 优化器只执行一次，不乘以 microbatch 数量
    ir_optim_total = ir_optim_time if isinstance(ir_optim_time, (int, float)) else 0
    # 重计算时间已经是总时间（在 OptimizerPass 中计算为 per_mb × num_microbatches）
    ir_recompute_total = ir_recompute_time if isinstance(ir_recompute_time, (int, float)) else 0
    
    # 使用单层估计值来一致性验证
    # 单层时间 × 层数 × 微批数 应该约等于 ir_compute_total + ir_comm_total
    ir_e2e_from_layer = ir_per_layer_time * num_layers * num_microbatches if isinstance(ir_per_layer_time, (int, float)) else 0
    
    # Calculon 的端到端时间分解
    calc_fw_total = calculon_stats.get("fw_time", 0)
    calc_bw_total = calculon_stats.get("bw_time", 0)
    calc_compute_total = calc_fw_total + calc_bw_total
    calc_comm_total = calculon_stats.get("tp_comm_exposed_time", 0)
    calc_optim_total = calculon_stats.get("optim_step_time", 0)
    calc_recompute_total = calculon_stats.get("recompute_time", 0)
    
    print(f"\n{'时间组成':<20} {'IR 编译器 (ms)':<18} {'Calculon (ms)':<18} {'差异':<12}")
    print("-" * 70)
    
    def safe_diff_pct(ir_val, calc_val):
        if calc_val > 0:
            return (ir_val - calc_val) / calc_val * 100
        return 0
    
    diff_compute = safe_diff_pct(ir_compute_total * 1e3, calc_compute_total * 1e3)
    print(f"{'计算 (FW+BW)':<20} {ir_compute_total*1e3:>15.2f} {calc_compute_total*1e3:>15.2f} {diff_compute:>+10.1f}%")
    
    diff_comm = safe_diff_pct(ir_comm_total * 1e3, calc_comm_total * 1e3)
    print(f"{'通信暴露':<20} {ir_comm_total*1e3:>15.2f} {calc_comm_total*1e3:>15.2f} {diff_comm:>+10.1f}%")
    
    diff_optim = safe_diff_pct(ir_optim_total * 1e3, calc_optim_total * 1e3)
    print(f"{'优化器':<20} {ir_optim_total*1e3:>15.2f} {calc_optim_total*1e3:>15.2f} {diff_optim:>+10.1f}%")
    
    diff_recompute = safe_diff_pct(ir_recompute_total * 1e3, calc_recompute_total * 1e3)
    print(f"{'重计算':<20} {ir_recompute_total*1e3:>15.2f} {calc_recompute_total*1e3:>15.2f} {diff_recompute:>+10.1f}%")
    
    print("-" * 70)
    
    ir_e2e_total = ir_compute_total + ir_comm_total + ir_optim_total + ir_recompute_total
    calc_e2e_total = calc_compute_total + calc_comm_total + calc_optim_total + calc_recompute_total
    diff_e2e = safe_diff_pct(ir_e2e_total * 1e3, calc_e2e_total * 1e3)
    print(f"{'端到端总计':<20} {ir_e2e_total*1e3:>15.2f} {calc_e2e_total*1e3:>15.2f} {diff_e2e:>+10.1f}%")
    
    # 使用单层估计的一致性验证
    # 计算 IR 纯计算的单层时间
    ir_compute_per_layer_val = ir_pure_compute / num_layers if num_layers > 0 else 0
    ir_compute_e2e_from_layer = ir_compute_per_layer_val * num_layers * num_microbatches if isinstance(ir_compute_per_layer_val, (int, float)) else 0
    
    if ir_compute_e2e_from_layer > 0:
        print(f"\n  一致性验证 (纯计算对比):")
        print(f"    IR: 单层计算 {ir_compute_per_layer_val*1e6:.2f}µs × {num_layers}层 × {num_microbatches}微批 = {ir_compute_e2e_from_layer*1e3:.2f}ms")
        
        # Calculon 的纯计算时间
        calc_layer_compute = calc_block_fw_time + calc_block_bw_time  # 纯计算
        calc_compute_from_layer = calc_layer_compute * num_layers * num_microbatches
        print(f"    Calculon: 单层计算 {calc_layer_compute*1e6:.2f}µs × {num_layers}层 × {num_microbatches}微批 = {calc_compute_from_layer*1e3:.2f}ms")
        
        # 纯计算的差异
        compute_layer_diff = (ir_compute_e2e_from_layer - calc_compute_from_layer) / calc_compute_from_layer * 100 if calc_compute_from_layer > 0 else 0
        print(f"    纯计算差异: {compute_layer_diff:+.1f}% (与 FW/BW 一致)")
        
        # 检查差异是否与表格中的一致
        if abs(compute_layer_diff - diff_compute) > 5:
            print(f"    ⚠ 警告: 单层推算 ({compute_layer_diff:+.1f}%) 与分类汇总 ({diff_compute:+.1f}%) 不一致！")
    
    # 差异来源分析
    print(f"\n  差异来源分析:")
    if calc_recompute_total > 0 and ir_recompute_total == 0:
        print(f"    ⚠ IR 缺少重计算时间: {calc_recompute_total*1e3:.2f}ms ({calc_recompute_total/calc_e2e_total*100:.1f}%)")
    if abs(diff_compute) > 20:
        print(f"    ⚠ 计算时间差异较大: {diff_compute:+.1f}% (可能是 Calculon 的 fw/bw 时间包含了 FW+BW 的内存访问重叠)")
    if abs(diff_comm) > 20:
        print(f"    ⚠ 通信时间差异较大: {diff_comm:+.1f}%")
    if abs(diff_optim) > 50:
        print(f"    ⚠ 优化器时间差异较大: {diff_optim:+.1f}% (IR 优化器模型需要调整)")
    
    # ========== 单层时间比值（用于验证模型精度）==========
    # 注：这些比值反映 IR 与 Calculon 在单层级别的时间估计差异
    # 比值接近 1.0 表示模型精度高
    print("\n【单层时间比值】")
    
    calc_total_time = calculon_stats.get("total_time", 0)
    
    if isinstance(ir_total_time, (int, float)) and calc_total_time > 0:
        # 分项比值（基于单层）
        if calc_block_fw_time > 0 and ir_fw_per_layer > 0:
            fw_ratio = calc_block_fw_time / ir_fw_per_layer
            print(f"  前向:     Calculon/IR = {fw_ratio:.3f}x (单层)")
        
        if calc_block_bw_time > 0 and ir_bw_per_layer > 0:
            bw_ratio = calc_block_bw_time / ir_bw_per_layer
            print(f"  反向:     Calculon/IR = {bw_ratio:.3f}x (单层)")
        
        if calc_fw_tp_time + calc_agrad_tp_time > 0 and ir_comm_per_layer > 0:
            comm_ratio = (calc_fw_tp_time + calc_agrad_tp_time) / ir_comm_per_layer
            print(f"  通信:     Calculon/IR = {comm_ratio:.3f}x (单层)")
    
    # ========== 差异根因分析 ==========
    print("\n【差异根因分析】")
    
    # 计算强度分析
    calc_ai_fw = calc_block_fw_flops / calc_block_fw_mem if calc_block_fw_mem > 0 else 0
    calc_ai_bw = (calc_block_agrad_flops + calc_block_wgrad_flops) / (calc_block_agrad_mem + calc_block_wgrad_mem) if (calc_block_agrad_mem + calc_block_wgrad_mem) > 0 else 0
    
    print(f"  Calculon 计算强度 (Arithmetic Intensity):")
    print(f"    前向: {calc_ai_fw:.2f} FLOPs/Byte")
    print(f"    反向: {calc_ai_bw:.2f} FLOPs/Byte")
    
    # H100 的屋顶线模型
    h100_peak_tflops = 1000  # TFLOPs/s
    h100_mem_bw = 3072  # GB/s
    ridge_point = h100_peak_tflops * 1e12 / (h100_mem_bw * 1e9)  # ~326 FLOPs/Byte
    
    print(f"  H100 屋顶线拐点: {ridge_point:.1f} FLOPs/Byte")
    
    if calc_ai_fw < ridge_point:
        print(f"  ⚠ 前向是内存受限 (AI={calc_ai_fw:.1f} < {ridge_point:.1f})")
    else:
        print(f"  ✓ 前向是计算受限 (AI={calc_ai_fw:.1f} > {ridge_point:.1f})")
    
    if calc_ai_bw < ridge_point:
        print(f"  ⚠ 反向是内存受限 (AI={calc_ai_bw:.1f} < {ridge_point:.1f})")
    else:
        print(f"  ✓ 反向是计算受限 (AI={calc_ai_bw:.1f} > {ridge_point:.1f})")
    
    # IR 编译器精度分析
    print("\n  IR 编译器精度分析:")
    
    # 检查 IR 是否已有优化器时间
    ir_optim_time = ir_times_by_op.get("OptimizerStep", 0)
    
    insights = []
    
    # 计算单层时间误差
    fw_error = abs((ir_fw_per_layer - calc_block_fw_time) / calc_block_fw_time * 100) if calc_block_fw_time > 0 else 0
    bw_error = abs((ir_bw_per_layer - calc_block_bw_time) / calc_block_bw_time * 100) if calc_block_bw_time > 0 else 0
    
    if fw_error < 20:
        insights.append(f"✓ 前向时间误差 {fw_error:.1f}%，Roofline 模型有效")
    else:
        insights.append(f"⚠ 前向时间误差 {fw_error:.1f}%，可能需要调整效率因子")
    
    if bw_error < 20:
        insights.append(f"✓ 反向时间误差 {bw_error:.1f}%，Roofline 模型有效")
    else:
        insights.append(f"⚠ 反向时间误差 {bw_error:.1f}%，可能需要调整效率因子")
    
    # 检查优化器时间
    if ir_optim_time > 0:
        ir_optim_per_layer = ir_optim_time / num_layers
        optim_error = abs((ir_optim_per_layer - calc_block_optim_time) / calc_block_optim_time * 100) if calc_block_optim_time > 0 else 0
        if optim_error < 50:
            insights.append(f"✓ 优化器时间误差 {optim_error:.1f}%，建模有效")
        else:
            insights.append(f"⚠ 优化器时间误差 {optim_error:.1f}%，需要调整优化器模型")
    else:
        if calc_block_optim_time > 0:
            insights.append("⚠ IR 未输出优化器时间，需检查 OptimizerPass")
    
    # 通信时间
    comm_error = abs((ir_comm_per_layer - (calc_fw_tp_time + calc_agrad_tp_time)) / (calc_fw_tp_time + calc_agrad_tp_time) * 100) if (calc_fw_tp_time + calc_agrad_tp_time) > 0 else 0
    if comm_error < 30:
        insights.append(f"✓ 通信时间误差 {comm_error:.1f}%，带宽模型有效")
    else:
        insights.append(f"⚠ 通信时间误差 {comm_error:.1f}%，需要调整网络效率因子")
    
    # 重计算 - 已通过 OptimizerPass 建模
    ir_recompute_time = ir_times_by_op.get("RecomputeStep", 0)
    if ir_recompute_time > 0:
        recompute_error = abs((ir_recompute_time - calc_block_re_time * num_layers * num_microbatches) / (calc_block_re_time * num_layers * num_microbatches) * 100) if calc_block_re_time > 0 else 0
        if recompute_error < 30:
            insights.append(f"✓ 重计算时间误差 {recompute_error:.1f}%，梯度检查点建模有效")
        else:
            insights.append(f"⚠ 重计算时间误差 {recompute_error:.1f}%，需调整 checkpoint_ratio")
    elif calc_block_re_time > 0:
        insights.append("⚠ Calculon 有重计算时间，IR 需添加梯度检查点建模")
    
    for insight in insights:
        print(f"    {insight}")
    
    print("\n" + "=" * 100)
    
    return {
        "ir_times_by_op": ir_times_by_op,
        "calc_block_times": {
            "fw": calc_block_fw_time,
            "agrad": calc_block_agrad_time,
            "wgrad": calc_block_wgrad_time,
            "optim": calc_block_optim_time,
        }
    }


def run_single_layer_demo():
    """运行单层 Transformer 的编译演示（用于快速测试）。"""
    print_separator("单层 Transformer 编译演示")
    
    # 简化配置
    config = {
        "hidden": 12288,
        "feedforward": 49152,
        "num_heads": 96,
        "head_dim": 128,
        "num_layers": 1,  # 只用1层
        "vocab_size": 50257,
    }
    
    system_config = {
        "peak_tflops": 1000,
        "memory_bandwidth_gbps": 3072,
        "network_bandwidth_gbps": 450,
        "memory_capacity_gb": 80,
    }
    
    batch_size = 4
    seq_len = 2048
    tp = 8
    
    print(f"\n配置: batch_size={batch_size}, seq_len={seq_len}, tp={tp}")
    print(f"构建单层 Transformer GraphIR...")
    
    graph = build_gpt175b_graph(config, batch_size, seq_len, tp)
    stats = analyze_graph(graph)
    
    print(f"  节点数: {stats['total_nodes']}")
    print(f"  边数: {stats['total_edges']}")
    print(f"  操作类型分布: {stats['op_counts']}")
    
    # 创建编译器
    compiler = create_compiler_pipeline(
        system_config=system_config,
        tp=tp, pp=1, dp=1,
        batch_size=batch_size,
        seq_len=seq_len,
        hidden=config["hidden"],
        feedforward=config["feedforward"],
        num_layers=config["num_layers"],
        training=True
    )
    
    print(f"\n编译流水线: {compiler}")
    print("执行编译...")
    
    # 编译
    result = compiler.compile(graph)
    
    # 获取中间 IR (需要手动执行部分 passes)
    clear_expr_cache()
    
    # 手动执行以获取 Timeline
    current = graph
    timeline = None
    for p in compiler.passes:
        current = p.run(current)
        if isinstance(current, TimelineIR):
            timeline = current
    
    if timeline:
        print(f"\nTimeline 信息:")
        print(timeline.summary())
    
    print(f"\n最终结果:")
    print(result)
    
    # 缓存统计
    cache_stats = get_cache_stats()
    print(f"\n表达式缓存统计:")
    print(f"  命中: {cache_stats['hits']}, 未命中: {cache_stats['misses']}")
    print(f"  命中率: {cache_stats['hit_rate']:.1%}, 缓存大小: {cache_stats['size']}")


def run_full_model_simulation():
    """运行完整 GPT-175B 模型仿真。"""
    print_separator("GPT-175B 完整模型编译仿真")
    
    # 使用配置
    model_config = GPT_175B_CONFIG
    system_config = {
        "peak_tflops": H100_NVL8_CONFIG["peak_tflops"],
        "memory_bandwidth_gbps": H100_NVL8_CONFIG["memory_bandwidth_gbps"],
        "network_bandwidth_gbps": H100_NVL8_CONFIG["network_bandwidth_gbps"],
        "memory_capacity_gb": H100_NVL8_CONFIG["memory_capacity_gb"],
    }
    training_config = TRAINING_CONFIG
    
    tp = training_config["tensor_parallel"]
    pp = training_config["pipeline_parallel"]
    dp = training_config["data_parallel"]
    
    # 计算理论指标
    metrics = calculate_model_metrics(model_config, training_config)
    
    # 每 PP stage 的层数
    layers_per_stage = metrics["layers_per_stage"]
    
    print(f"\n【模型配置】")
    print(f"  模型: {model_config['name']}")
    print(f"  Hidden: {model_config['hidden']}")
    print(f"  FFN: {model_config['feedforward']}")
    print(f"  Layers: {model_config['num_layers']}")
    print(f"  Heads: {model_config['num_heads']}")
    print(f"  参数量: {format_flops(metrics['total_params']).replace('FLOPs', '')}")
    
    print(f"\n【并行配置】")
    print(f"  TP={tp}, PP={pp}, DP={dp}")
    print(f"  总GPU数: {tp * pp * dp}")
    print(f"  每 PP Stage 层数: {layers_per_stage}")
    
    print(f"\n【训练配置】")
    print(f"  Global Batch Size: {training_config['batch_size']}")
    print(f"  Micro Batch Size: {training_config['micro_batch_size']}")
    print(f"  Sequence Length: {model_config['seq_len']}")
    
    # 对于完整模型，我们只模拟单个 PP stage（避免构建过大的图）
    print(f"\n构建单个 PP Stage 的 GraphIR (layers={layers_per_stage})...")
    
    single_stage_config = dict(model_config)
    single_stage_config["num_layers"] = layers_per_stage
    
    micro_batch = training_config["micro_batch_size"]
    seq_len = model_config["seq_len"]
    
    graph = build_gpt175b_graph(
        single_stage_config,
        batch_size=micro_batch,
        seq_len=seq_len,
        tp=tp
    )
    
    stats = analyze_graph(graph)
    print(f"  节点数: {stats['total_nodes']}")
    print(f"  边数: {stats['total_edges']}")
    
    # 创建编译器
    print(f"\n创建编译器流水线...")
    gradient_checkpointing = training_config.get("gradient_checkpointing", False)
    num_microbatches_for_optim = metrics["num_microbatches"]
    compiler = create_compiler_pipeline(
        system_config=system_config,
        tp=tp, pp=1, dp=1,  # 单 stage，不考虑 PP
        batch_size=micro_batch,
        seq_len=seq_len,
        hidden=model_config["hidden"],
        feedforward=model_config["feedforward"],
        num_layers=layers_per_stage,
        training=True,
        gradient_checkpointing=gradient_checkpointing,
        num_microbatches=num_microbatches_for_optim,
    )
    
    print(f"编译流水线: {compiler}")
    
    # 清除缓存
    clear_expr_cache()
    
    # 手动执行 passes 以获取中间 IR
    print(f"\n执行编译 passes...")
    current = graph
    schedule_ir = None
    timeline_ir = None
    
    for i, p in enumerate(compiler.passes):
        print(f"  [{i+1}/{len(compiler.passes)}] 执行 {p.name}...")
        current = p.run(current)
        
        if isinstance(current, ScheduleIR):
            schedule_ir = current
        elif isinstance(current, TimelineIR):
            timeline_ir = current
    
    result = current  # 最终结果应该是 SimulationResult
    
    # 扩展到完整模型
    print(f"\n【扩展到完整模型】")
    
    # 单 stage 结果
    single_stage_time = result.e2e_time
    
    # PP 调度开销（1F1B）
    num_microbatches = metrics["num_microbatches"]
    bubble_ratio = (pp - 1) / num_microbatches if num_microbatches > 0 else 0
    
    # 完整 iteration 时间 = pp_stages * stage_time * (1 + bubble)
    # 简化：每个 stage 串行执行，有 bubble
    full_iteration_time = single_stage_time * pp * (1 + bubble_ratio)
    
    print(f"  单 Stage 时间: {format_time(single_stage_time)}")
    print(f"  微批数: {num_microbatches}")
    print(f"  Bubble 比例: {bubble_ratio:.1%}")
    print(f"  完整 Iteration 时间: {format_time(full_iteration_time)}")
    
    # 吞吐量计算
    tokens_per_iteration = training_config["batch_size"] * seq_len
    throughput = tokens_per_iteration / full_iteration_time
    
    print(f"  Tokens/Iteration: {tokens_per_iteration:,}")
    print(f"  吞吐量: {throughput:,.0f} tokens/s")
    
    # 计算 MFU
    total_gpus = tp * pp * dp
    peak_flops_per_gpu = system_config["peak_tflops"] * 1e12
    total_peak_flops = peak_flops_per_gpu * total_gpus
    
    # 理论最小时间 (假设 100% 利用率)
    theoretical_min_time = metrics["total_flops_per_iteration"] / total_peak_flops
    
    # 实际 MFU = 理论最小时间 / 实际时间
    # 或者: MFU = 实际 FLOPs/s / 峰值 FLOPs/s
    # 注意: IR 调度时间可能因符号替换问题不准确，所以同时显示两种估计
    
    # 基于 IR 调度的估计
    ir_achieved_flops = metrics["total_flops_per_iteration"] / full_iteration_time
    ir_mfu = theoretical_min_time / full_iteration_time
    
    # 基于经验的 MFU 估计 (考虑通信、bubble 等开销)
    # 典型 LLM 训练 MFU 在 30-50% 之间
    comm_overhead = 0.15  # 通信开销约 15%
    bubble_overhead = bubble_ratio
    efficiency_loss = 0.1  # 其他效率损失约 10%
    estimated_mfu = (1 - comm_overhead) * (1 - bubble_overhead) * (1 - efficiency_loss)
    
    # 基于估计 MFU 的实际训练时间
    estimated_iteration_time = theoretical_min_time / estimated_mfu
    
    # 打印完整结果
    print_separator("仿真结果摘要")
    
    print(f"\n{'='*60}")
    print(f"  模型: GPT-175B ({metrics['total_params']/1e9:.1f}B 参数)")
    print(f"  系统: H100-80G-NVL8 × {total_gpus} GPUs")
    print(f"{'='*60}")
    
    # 并行配置
    print(f"\n【并行配置】")
    print(f"  Tensor Parallel (TP):   {tp}")
    print(f"  Pipeline Parallel (PP): {pp}")
    print(f"  Data Parallel (DP):     {dp}")
    print(f"  总GPU数:                {total_gpus}")
    
    # 训练配置
    print(f"\n【训练配置】")
    print(f"  Global Batch Size:      {training_config['batch_size']}")
    print(f"  Micro Batch Size:       {micro_batch}")
    print(f"  Sequence Length:        {seq_len}")
    print(f"  微批数 (per DP group):  {num_microbatches}")
    
    # 内存分析 (使用理论计算)
    print(f"\n【内存分析 (每 GPU)】")
    print(f"  模型参数:               {metrics['model_memory_gb']:.2f} GB")
    print(f"  梯度:                   {metrics['gradient_memory_gb']:.2f} GB")
    print(f"  优化器状态:             {metrics['optimizer_memory_gb']:.2f} GB")
    print(f"  激活 (峰值):            {metrics['activation_memory_gb']:.2f} GB")
    print(f"  ------------------------------------")
    print(f"  总计:                   {metrics['total_memory_per_gpu_gb']:.2f} GB")
    print(f"  GPU 内存容量:           {system_config['memory_capacity_gb']:.0f} GB")
    mem_ok = metrics['total_memory_per_gpu_gb'] <= system_config['memory_capacity_gb']
    print(f"  内存是否足够:           {'✓ 是' if mem_ok else '✗ 否 (OOM)'}")
    
    # 性能分析
    print(f"\n【性能分析】")
    print(f"  每 Iteration FLOPs:     {format_flops(metrics['total_flops_per_iteration'])}")
    print(f"  峰值 FLOPs (集群):      {format_flops(total_peak_flops)}")
    print(f"  理论最小时间:           {format_time(theoretical_min_time)}")
    print(f"  ------------------------------------")
    print(f"  IR 调度分析 (单 Stage):")
    print(f"    单 Stage 时间:        {format_time(single_stage_time)}")
    print(f"    Pipeline Bubble:      {bubble_ratio:.1%}")
    print(f"    完整 Iteration:       {format_time(full_iteration_time)}")
    print(f"  ------------------------------------")
    print(f"  经验估计 (考虑实际开销):")
    print(f"    通信开销:             ~15%")
    print(f"    Pipeline Bubble:      {bubble_ratio:.1%}")
    print(f"    其他效率损失:         ~10%")
    print(f"    预估 MFU:             {estimated_mfu:.1%}")
    print(f"    预估 Iteration 时间:  {format_time(estimated_iteration_time)}")
    print(f"  ------------------------------------")
    
    # 使用经验估计计算吞吐量
    estimated_throughput = tokens_per_iteration / estimated_iteration_time
    print(f"  预估 Token 吞吐量:      {estimated_throughput:,.0f} tokens/s")
    print(f"  Tokens/Iteration:       {tokens_per_iteration:,}")
    
    # Timeline 分析
    if timeline_ir:
        print(f"\n【Timeline 分析 (单 Stage)】")
        print(f"  事件总数:               {len(timeline_ir.events)}")
        
        for dev in range(min(timeline_ir.num_devices, 2)):  # 最多显示2个设备
            compute = timeline_ir.compute_time(dev)
            comm = timeline_ir.comm_time(dev)
            overlap = timeline_ir.overlap_ratio(dev)
            
            if isinstance(compute, (int, float)) and isinstance(comm, (int, float)):
                print(f"  Device {dev}:")
                print(f"    - 计算时间:           {format_time(compute)}")
                print(f"    - 通信时间:           {format_time(comm)}")
                print(f"    - 计算/通信比:        {compute/comm:.2f}x" if comm > 0 else "    - 计算/通信比:        N/A")
                print(f"    - 重叠率:             {overlap:.1%}")
    
    print(f"\n{'='*60}\n")
    
    # 导出 Chrome Trace
    if timeline_ir:
        output_dir = Path(__file__).parent
        trace_path = output_dir / "gpt175b_timeline.json"
        export_chrome_trace(timeline_ir, str(trace_path))
        
        # 保存完整仿真结果到 JSON
        result_path = output_dir / "gpt175b_result.json"
        full_result = {
            "model": {
                "name": "GPT-175B",
                "params": metrics["total_params"],
                "layers": model_config["num_layers"],
                "hidden": model_config["hidden"],
                "feedforward": model_config["feedforward"],
                "num_heads": model_config["num_heads"],
            },
            "parallel": {
                "tp": tp,
                "pp": pp,
                "dp": dp,
                "total_gpus": tp * pp * dp,
            },
            "training": {
                "batch_size": training_config["batch_size"],
                "micro_batch_size": micro_batch,
                "seq_len": seq_len,
                "num_microbatches": num_microbatches,
            },
            "memory_per_gpu_gb": {
                "model": metrics["model_memory_gb"],
                "gradient": metrics["gradient_memory_gb"],
                "optimizer": metrics["optimizer_memory_gb"],
                "activation": metrics["activation_memory_gb"],
                "total": metrics["total_memory_per_gpu_gb"],
            },
            "performance": {
                "flops_per_iteration": metrics["total_flops_per_iteration"],
                "peak_cluster_flops": total_peak_flops,
                "theoretical_min_time_s": theoretical_min_time,
                "ir_schedule": {
                    "single_stage_time_ms": single_stage_time * 1e3,
                    "iteration_time_ms": full_iteration_time * 1e3,
                    "bubble_ratio": bubble_ratio,
                },
                "estimated": {
                    "mfu": estimated_mfu,
                    "iteration_time_s": estimated_iteration_time,
                    "throughput_tokens_per_sec": estimated_throughput,
                },
            },
            "ir_result": result.to_dict(),
        }
        with open(result_path, 'w') as f:
            json.dump(full_result, f, indent=2, default=str)
        print(f"完整仿真结果已保存到: {result_path}")
    
    return result, timeline_ir, metrics


def run_calculon_comparison(metrics: Dict[str, Any]):
    """运行 Calculon 对比分析。
    
    使用完全相同的模型配置和并行参数，同时运行 IR 编译器和 Calculon，
    确保对比的公平性。
    """
    print("\n" + "=" * 70)
    print("  Part 3: 与 Calculon 结果对比")
    print("=" * 70)
    
    # 系统配置文件路径
    data_dir = Path(__file__).parent.parent / "data"
    system_file = data_dir / "systems" / "h100_80g_nvl8.json"
    
    if not system_file.exists():
        print(f"\n⚠ 系统配置文件不存在: {system_file}")
        print("  跳过 Calculon 对比...")
        return None
    
    print(f"\n使用系统配置: {system_file}")
    
    # 使用 13B 模型配置进行对比（可以在单节点 8 GPU 运行）
    # 类似 Llama-13B 的配置
    model_config = {
        "name": "Llama-13B",
        "hidden": 5120,
        "feedforward": 13824,  # 约 2.7x hidden
        "seq_len": 2048,
        "num_heads": 40,
        "head_dim": 128,
        "num_layers": 40,
        "vocab_size": 32000,
    }
    
    training_config = {
        "batch_size": 8,                # 单节点较小的 batch size
        "micro_batch_size": 1,
        "tensor_parallel": 8,           # TP=8 (单节点内)
        "pipeline_parallel": 1,         # PP=1 (无流水线)
        "data_parallel": 1,             # DP=1 (无数据并行)
        "dtype": "float16",
        "optimizer": "adam",
        "gradient_checkpointing": True,
    }
    
    # 使用 Calculon 配置文件 (IR 和 Calculon 共享同一配置)
    system_config = system_file  # 直接使用配置文件路径
    
    # 加载配置以显示信息
    from blueprinting.ir import SystemConfig
    sys_cfg = SystemConfig(system_file)
    
    tp = training_config["tensor_parallel"]
    pp = training_config["pipeline_parallel"]
    dp = training_config["data_parallel"]
    micro_batch = training_config["micro_batch_size"]
    seq_len = model_config["seq_len"]
    
    print(f"\n【统一对比配置】")
    print(f"  模型: {model_config['name']} (H={model_config['hidden']}, L={model_config['num_layers']})")
    print(f"  并行: TP={tp}, PP={pp}, DP={dp} ({tp*pp*dp} GPUs)")
    print(f"  Batch Size: {training_config['batch_size']}")
    print(f"  Micro Batch Size: {micro_batch}")
    print(f"  Sequence Length: {seq_len}")
    
    print(f"\n【系统配置 (与 Calculon 共享)】")
    print(f"  配置文件: {system_file.name}")
    print(f"  Peak TFLOPs:     {sys_cfg.peak_tflops}")
    print(f"  Memory BW:       {sys_cfg.memory_bandwidth/1e9:.0f} GB/s")
    print(f"  Network BW:      {sys_cfg.network_bandwidth/1e9:.0f} GB/s")
    print(f"  Processing Mode: {sys_cfg.processing_mode}")
    print(f"  动态效率: ✓ (从配置文件加载)")
    
    # 显示效率因子范围
    print(f"  Compute Eff:     {sys_cfg.get_compute_efficiency(1e12):.0%} (large ops) / {sys_cfg.get_compute_efficiency(1e9):.0%} (small ops)")
    print(f"  Memory Eff:      {sys_cfg.get_memory_efficiency(100e6):.0%} (large) / {sys_cfg.get_memory_efficiency(1e6):.0%} (small)")
    
    # ========== 1. 运行 IR 编译器 ==========
    print(f"\n【1. IR 编译器仿真】")
    
    # 构建 GraphIR
    graph = build_gpt175b_graph(
        model_config,
        batch_size=micro_batch,
        seq_len=seq_len,
        tp=tp
    )
    
    stats = analyze_graph(graph)
    print(f"  GraphIR: {stats['total_nodes']} 节点, {stats['total_edges']} 边")
    
    # 创建编译器
    gradient_checkpointing = training_config.get("gradient_checkpointing", False)
    num_microbatches = training_config["batch_size"] // (micro_batch * dp)
    compiler = create_compiler_pipeline(
        system_config=system_config,
        tp=tp, pp=pp, dp=dp,
        batch_size=micro_batch,
        seq_len=seq_len,
        hidden=model_config["hidden"],
        feedforward=model_config["feedforward"],
        num_layers=model_config["num_layers"],
        training=True,
        gradient_checkpointing=gradient_checkpointing,
        num_microbatches=num_microbatches,
    )
    
    # 清除缓存
    clear_expr_cache()
    
    # 执行编译
    current = graph
    timeline_ir = None
    schedule_ir = None
    for p in compiler.passes:
        current = p.run(current)
        if isinstance(current, ScheduleIR):
            schedule_ir = current
        elif isinstance(current, TimelineIR):
            timeline_ir = current
    
    ir_result = current  # SimulationResult
    
    # 获取 IR 编译结果
    ir_e2e_time = ir_result.e2e_time
    
    # 计算完整 iteration 时间（考虑多个 micro batch）
    num_microbatches = training_config["batch_size"] // (micro_batch * dp)
    ir_iteration_time = ir_e2e_time * num_microbatches
    
    print(f"  单 Micro Batch 时间: {ir_e2e_time*1e3:.2f} ms")
    print(f"  微批数: {num_microbatches}")
    print(f"  完整 Iteration 时间: {ir_iteration_time*1e3:.2f} ms")
    
    if timeline_ir:
        ir_compute_time = timeline_ir.compute_time(0)
        ir_comm_time = timeline_ir.comm_time(0)
        if isinstance(ir_compute_time, (int, float)) and isinstance(ir_comm_time, (int, float)):
            print(f"  计算时间: {ir_compute_time*1e3:.2f} ms")
            print(f"  通信时间: {ir_comm_time*1e3:.2f} ms")
    
    # ========== 2. 运行 Calculon ==========
    print(f"\n【2. Calculon 仿真】")
    
    calculon_result = run_calculon_simulation(
        model_config=model_config,
        training_config=training_config,
        system_file=str(system_file),
    )
    
    if calculon_result.get("success", False):
        calc_stats = calculon_result["stats"]
        calc_total_time = calc_stats.get("total_time", 0)
        calc_fw_time = calc_stats.get("fw_time", 0)
        calc_bw_time = calc_stats.get("bw_time", 0)
        calc_mfu = calc_stats.get("total_efficiency", 0) * 100
        
        print(f"  总时间: {calc_total_time*1e3:.2f} ms")
        print(f"  前向时间: {calc_fw_time*1e3:.2f} ms")
        print(f"  反向时间: {calc_bw_time*1e3:.2f} ms")
        print(f"  MFU: {calc_mfu:.1f}%")
    else:
        print(f"  ⚠ Calculon 仿真失败: {calculon_result.get('error', 'Unknown error')}")
        return None
    
    # ========== 3. 对比结果 ==========
    # 计算理论指标（用于对比）
    ir_metrics = calculate_model_metrics(model_config, training_config)
    
    # 更新 IR 指标中的实际运行时间
    ir_metrics["ir_iteration_time"] = ir_iteration_time
    ir_metrics["ir_micro_batch_time"] = ir_e2e_time
    if timeline_ir:
        ir_metrics["ir_compute_time"] = timeline_ir.compute_time(0)
        ir_metrics["ir_comm_time"] = timeline_ir.comm_time(0)
    
    comparison = compare_with_calculon(
        ir_metrics=ir_metrics,
        calculon_result=calculon_result,
        training_config=training_config,
    )
    
    # ========== 4. 细粒度时间分析 ==========
    if timeline_ir and calculon_result.get("success", False):
        calibration = fine_grained_time_analysis(
            timeline_ir=timeline_ir,
            calculon_stats=calculon_result["stats"],
            model_config=model_config,
            training_config=training_config,
        )
        if comparison:
            comparison["calibration"] = calibration
    
    # 保存结果
    output_dir = Path(__file__).parent
    
    if calculon_result.get("success", False):
        calc_result_path = output_dir / "comparison_calculon_result.json"
        with open(calc_result_path, 'w') as f:
            json.dump(calculon_result["stats"], f, indent=2, default=str)
        print(f"\nCalculon 结果已保存到: {calc_result_path}")
    
    # 保存对比摘要
    comparison_summary = {
        "config": {
            "model": model_config["name"],
            "hidden": model_config["hidden"],
            "layers": model_config["num_layers"],
            "tp": tp, "pp": pp, "dp": dp,
            "batch_size": training_config["batch_size"],
            "micro_batch_size": micro_batch,
        },
        "ir_compiler": {
            "iteration_time_ms": ir_iteration_time * 1e3,
            "micro_batch_time_ms": ir_e2e_time * 1e3,
        },
        "calculon": {
            "total_time_ms": calc_total_time * 1e3,
            "fw_time_ms": calc_fw_time * 1e3,
            "bw_time_ms": calc_bw_time * 1e3,
            "mfu_pct": calc_mfu,
        },
        "diff_pct": {
            "iteration_time": (ir_iteration_time - calc_total_time) / calc_total_time * 100 if calc_total_time > 0 else 0,
        }
    }
    
    summary_path = output_dir / "comparison_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(comparison_summary, f, indent=2, default=str)
    print(f"对比摘要已保存到: {summary_path}")
    
    return comparison


def main():
    """主函数。"""
    print("=" * 70)
    print("  GPT-175B 模型训练仿真")
    print("  使用 blueprinting IR 编译器 + Calculon 对比")
    print("=" * 70)
    
    # 1. 运行单层演示（快速测试）
    print("\n" + "=" * 70)
    print("  Part 1: 单层 Transformer 编译演示")
    print("=" * 70)
    run_single_layer_demo()
    
    # 2. 运行完整模型仿真
    print("\n" + "=" * 70)
    print("  Part 2: GPT-175B 完整模型仿真")
    print("=" * 70)
    result, timeline, metrics = run_full_model_simulation()
    
    # 3. 与 Calculon 对比
    comparison = run_calculon_comparison(metrics)
    
    print("\n" + "=" * 70)
    print("  仿真完成!")
    print("=" * 70)
    
    return metrics, comparison


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""GPT-175B 模型训练仿真示例

本示例展示如何使用 blueprinting IR 编译器对 GPT-175B 模型进行
训练仿真分析，并与 Calculon 的结果进行对比。

编译流程：
    GraphIR → WorkloadPass → ParallelPass → SchedulePass → TimelinePass → EvaluatePass

使用方法:
    python examples/gpt175b_compile.py
    
配置文件：
    模型: data/models/gpt3-175B.json
    系统: data/systems/h100_80g_nvl8.json
    执行: data/examples/3072_t4_p64_d12_mbs4_full.json
"""

import json
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional

# 添加 src 到 path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from blueprinting.ir import (
    GraphIR, IRBuilder, Compiler,
    WorkloadPass, ParallelPass, SchedulePass,
    TimelinePass, OverlapAnalysisPass, EvaluatePass,
    OptimizerPass, OptimizerConfig,
)
from blueprinting import Model, Execution
from calculon import System
from calculon.llm import Llm
import hyperparameter as hp
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# 配置加载
# ============================================================================

@dataclass
class Config:
    """统一配置管理"""
    model_path: Path
    system_path: Path
    execution_path: Path
    
    model: Dict[str, Any] = None
    system: Dict[str, Any] = None  
    execution: Dict[str, Any] = None
    
    def __post_init__(self):
        self.model = self._load_json(self.model_path)
        self.system = self._load_json(self.system_path)
        self.execution = self._load_json(self.execution_path)
    
    @staticmethod
    def _load_json(path: Path) -> Dict[str, Any]:
        with open(path) as f:
            return json.load(f)
    
    @classmethod
    def from_names(cls, model: str, system: str, execution: str) -> "Config":
        """从名称创建配置（自动查找 data/ 目录）"""
        data_dir = ROOT / "data"
        return cls(
            model_path=data_dir / "models" / f"{model}.json",
            system_path=data_dir / "systems" / f"{system}.json",
            execution_path=data_dir / "examples" / f"{execution}.json",
        )
    
    # 便捷属性
    @property
    def hidden(self) -> int:
        return self.model["hidden"]
    
    @property
    def feedforward(self) -> int:
        return self.model["feedforward"]
    
    @property
    def seq_len(self) -> int:
        return self.model.get("seq_size", 2048)
    
    @property
    def num_heads(self) -> int:
        return self.model["attn_heads"]
    
    @property
    def head_dim(self) -> int:
        return self.model["attn_size"]
    
    @property
    def num_layers(self) -> int:
        return self.model["num_blocks"]
    
    @property
    def tp(self) -> int:
        return self.execution["tensor_par"]
    
    @property
    def pp(self) -> int:
        return self.execution["pipeline_par"]
    
    @property
    def dp(self) -> int:
        return self.execution["data_par"]
    
    @property
    def batch_size(self) -> int:
        return self.execution["batch_size"]
    
    @property
    def micro_batch_size(self) -> int:
        return self.execution["microbatch_size"]
    
    @property
    def num_gpus(self) -> int:
        return self.tp * self.pp * self.dp
    
    @property
    def gradient_checkpointing(self) -> bool:
        return self.execution.get("activation_recompute", "none") != "none"


# ============================================================================
# 输出格式化
# ============================================================================

def print_header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

def print_section(title: str):
    print(f"\n[{title}]")

def print_comparison(name: str, ir_val, calc_val):
    """打印对比行 - ir_val 和 calc_val 都是格式化后的字符串"""
    ir_str = ir_val if ir_val else "-"
    calc_str = calc_val if calc_val else "-"
    print(f"  {name:12} IR: {ir_str:>12}  Calculon: {calc_str:>12}")


def format_diff(ir_val: float, calc_val: float) -> str:
    """格式化相对 Calculon 的误差百分比"""
    if not calc_val:
        return ""
    diff = pct_diff(ir_val, calc_val)
    return f"  diff:{diff:+.1f}%"


def print_comparison_with_diff(name: str, ir_val, calc_val, ir_num: float, calc_num: float):
    """打印对比行并追加相对误差"""
    ir_str = ir_val if ir_val else "-"
    calc_str = calc_val if calc_val else "-"
    diff_str = format_diff(ir_num, calc_num)
    print(f"  {name:12} IR: {ir_str:>12}  Calculon: {calc_str:>12}{diff_str}")


def format_bytes(n: float) -> str:
    """格式化字节数"""
    if n >= 1e12: return f"{n/1e12:.2f} TB"
    if n >= 1e9: return f"{n/1e9:.2f} GB"
    if n >= 1e6: return f"{n/1e6:.2f} MB"
    return f"{n:.0f} B"


def format_time(s: float) -> str:
    """格式化时间"""
    if s >= 1: return f"{s:.2f} s"
    if s >= 1e-3: return f"{s*1e3:.2f} ms"
    return f"{s*1e6:.2f} µs"


def format_num(n: float) -> str:
    """格式化数字"""
    if n >= 1e12: return f"{n/1e12:.2f}T"
    if n >= 1e9: return f"{n/1e9:.2f}B"
    if n >= 1e6: return f"{n/1e6:.2f}M"
    return f"{n:.0f}"


def pct_diff(a: float, b: float) -> float:
    """计算百分比差异"""
    if b == 0: return 0
    return (a - b) / b * 100


def symbol(diff: float, threshold: float = 10) -> str:
    """根据差异返回符号"""
    if abs(diff) < threshold: return "✓"
    if abs(diff) < threshold * 2: return "⚠"
    return "✗"


def normalize_ir_metrics(ir_result) -> Dict[str, Any]:
    """Normalize IR metrics to comparison schema."""
    if not ir_result:
        return {}
    
    config = getattr(ir_result, "config", {}) or {}
    metrics = config.get("comparison_metrics")
    if metrics:
        return metrics
    
    ir_mb = getattr(ir_result, "memory_breakdown", None)
    ir_tb = getattr(ir_result, "time_breakdown", None)
    
    weights_fp32 = ir_mb.weights if ir_mb else 0
    activations = ir_mb.activations if ir_mb else 0
    gradients = ir_mb.gradients if ir_mb else 0
    optimizer_states = ir_mb.optimizer_states if ir_mb else 0
    total_memory = ir_mb.total if ir_mb else 0
    
    return {
        "schema": "comparison_v1",
        "units": {"memory": "bytes", "time": "seconds"},
        "basis": {
            "weights": "fp32_master",
            "activations": "unknown",
            "gradients": "fp16",
            "optimizer_states": "fp32",
        },
        "per_gpu": {
            "memory": {
                "weights_fp16_bytes": weights_fp32 / 2 if weights_fp32 else 0,
                "weights_fp32_bytes": weights_fp32,
                "activations_bytes": activations,
                "activations_block_bytes": 0,
                "activations_peak_bytes": getattr(ir_result, "peak_memory", 0),
                "gradients_bytes": gradients,
                "optimizer_bytes": optimizer_states,
                "total_bytes": total_memory,
            },
            "time": {
                "iteration_time": getattr(ir_result, "e2e_time", 0) or 0,
                "forward_time": ir_tb.forward if ir_tb else 0,
                "backward_time": ir_tb.backward if ir_tb else 0,
                "optimizer_time": ir_tb.optimizer if ir_tb else 0,
                "communication_time": ir_tb.communication if ir_tb else 0,
                "bubble_time": ir_tb.bubble if ir_tb else 0,
            },
        },
    }


def normalize_calculon_stats(calc_stats: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize Calculon stats to comparison schema."""
    weight = calc_stats.get("weight_space", 0)
    act = calc_stats.get("act_space", 0)
    optim = calc_stats.get("optim_space", 0) or 0
    total_memory = weight + act + optim
    
    block_weight = calc_stats.get("block_weight_space", 0)
    block_act = calc_stats.get("block_act_working_space", 0)
    block_optim = calc_stats.get("block_optimizer_space", 0)
    
    block_fw = calc_stats.get("block_fw_time", 0)
    block_agrad = calc_stats.get("block_agrad_time", 0)
    block_wgrad = calc_stats.get("block_wgrad_time", 0)
    block_compute = block_fw + block_agrad + block_wgrad
    tp_fw = calc_stats.get("baseblock_fw_tp_time", 0)
    tp_bw = calc_stats.get("baseblock_agrad_tp_time", 0)
    block_comm = tp_fw + tp_bw
    block_total = block_compute + block_comm
    
    return {
        "schema": "comparison_v1",
        "units": {"memory": "bytes", "time": "seconds"},
        "basis": {
            "weights": "fp16",
            "activations": "block_working_set",
            "gradients": "fp16",
            "optimizer_states": "fp32",
        },
        "per_gpu": {
            "memory": {
                "weights_fp16_bytes": weight,
                "weights_fp32_bytes": weight * 2 if weight else 0,
                "activations_bytes": act,
                "activations_block_bytes": block_act,
                "activations_peak_bytes": 0,
                "gradients_bytes": 0,
                "optimizer_bytes": optim,
                "total_bytes": total_memory,
            },
            "time": {
                "iteration_time": calc_stats.get("total_time", 0),
                "forward_time": 0,
                "backward_time": 0,
                "optimizer_time": 0,
                "communication_time": 0,
                "bubble_time": 0,
            },
        },
        "block": {
            "memory": {
                "weights_bytes": block_weight,
                "activations_bytes": block_act,
                "optimizer_bytes": block_optim,
            },
            "time": {
                "forward_time": block_fw,
                "agrad_time": block_agrad,
                "wgrad_time": block_wgrad,
                "compute_time": block_compute,
                "tp_comm_fw": tp_fw,
                "tp_comm_bw": tp_bw,
                "comm_time": block_comm,
                "total_time": block_total,
            },
        },
    }


# ============================================================================
# 图构建
# ============================================================================

def build_transformer_graph(cfg: Config) -> GraphIR:
    """构建 Transformer 模型的 GraphIR
    
    只定义逻辑结构，不做 TP 相关判断。
    所有并行变换由 ParallelPass 处理。
    """
    builder = IRBuilder()
    
    hidden = cfg.hidden
    feedforward = cfg.feedforward
    num_heads = cfg.num_heads
    head_dim = cfg.head_dim
    num_layers = cfg.num_layers
    batch_seq = cfg.micro_batch_size * cfg.seq_len
    
    # 元数据
    builder.set_metadata("model_name", cfg.model_path.stem)
    builder.set_metadata("num_layers", num_layers)
    builder.set_metadata("hidden", hidden)
    builder.set_metadata("feedforward", feedforward)
    builder.set_metadata("batch_size", cfg.micro_batch_size)
    builder.set_metadata("seq_len", cfg.seq_len)
    builder.set_metadata("tp", cfg.tp)
    builder.set_metadata("optimizer_sharding", cfg.execution.get("optimizer_sharding", False))
    builder.set_metadata("zero", cfg.execution.get("zero", 0))
    builder.set_metadata("activation_recompute", cfg.execution.get("activation_recompute", "none"))
    
    # Input
    x = builder.add_input("input_ids", shape=[cfg.micro_batch_size, cfg.seq_len, hidden])
    prev = x
    
    # Transformer layers
    for i in range(num_layers):
        p = f"layer{i}_"
        
        # Attention block
        attn_norm = builder.add_rmsnorm(f"{p}attn_norm", [prev], hidden)
        builder._nodes[attn_norm].attrs["batch_seq"] = batch_seq
        
        # QKV projections (Column Parallel)
        q = builder.add_linear(f"{p}q_proj", [attn_norm], hidden, hidden, shard="tp_col")
        k = builder.add_linear(f"{p}k_proj", [attn_norm], hidden, hidden, shard="tp_col")
        v = builder.add_linear(f"{p}v_proj", [attn_norm], hidden, hidden, shard="tp_col")
        for node in [q, k, v]:
            builder._nodes[node].attrs["batch_seq"] = batch_seq
        
        # Attention
        attn = builder.add_attention(f"{p}attn", [q, k, v], num_heads, head_dim, cfg.seq_len)
        builder._nodes[attn].attrs["batch_size"] = cfg.micro_batch_size
        builder._nodes[attn].attrs["shard"] = "tp_col"
        
        # Output projection (Row Parallel)
        attn_out = builder.add_linear(f"{p}attn_out", [attn], hidden, hidden, shard="tp_row")
        builder._nodes[attn_out].attrs["batch_seq"] = batch_seq
        
        # Residual
        attn_add = builder.add_elementwise(f"{p}attn_add", "Add", [prev, attn_out])
        builder._nodes[attn_add].attrs["num_elements"] = batch_seq * hidden
        
        # FFN block (标准 2 层 FFN，与 Calculon 对齐)
        ffn_norm = builder.add_rmsnorm(f"{p}ffn_norm", [attn_add], hidden)
        builder._nodes[ffn_norm].attrs["batch_seq"] = batch_seq
        
        # FC1: hidden -> feedforward (Column Parallel)
        fc1 = builder.add_linear(f"{p}fc1", [ffn_norm], hidden, feedforward, shard="tp_col")
        builder._nodes[fc1].attrs["batch_seq"] = batch_seq
        
        # GELU activation
        gelu = builder.add_elementwise(f"{p}gelu", "SiLU", [fc1])  # 用 SiLU 近似 GELU
        builder._nodes[gelu].attrs["num_elements"] = batch_seq * feedforward
        builder._nodes[gelu].attrs["shard"] = "tp_col"
        
        # FC2: feedforward -> hidden (Row Parallel)
        fc2 = builder.add_linear(f"{p}fc2", [gelu], feedforward, hidden, shard="tp_row")
        builder._nodes[fc2].attrs["batch_seq"] = batch_seq
        
        # Residual
        ffn_add = builder.add_elementwise(f"{p}ffn_add", "Add", [attn_add, fc2])
        builder._nodes[ffn_add].attrs["num_elements"] = batch_seq * hidden
        
        prev = ffn_add
    
    # Final norm + LM head
    final_norm = builder.add_rmsnorm("final_norm", [prev], hidden)
    builder._nodes[final_norm].attrs["batch_seq"] = batch_seq
    
    # 注：LM Head 通常与 embedding 共享权重
    builder.mark_output(final_norm)
    
    return builder.build()


# ============================================================================
# 编译流水线
# ============================================================================

def create_compiler(cfg: Config) -> Compiler:
    """创建编译器流水线"""
    compiler = Compiler()
    
    # 计算 microbatch 数量
    num_microbatches = cfg.batch_size // (cfg.micro_batch_size * cfg.dp)
    
    compiler.add_pass(WorkloadPass(dtype_bytes=2))
    compiler.add_pass(ParallelPass(
        tp=cfg.tp,
        pp=cfg.pp,
        dp=cfg.dp,
        tp_comm_type=cfg.execution.get("tensor_par_comm_type", "ar"),
        sequence_parallel=cfg.execution.get("sequence_par", False),
    ))
    compiler.add_pass(SchedulePass(
        system_config=cfg.system_path,
        training=True,
    ))
    compiler.add_pass(OptimizerPass(
        OptimizerConfig(
            optimizer_type="adam",
            master_weights=True,
            gradient_checkpointing=cfg.gradient_checkpointing,
            recompute_mode="full" if cfg.gradient_checkpointing else "none",
            checkpoint_ratio=1.0 if cfg.gradient_checkpointing else 0.0,
            num_microbatches=num_microbatches,
        ),
        system_config={
            "memory_bandwidth_gbps": 3072,
            "peak_tflops": 1000,
        }
    ))
    compiler.add_pass(TimelinePass())
    compiler.add_pass(OverlapAnalysisPass())
    compiler.add_pass(EvaluatePass(
        subs={"batch_seq": cfg.micro_batch_size * cfg.seq_len},
        training=True,
    ))
    
    return compiler


# ============================================================================
# Calculon 对比
# ============================================================================

def run_calculon(cfg: Config) -> Dict[str, Any]:
    """运行 Calculon 仿真"""
    # 使用 hp.scope 传递配置参数
    with hp.scope(app=cfg.model, exe=cfg.execution) as ps:
        app = Model.from_cfg(ps.app)
        exe = Execution(ps.exe)
        syst = System(cfg.system)
        
        model = Llm(app, logger)
        model.compile(syst, exe)
        model.run(syst)
        
        return model.get_stats_json(False)


def compare_results(cfg: Config, ir_result, calc_stats: Dict[str, Any]):
    """对比 IR 编译器与 Calculon 结果
    
    只从结果中读取数据并对齐单位，不做任何计算和估算。
    """
    
    ir_metrics = normalize_ir_metrics(ir_result)
    calc_metrics = normalize_calculon_stats(calc_stats)
    
    ir_mem = ir_metrics.get("per_gpu", {}).get("memory", {})
    ir_time = ir_metrics.get("per_gpu", {}).get("time", {})
    ir_block = ir_metrics.get("block", {})
    ir_basis = ir_metrics.get("basis", {})
    
    calc_mem = calc_metrics.get("per_gpu", {}).get("memory", {})
    calc_time = calc_metrics.get("per_gpu", {}).get("time", {})
    calc_block = calc_metrics.get("block", {})
    calc_basis = calc_metrics.get("basis", {})
    
    # Raw values for display (Calculon optimizer can be None with ZeRO)
    calc_optim_raw = calc_stats.get("optim_space", None)
    
    # ========== 输出对比 ==========
    print_header("IR 编译器 vs Calculon 对比")
    
    print(f"\n配置: {cfg.model_path.stem}, TP={cfg.tp}, PP={cfg.pp}, DP={cfg.dp}")
    print(f"GPUs: {cfg.num_gpus}, Batch: {cfg.batch_size}, MicroBatch: {cfg.micro_batch_size}")
    
    # 总内存
    print_section("总内存 (per GPU)")
    ir_weight_basis = ir_basis.get("weights", "fp32")
    calc_weight_basis = calc_basis.get("weights", "fp32")
    ir_weight_bytes = ir_mem.get("weights_fp16_bytes", 0) if ir_weight_basis == "fp16" else ir_mem.get("weights_fp32_bytes", 0)
    calc_weight_bytes = calc_mem.get("weights_fp16_bytes", 0) if calc_weight_basis == "fp16" else calc_mem.get("weights_fp32_bytes", 0)
    print_comparison_with_diff(
        "权重",
        format_bytes(ir_weight_bytes),
        format_bytes(calc_weight_bytes),
        ir_weight_bytes,
        calc_weight_bytes,
    )
    print_comparison_with_diff(
        "激活",
        format_bytes(ir_mem.get("activations_bytes", 0)),
        format_bytes(calc_mem.get("activations_bytes", 0)),
        ir_mem.get("activations_bytes", 0),
        calc_mem.get("activations_bytes", 0),
    )
    ir_grad = ir_mem.get("gradients_bytes", 0)
    print_comparison("梯度", format_bytes(ir_grad) if ir_grad else "-", "-")
    ir_optim = ir_mem.get("optimizer_bytes", 0)
    print_comparison("优化器", format_bytes(ir_optim) if ir_optim else "-", format_bytes(calc_optim_raw) if calc_optim_raw else "-")
    print_comparison_with_diff(
        "合计",
        format_bytes(ir_mem.get("total_bytes", 0)),
        format_bytes(calc_mem.get("total_bytes", 0)),
        ir_mem.get("total_bytes", 0),
        calc_mem.get("total_bytes", 0),
    )
    
    # 总时间
    print_section("总时间")
    print_comparison_with_diff(
        "迭代时间",
        format_time(ir_time.get("iteration_time", 0)) if ir_time.get("iteration_time", 0) else "-",
        format_time(calc_time.get("iteration_time", 0)),
        ir_time.get("iteration_time", 0),
        calc_time.get("iteration_time", 0),
    )
    print_comparison("前向", format_time(ir_time.get("forward_time", 0)) if ir_time.get("forward_time", 0) else "-", "-")
    print_comparison("反向", format_time(ir_time.get("backward_time", 0)) if ir_time.get("backward_time", 0) else "-", "-")
    print_comparison("通信", format_time(ir_time.get("communication_time", 0)) if ir_time.get("communication_time", 0) else "-", "-")
    print_comparison("Bubble", format_time(ir_time.get("bubble_time", 0)) if ir_time.get("bubble_time", 0) else "-", "-")
    print_comparison("优化器", format_time(ir_time.get("optimizer_time", 0)) if ir_time.get("optimizer_time", 0) else "-", "-")
    
    # 单层内存
    print_section("单层内存")
    print_comparison_with_diff(
        "权重",
        format_bytes(ir_block.get("memory", {}).get("weights_bytes", 0)) if ir_block else "-",
        format_bytes(calc_block.get("memory", {}).get("weights_bytes", 0)),
        ir_block.get("memory", {}).get("weights_bytes", 0) if ir_block else 0,
        calc_block.get("memory", {}).get("weights_bytes", 0),
    )
    print_comparison_with_diff(
        "激活",
        format_bytes(ir_block.get("memory", {}).get("activations_bytes", 0)) if ir_block else "-",
        format_bytes(calc_block.get("memory", {}).get("activations_bytes", 0)),
        ir_block.get("memory", {}).get("activations_bytes", 0) if ir_block else 0,
        calc_block.get("memory", {}).get("activations_bytes", 0),
    )
    print_comparison("优化器", format_bytes(ir_block.get("memory", {}).get("optimizer_bytes", 0)) if ir_block and not ir_basis.get("optimizer_sharding") else "-", format_bytes(calc_block.get("memory", {}).get("optimizer_bytes", 0)))
    
    # 单层时间
    print_section("单层时间")
    print_comparison_with_diff(
        "前向",
        format_time(ir_block.get("time", {}).get("forward_time", 0)) if ir_block else "-",
        format_time(calc_block.get("time", {}).get("forward_time", 0)),
        ir_block.get("time", {}).get("forward_time", 0) if ir_block else 0,
        calc_block.get("time", {}).get("forward_time", 0),
    )
    print_comparison_with_diff(
        "激活梯度",
        format_time(ir_block.get("time", {}).get("agrad_time", 0)) if ir_block else "-",
        format_time(calc_block.get("time", {}).get("agrad_time", 0)),
        ir_block.get("time", {}).get("agrad_time", 0) if ir_block else 0,
        calc_block.get("time", {}).get("agrad_time", 0),
    )
    print_comparison_with_diff(
        "权重梯度",
        format_time(ir_block.get("time", {}).get("wgrad_time", 0)) if ir_block else "-",
        format_time(calc_block.get("time", {}).get("wgrad_time", 0)),
        ir_block.get("time", {}).get("wgrad_time", 0) if ir_block else 0,
        calc_block.get("time", {}).get("wgrad_time", 0),
    )
    print_comparison_with_diff(
        "计算合计",
        format_time(ir_block.get("time", {}).get("compute_time", 0)) if ir_block else "-",
        format_time(calc_block.get("time", {}).get("compute_time", 0)),
        ir_block.get("time", {}).get("compute_time", 0) if ir_block else 0,
        calc_block.get("time", {}).get("compute_time", 0),
    )
    print_comparison("TP通信(FW)", "-", format_time(calc_block.get("time", {}).get("tp_comm_fw", 0)))
    print_comparison("TP通信(BW)", "-", format_time(calc_block.get("time", {}).get("tp_comm_bw", 0)))
    print_comparison_with_diff(
        "通信合计",
        format_time(ir_block.get("time", {}).get("comm_time", 0)) if ir_block else "-",
        format_time(calc_block.get("time", {}).get("comm_time", 0)),
        ir_block.get("time", {}).get("comm_time", 0) if ir_block else 0,
        calc_block.get("time", {}).get("comm_time", 0),
    )
    print_comparison_with_diff(
        "单层总计",
        format_time(ir_block.get("time", {}).get("total_time", 0)) if ir_block else "-",
        format_time(calc_block.get("time", {}).get("total_time", 0)),
        ir_block.get("time", {}).get("total_time", 0) if ir_block else 0,
        calc_block.get("time", {}).get("total_time", 0),
    )
    
    return {"ir": ir_result, "calculon": calc_stats}


# ============================================================================
# 主程序
# ============================================================================

def main():
    """主程序入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="GPT-175B 训练仿真")
    parser.add_argument("--model", default="gpt3-175B", help="模型配置名")
    parser.add_argument("--system", default="h100_80g_nvl8", help="系统配置名")
    parser.add_argument("--execution", default="3072_t4_p64_d12_mbs4_full", help="执行配置名")
    parser.add_argument("--output", help="输出 Chrome Trace 文件路径")
    parser.add_argument("-v", "--verbose", action="store_true", help="详细输出")
    args = parser.parse_args()
    
    # 加载配置
    cfg = Config.from_names(args.model, args.system, args.execution)
    
    print_header("GPT 模型训练仿真")
    print(f"模型: {args.model}")
    print(f"系统: {args.system}")
    print(f"执行: {args.execution}")
    
    # 1. 构建图
    print_section("构建计算图")
    graph = build_transformer_graph(cfg)
    print(f"  节点数: {len(graph.nodes)}")
    print(f"  边数:   {len(graph.edges)}")
    print(f"  层数:   {cfg.num_layers}")
    
    # 2. 编译
    print_section("编译")
    compiler = create_compiler(cfg)
    result = compiler.compile(graph)
    print(f"  完成")
    
    # 3. Calculon 对比
    print_section("Calculon 仿真")
    try:
        calc_stats = run_calculon(cfg)
        print(f"  完成")
        compare_results(cfg, result, calc_stats)
    except Exception as e:
        print(f"  错误: {e}")
    
    # 4. 导出 Timeline (可选)
    if args.output:
        print_section("导出 Chrome Trace")
        for ir in compiler._intermediate_results:
            if hasattr(ir, 'to_trace_events'):
                events = ir.to_trace_events()
                with open(args.output, 'w') as f:
                    json.dump(events, f, indent=2)
                print(f"  已导出: {args.output}")
                break
    
    print("\n" + "="*60)
    print("  完成")
    print("="*60)


if __name__ == "__main__":
    main()

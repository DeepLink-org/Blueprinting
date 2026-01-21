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
from typing import Dict, Any

# 添加 src 到 path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

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
console = Console()


# ============================================================================
# 配置加载
# ============================================================================

def _load_json(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def load_configs(model: str, system: str, execution: str) -> Dict[str, Any]:
    """从名称加载配置（自动查找 data/ 目录）"""
    data_dir = ROOT / "data"
    model_path = data_dir / "models" / f"{model}.json"
    system_path = data_dir / "systems" / f"{system}.json"
    execution_path = data_dir / "examples" / f"{execution}.json"
    return {
        "model_name": model,
        "system_name": system,
        "execution_name": execution,
        "model_path": model_path,
        "system_path": system_path,
        "execution_path": execution_path,
        "model": _load_json(model_path),
        "system": _load_json(system_path),
        "execution": _load_json(execution_path),
    }


# ============================================================================
# 输出格式化
# ============================================================================

def format_bytes(n: float) -> str:
    """格式化字节数"""
    if not n:
        return "-"
    if n >= 1e12: return f"{n/1e12:.2f} TB"
    if n >= 1e9: return f"{n/1e9:.2f} GB"
    if n >= 1e6: return f"{n/1e6:.2f} MB"
    return f"{n:.0f} B"


def format_time(s: float) -> str:
    """格式化时间"""
    if not s:
        return "-"
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


def format_diff(ir_val: float, calc_val: float) -> str:
    """格式化相对误差，带颜色"""
    if not calc_val:
        return "-"
    diff = pct_diff(ir_val, calc_val)
    if abs(diff) < 5:
        return f"[green]{diff:+.1f}%[/green]"
    elif abs(diff) < 15:
        return f"[yellow]{diff:+.1f}%[/yellow]"
    else:
        return f"[red]{diff:+.1f}%[/red]"


def normalize_ir_metrics(ir_result, num_layers: int = 1, pp: int = 1) -> Dict[str, Any]:
    """Normalize IR metrics to comparison schema.
    
    Args:
        ir_result: IR simulation result
        num_layers: Total number of transformer layers
        pp: Pipeline parallelism degree
    """
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
    
    # 计算单层指标 (每个 PP stage 的层数)
    layers_per_stage = num_layers // pp if pp > 0 else num_layers
    block_weights = weights_fp32 / layers_per_stage if layers_per_stage > 0 else 0
    block_activations = activations  # 激活是单层的
    block_optimizer = optimizer_states / layers_per_stage if layers_per_stage > 0 else 0
    
    # 时间：单层时间
    block_fw = ir_tb.forward / layers_per_stage if ir_tb and layers_per_stage > 0 else 0
    block_bw = ir_tb.backward / layers_per_stage if ir_tb and layers_per_stage > 0 else 0
    block_comm = ir_tb.communication / layers_per_stage if ir_tb and layers_per_stage > 0 else 0
    
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
        "block": {
            "memory": {
                "weights_bytes": block_weights / 2,  # fp16 权重
                "activations_bytes": block_activations,
                "optimizer_bytes": block_optimizer,
            },
            "time": {
                "forward_time": block_fw,
                "agrad_time": block_bw / 2,  # 假设激活梯度和权重梯度各占一半
                "wgrad_time": block_bw / 2,
                "compute_time": block_fw + block_bw,
                "comm_time": block_comm,
                "total_time": block_fw + block_bw + block_comm,
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

def build_transformer_graph(model: Dict[str, Any], execution: Dict[str, Any], model_name: str) -> GraphIR:
    """构建 Transformer 模型的 GraphIR (层级结构)"""
    hidden = model["hidden"]
    feedforward = model["feedforward"]
    num_heads = model["attn_heads"]
    head_dim = model["attn_size"]
    num_layers = model["num_blocks"]
    seq_len = model.get("seq_size", 2048)
    micro_batch_size = execution["microbatch_size"]
    tp = execution["tensor_par"]
    batch_seq = micro_batch_size * seq_len
    
    builder = IRBuilder(model_name, "Transformer")
    
    # 元数据
    builder.set_metadata("model_name", model_name)
    builder.set_metadata("num_layers", num_layers)
    builder.set_metadata("hidden", hidden)
    builder.set_metadata("feedforward", feedforward)
    builder.set_metadata("num_heads", num_heads)
    builder.set_metadata("head_dim", head_dim)
    builder.set_metadata("batch_size", micro_batch_size)
    builder.set_metadata("seq_len", seq_len)
    builder.set_metadata("tp", tp)
    builder.set_metadata("optimizer_sharding", execution.get("optimizer_sharding", False))
    builder.set_metadata("zero", execution.get("zero", 0))
    builder.set_metadata("activation_recompute", execution.get("activation_recompute", "none"))
    
    # Input
    x = builder.add_input("input_ids", shape=[micro_batch_size, seq_len, hidden])
    prev = x
    
    # Transformer layers
    for i in range(num_layers):
        with builder.block(f"layer{i}", "TransformerLayer", layer_idx=i):
            with builder.block("attn", "Attention"):
                norm = builder.rmsnorm("norm", prev, hidden, batch_seq=batch_seq)
                norm_out = f"{norm.name}_out"
                
                q = builder.linear("q_proj", norm_out, hidden, hidden, 
                                   shard="tp_col", batch_seq=batch_seq)
                k = builder.linear("k_proj", norm_out, hidden, hidden,
                                   shard="tp_col", batch_seq=batch_seq)
                v = builder.linear("v_proj", norm_out, hidden, hidden,
                                   shard="tp_col", batch_seq=batch_seq)
                
                attn = builder.attention("mha", 
                                        [f"{q.name}_out", f"{k.name}_out", f"{v.name}_out"],
                                        num_heads, head_dim, seq_len,
                                        batch_size=micro_batch_size,
                                        shard="tp_col")
                
                attn_out = builder.linear("out_proj", f"{attn.name}_out", hidden, hidden,
                                          shard="tp_row", batch_seq=batch_seq)
                
                res1 = builder.add("residual", [prev, f"{attn_out.name}_out"],
                                   num_elements=batch_seq * hidden)
            
            attn_block_out = f"{res1.name}_out"
            
            with builder.block("ffn", "FFN"):
                norm2 = builder.rmsnorm("norm", attn_block_out, hidden, batch_seq=batch_seq)
                norm2_out = f"{norm2.name}_out"
                
                fc1 = builder.linear("fc1", norm2_out, hidden, feedforward,
                                     shard="tp_col", batch_seq=batch_seq)
                
                act = builder.activation("act", f"{fc1.name}_out", "SiLU",
                                        num_elements=batch_seq * feedforward,
                                        shard="tp_col")
                
                fc2 = builder.linear("fc2", f"{act.name}_out", feedforward, hidden,
                                     shard="tp_row", batch_seq=batch_seq)
                
                res2 = builder.add("residual", [attn_block_out, f"{fc2.name}_out"],
                                   num_elements=batch_seq * hidden)
            
            prev = f"{res2.name}_out"
    
    # Final norm
    with builder.block("final", "Output"):
        final_norm = builder.rmsnorm("norm", prev, hidden, batch_seq=batch_seq)
    
    builder.add_output(f"{final_norm.name}_out")
    
    return builder.build()


# ============================================================================
# 编译流水线
# ============================================================================

def create_compiler(execution: Dict[str, Any], system_path: Path, seq_len: int, debug: bool = False) -> Compiler:
    """创建编译器流水线"""
    compiler = Compiler(debug=debug)
    
    tp = execution["tensor_par"]
    pp = execution["pipeline_par"]
    dp = execution["data_par"]
    batch_size = execution["batch_size"]
    micro_batch_size = execution["microbatch_size"]
    gradient_checkpointing = execution.get("activation_recompute", "none") != "none"
    num_microbatches = batch_size // (micro_batch_size * dp)
    
    compiler.add_pass(WorkloadPass(dtype_bytes=2))
    compiler.add_pass(ParallelPass(
        tp=tp,
        pp=pp,
        dp=dp,
        tp_comm_type=execution.get("tensor_par_comm_type", "ar"),
        sequence_parallel=execution.get("sequence_par", False),
    ))
    compiler.add_pass(SchedulePass(
        system_config=system_path,
        training=True,
    ))
    compiler.add_pass(OptimizerPass(
        OptimizerConfig(
            optimizer_type="adam",
            master_weights=True,
            gradient_checkpointing=gradient_checkpointing,
            recompute_mode="full" if gradient_checkpointing else "none",
            checkpoint_ratio=1.0 if gradient_checkpointing else 0.0,
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
        subs={"batch_seq": micro_batch_size * seq_len},
        training=True,
    ))
    
    return compiler


# ============================================================================
# Calculon 对比
# ============================================================================

def run_calculon(model_cfg: Dict[str, Any], execution_cfg: Dict[str, Any], system_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """运行 Calculon 仿真"""
    with hp.scope(app=model_cfg, exe=execution_cfg) as ps:
        app = Model.from_cfg(ps.app)
        exe = Execution(ps.exe)
        syst = System(system_cfg)
        
        model = Llm(app, logger)
        model.compile(syst, exe)
        model.run(syst)
        
        return model.get_stats_json(False)


def compare_results(model_name: str, model_cfg: Dict[str, Any], execution_cfg: Dict[str, Any], ir_result, calc_stats: Dict[str, Any]):
    """对比 IR 编译器与 Calculon 结果 (使用 Rich 表格)"""
    
    tp = execution_cfg["tensor_par"]
    pp = execution_cfg["pipeline_par"]
    dp = execution_cfg["data_par"]
    num_layers = model_cfg.get("num_blocks", 1)
    
    ir_metrics = normalize_ir_metrics(ir_result, num_layers=num_layers, pp=pp)
    calc_metrics = normalize_calculon_stats(calc_stats)
    
    ir_mem = ir_metrics.get("per_gpu", {}).get("memory", {})
    ir_time = ir_metrics.get("per_gpu", {}).get("time", {})
    ir_block = ir_metrics.get("block", {})
    ir_basis = ir_metrics.get("basis", {})
    
    calc_mem = calc_metrics.get("per_gpu", {}).get("memory", {})
    calc_time = calc_metrics.get("per_gpu", {}).get("time", {})
    calc_block = calc_metrics.get("block", {})
    calc_basis = calc_metrics.get("basis", {})
    
    calc_optim_raw = calc_stats.get("optim_space", None)
    batch_size = execution_cfg["batch_size"]
    micro_batch_size = execution_cfg["microbatch_size"]
    num_gpus = tp * pp * dp
    
    # 配置信息
    config_text = Text()
    config_text.append(f"{model_name}", style="bold cyan")
    config_text.append(f"  TP={tp} PP={pp} DP={dp}  ", style="dim")
    config_text.append(f"GPUs={num_gpus}", style="green")
    config_text.append(f"  Batch={batch_size} MicroBatch={micro_batch_size}", style="dim")
    
    console.print()
    console.print(Panel(config_text, title="[bold]配置[/bold]", border_style="blue"))
    
    # === 总内存表格 ===
    mem_table = Table(title="总内存 (per GPU)", box=box.ROUNDED, show_header=True, header_style="bold magenta")
    mem_table.add_column("指标", style="cyan", width=12)
    mem_table.add_column("IR", justify="right", style="green", width=12)
    mem_table.add_column("Calculon", justify="right", style="yellow", width=12)
    mem_table.add_column("Diff", justify="right", width=10)
    
    ir_weight_basis = ir_basis.get("weights", "fp32")
    calc_weight_basis = calc_basis.get("weights", "fp32")
    ir_weight_bytes = ir_mem.get("weights_fp16_bytes", 0) if ir_weight_basis == "fp16" else ir_mem.get("weights_fp32_bytes", 0)
    calc_weight_bytes = calc_mem.get("weights_fp16_bytes", 0) if calc_weight_basis == "fp16" else calc_mem.get("weights_fp32_bytes", 0)
    
    mem_table.add_row("权重", format_bytes(ir_weight_bytes), format_bytes(calc_weight_bytes),
                      format_diff(ir_weight_bytes, calc_weight_bytes))
    mem_table.add_row("激活", format_bytes(ir_mem.get("activations_bytes", 0)),
                      format_bytes(calc_mem.get("activations_bytes", 0)),
                      format_diff(ir_mem.get("activations_bytes", 0), calc_mem.get("activations_bytes", 0)))
    mem_table.add_row("梯度", format_bytes(ir_mem.get("gradients_bytes", 0)), "-", "-")
    mem_table.add_row("优化器", format_bytes(ir_mem.get("optimizer_bytes", 0)),
                      format_bytes(calc_optim_raw) if calc_optim_raw else "-", "-")
    mem_table.add_row("合计", format_bytes(ir_mem.get("total_bytes", 0)),
                      format_bytes(calc_mem.get("total_bytes", 0)),
                      format_diff(ir_mem.get("total_bytes", 0), calc_mem.get("total_bytes", 0)),
                      style="bold")
    
    console.print(mem_table)
    
    # === 总时间表格 ===
    time_table = Table(title="总时间", box=box.ROUNDED, show_header=True, header_style="bold magenta")
    time_table.add_column("指标", style="cyan", width=12)
    time_table.add_column("IR", justify="right", style="green", width=12)
    time_table.add_column("Calculon", justify="right", style="yellow", width=12)
    time_table.add_column("Diff", justify="right", width=10)
    
    time_table.add_row("迭代时间", format_time(ir_time.get("iteration_time", 0)),
                       format_time(calc_time.get("iteration_time", 0)),
                       format_diff(ir_time.get("iteration_time", 0), calc_time.get("iteration_time", 0)),
                       style="bold")
    time_table.add_row("前向", format_time(ir_time.get("forward_time", 0)), "-", "-")
    time_table.add_row("反向", format_time(ir_time.get("backward_time", 0)), "-", "-")
    time_table.add_row("通信", format_time(ir_time.get("communication_time", 0)), "-", "-")
    time_table.add_row("Bubble", format_time(ir_time.get("bubble_time", 0)), "-", "-")
    time_table.add_row("优化器", format_time(ir_time.get("optimizer_time", 0)), "-", "-")
    
    console.print(time_table)
    
    # === 单层内存表格 ===
    layer_mem_table = Table(title="单层内存", box=box.ROUNDED, show_header=True, header_style="bold magenta")
    layer_mem_table.add_column("指标", style="cyan", width=12)
    layer_mem_table.add_column("IR", justify="right", style="green", width=12)
    layer_mem_table.add_column("Calculon", justify="right", style="yellow", width=12)
    layer_mem_table.add_column("Diff", justify="right", width=10)
    
    ir_layer_weight = ir_block.get("memory", {}).get("weights_bytes", 0) if ir_block else 0
    calc_layer_weight = calc_block.get("memory", {}).get("weights_bytes", 0)
    layer_mem_table.add_row("权重", format_bytes(ir_layer_weight), format_bytes(calc_layer_weight),
                            format_diff(ir_layer_weight, calc_layer_weight))
    
    ir_layer_act = ir_block.get("memory", {}).get("activations_bytes", 0) if ir_block else 0
    calc_layer_act = calc_block.get("memory", {}).get("activations_bytes", 0)
    layer_mem_table.add_row("激活", format_bytes(ir_layer_act), format_bytes(calc_layer_act),
                            format_diff(ir_layer_act, calc_layer_act))
    
    ir_layer_optim = ir_block.get("memory", {}).get("optimizer_bytes", 0) if ir_block else 0
    calc_layer_optim = calc_block.get("memory", {}).get("optimizer_bytes", 0)
    layer_mem_table.add_row("优化器", format_bytes(ir_layer_optim), format_bytes(calc_layer_optim),
                            format_diff(ir_layer_optim, calc_layer_optim) if ir_layer_optim and calc_layer_optim else "-")
    
    console.print(layer_mem_table)
    
    # === 单层时间表格 ===
    layer_time_table = Table(title="单层时间", box=box.ROUNDED, show_header=True, header_style="bold magenta")
    layer_time_table.add_column("指标", style="cyan", width=12)
    layer_time_table.add_column("IR", justify="right", style="green", width=12)
    layer_time_table.add_column("Calculon", justify="right", style="yellow", width=12)
    layer_time_table.add_column("Diff", justify="right", width=10)
    
    def add_time_row(name, ir_key, calc_key):
        ir_val = ir_block.get("time", {}).get(ir_key, 0) if ir_block else 0
        calc_val = calc_block.get("time", {}).get(calc_key, 0)
        layer_time_table.add_row(name, format_time(ir_val), format_time(calc_val),
                                  format_diff(ir_val, calc_val))
    
    add_time_row("前向", "forward_time", "forward_time")
    add_time_row("激活梯度", "agrad_time", "agrad_time")
    add_time_row("权重梯度", "wgrad_time", "wgrad_time")
    add_time_row("计算合计", "compute_time", "compute_time")
    
    ir_comm_fw = ir_block.get("time", {}).get("comm_fw", 0) if ir_block else 0
    ir_comm_bw = ir_block.get("time", {}).get("comm_bw", 0) if ir_block else 0
    layer_time_table.add_row(
        "TP通信(FW)",
        format_time(ir_comm_fw) if ir_comm_fw else "-",
        format_time(calc_block.get("time", {}).get("tp_comm_fw", 0)),
        format_diff(ir_comm_fw, calc_block.get("time", {}).get("tp_comm_fw", 0)) if ir_comm_fw else "-",
    )
    layer_time_table.add_row(
        "TP通信(BW)",
        format_time(ir_comm_bw) if ir_comm_bw else "-",
        format_time(calc_block.get("time", {}).get("tp_comm_bw", 0)),
        format_diff(ir_comm_bw, calc_block.get("time", {}).get("tp_comm_bw", 0)) if ir_comm_bw else "-",
    )
    
    add_time_row("通信合计", "comm_time", "comm_time")
    
    ir_total = ir_block.get("time", {}).get("total_time", 0) if ir_block else 0
    calc_total = calc_block.get("time", {}).get("total_time", 0)
    layer_time_table.add_row("单层总计", format_time(ir_total), format_time(calc_total),
                              format_diff(ir_total, calc_total), style="bold")
    
    console.print(layer_time_table)
    
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
    parser.add_argument("--debug", action="store_true", help="打印每个 pass 的 IR 摘要")
    parser.add_argument("-v", "--verbose", action="store_true", help="详细输出")
    args = parser.parse_args()
    
    # 加载配置
    cfg = load_configs(args.model, args.system, args.execution)
    model_cfg = cfg["model"]
    system_cfg = cfg["system"]
    execution_cfg = cfg["execution"]
    model_name = cfg["model_name"]
    
    # Header
    console.print()
    console.print(Panel.fit(
        f"[bold cyan]GPT 模型训练仿真[/bold cyan]\n\n"
        f"模型: [green]{args.model}[/green]\n"
        f"系统: [green]{args.system}[/green]\n"
        f"执行: [green]{args.execution}[/green]",
        border_style="blue"
    ))
    
    # 1. 构建图
    console.print("\n[bold cyan]▶ 构建计算图[/bold cyan]")
    graph = build_transformer_graph(model_cfg, execution_cfg, model_name)
    console.print(f"  结构: {graph}")
    console.print(f"  层数: {model_cfg['num_blocks']}")
    if args.verbose:
        console.print()
        console.print(graph.tree(max_depth=2))
    
    # 2. 编译
    console.print("\n[bold cyan]▶ 编译[/bold cyan]")
    compiler = create_compiler(
        execution_cfg,
        cfg["system_path"],
        model_cfg.get("seq_size", 2048),
        debug=args.debug,
    )
    result = compiler.compile(graph)
    console.print("  [green]✓[/green] 完成")
    
    # 3. Calculon 对比
    console.print("\n[bold cyan]▶ Calculon 仿真[/bold cyan]")
    try:
        calc_stats = run_calculon(model_cfg, execution_cfg, system_cfg)
        console.print("  [green]✓[/green] 完成")
        
        console.print()
        console.rule("[bold]IR 编译器 vs Calculon 对比[/bold]", style="blue")
        compare_results(model_name, model_cfg, execution_cfg, result, calc_stats)
    except Exception as e:
        console.print(f"  [red]✗[/red] 错误: {e}")
    
    console.print()
    console.rule("[green]完成[/green]", style="green")


if __name__ == "__main__":
    main()

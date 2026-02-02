#!/usr/bin/env python3
"""GPT-175B 模型训练仿真 - V2 (新架构)

本示例使用新的 passes_v2 架构对 GPT-175B 模型进行训练仿真分析，
并与 Calculon 的结果进行对比。

新架构编译流程:
    GraphIR → ParallelPass → ExpandPass → SchedulePass 
           → OptimizerPass → PipelineSchedulePass → TimelinePassV2 
           → OverlapAnalysisPass → SimulatePass

使用方法:
    python examples/gpt175b_compile_v2.py
    
配置文件:
    模型: data/models/gpt3-175B.json
    系统: data/systems/h100_80g_nvl8.json
    执行: data/examples/3072_t4_p64_d12_mbs4_full.json
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# 添加 src 到 path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

# 新架构导入
from blueprinting.ir.dsl import Transformer, Model
from blueprinting.ir.types import GraphIR
from blueprinting.ir.passes_v2 import (
    Pipeline,
    PrintGraphPass,
    ExpandPass,
    SchedulePass,
    ParallelPass,
    OptimizerPass,
    OptimizerConfig,
    PipelineSchedulePass,
    PipelineConfig,
    PPScheduleMode,
    TimelinePassV2,
    OverlapAnalysisPass,
    SimulatePass,
    PrintSchedulePass,
)

# Calculon 对比 (可选)
try:
    from blueprinting import Model as CalcModel, Execution
    from calculon import System
    from calculon.llm import Llm
    import hyperparameter as hp
    HAS_CALCULON = True
except ImportError:
    HAS_CALCULON = False

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


# ============================================================================
# 指标归一化
# ============================================================================

def normalize_ir_metrics(ir_result) -> Dict[str, Any]:
    """Normalize IR metrics to comparison schema.
    
    SimulatePass 已经计算了所有指标，这里只做格式转换。
    
    Args:
        ir_result: IR simulation result (SimulationResult)
    """
    if not ir_result:
        return {}
    
    # 从 SimulationResult 获取各种指标
    ir_mb = getattr(ir_result, "memory_breakdown", None)
    ir_tb = getattr(ir_result, "time_breakdown", None)
    ir_total_tb = getattr(ir_result, "total_time_breakdown", None)
    ir_block = getattr(ir_result, "block_metrics", None)
    
    # per-GPU 内存指标
    weights = ir_mb.weights if ir_mb else 0
    activations = ir_mb.activations if ir_mb else 0
    gradients = ir_mb.gradients if ir_mb else 0
    optimizer_states = ir_mb.optimizer_states if ir_mb else 0
    total_memory = ir_mb.total if ir_mb else 0
    
    # 总时间指标
    total_fw = ir_total_tb.forward if ir_total_tb else 0
    total_bw = ir_total_tb.backward if ir_total_tb else 0
    total_comm = ir_total_tb.communication if ir_total_tb else 0
    total_bubble = ir_total_tb.bubble if ir_total_tb else 0
    
    # 单层指标（由 SimulatePass 计算）
    block_weights = ir_block.weights if ir_block else 0
    block_activations = ir_block.activations if ir_block else 0
    block_optimizer = ir_block.optimizer_states if ir_block else 0
    block_fw = ir_block.forward_time if ir_block else 0
    block_bw = ir_block.backward_time if ir_block else 0
    block_comm = ir_block.communication_time if ir_block else 0
    block_compute = ir_block.compute_time if ir_block else 0
    block_total = ir_block.total_time if ir_block else 0
    
    return {
        "schema": "comparison_v1",
        "units": {"memory": "bytes", "time": "seconds"},
        "basis": {
            "weights": "fp16",
            "activations": "unknown",
            "gradients": "fp16",
            "optimizer_states": "fp32",
        },
        "per_gpu": {
            "memory": {
                "weights_fp16_bytes": weights,
                "activations_bytes": activations,
                "activations_peak_bytes": getattr(ir_result, "peak_memory", 0),
                "gradients_bytes": gradients,
                "optimizer_bytes": optimizer_states,
                "total_bytes": total_memory,
            },
            "time": {
                "iteration_time": getattr(ir_result, "e2e_time", 0) or 0,
                "forward_time": total_fw,
                "backward_time": total_bw,
                "communication_time": total_comm,
                "bubble_time": total_bubble,
            },
        },
        "block": {
            "memory": {
                "weights_bytes": block_weights,
                "activations_bytes": block_activations,
                "optimizer_bytes": block_optimizer,
            },
            "time": {
                "forward_time": block_fw,
                "agrad_time": block_bw / 2,  # Calculon 兼容：假设 agrad ≈ wgrad
                "wgrad_time": block_bw / 2,
                "compute_time": block_compute,
                "comm_time": block_comm,
                "total_time": block_total,
            },
        },
    }


def normalize_calculon_stats(calc_stats: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize Calculon stats to comparison schema."""
    weight = calc_stats.get("weight_space", 0) or 0
    act = calc_stats.get("act_space", 0) or 0
    optim = calc_stats.get("optimizer_space", 0) or 0  # 修正字段名
    total_memory = weight + act + optim
    
    block_weight = calc_stats.get("block_weight_space", 0) or 0
    block_act = calc_stats.get("block_act_working_space", 0) or 0
    block_optim = calc_stats.get("block_optimizer_space", 0) or 0
    
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
                "forward_time": calc_stats.get("fw_time", 0),
                "backward_time": calc_stats.get("bw_time", 0),
                "communication_time": calc_stats.get("tp_comm_exposed_time", 0) + calc_stats.get("dp_comm_exposed_time", 0) + calc_stats.get("pp_comm_exposed_time", 0),
                "bubble_time": calc_stats.get("bubble_time", 0),
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
# 使用新 DSL 构建模型
# ============================================================================

def build_transformer_graph_v2(
    model_name: str,
    hidden: int,
    feedforward: int,
    num_heads: int,
    head_dim: int,
    num_layers: int,
    seq_len: int,
    batch_size: int,
    tp: int = 1,
    pp: int = 1,
    dp: int = 1,
    gradient_checkpointing: bool = False,
    optimizer_sharding: bool = False,
) -> GraphIR:
    """使用新 DSL 构建 Transformer 模型的 GraphIR.
    
    Args:
        model_name: 模型名称
        hidden: 隐藏层维度
        feedforward: FFN 中间层维度
        num_heads: 注意力头数
        head_dim: 每个头的维度
        num_layers: Transformer 层数
        seq_len: 序列长度
        batch_size: batch 大小 (micro-batch)
        tp: Tensor Parallelism 度数
        pp: Pipeline Parallelism 度数
        dp: Data Parallelism 度数
    
    Returns:
        GraphIR
    """
    batch_seq = batch_size * seq_len
    
    with Transformer(model_name) as m:
        # 元数据
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
            optimizer_sharding=optimizer_sharding,
        )
        
        # Transformer layers
        for i in range(num_layers):
            with m.TransformerLayer(f"layer{i}") as layer:
                # Attention block
                with layer.Attention("attn") as attn:
                    # Pre-norm
                    attn.RMSNorm("norm", normalized_shape=hidden)
                    
                    # QKV projections (column parallel)
                    attn.Linear("q_proj", in_features=hidden, out_features=hidden, shard="tp_col")
                    attn.Linear("k_proj", in_features=hidden, out_features=hidden, shard="tp_col")
                    attn.Linear("v_proj", in_features=hidden, out_features=hidden, shard="tp_col")
                    
                    # Output projection (row parallel)
                    attn.Linear("out_proj", in_features=hidden, out_features=hidden, shard="tp_row")
                
                # FFN block
                with layer.FFN("ffn") as ffn:
                    # Pre-norm
                    ffn.RMSNorm("norm", normalized_shape=hidden)
                    
                    # Up projection (column parallel)
                    ffn.Linear("fc1", in_features=hidden, out_features=feedforward, shard="tp_col")
                    
                    # Down projection (row parallel)
                    ffn.Linear("fc2", in_features=feedforward, out_features=hidden, shard="tp_row")
    
    return m.build()


# ============================================================================
# 编译流水线 (新架构)
# ============================================================================

def create_compiler_v2(
    tp: int = 1,
    pp: int = 1,
    dp: int = 1,
    num_microbatches: int = 1,
    training: bool = True,
    gradient_checkpointing: bool = False,
    peak_tflops: float = 1000.0,  # H100 FP16 峰值
    memory_bandwidth: float = 3.35e12,  # H100 内存带宽 3.35 TB/s
    network_bandwidth: float = 450e9 * 0.65,  # NVLink 450 GB/s × 0.65 效率 (与 Calculon 对齐)
    debug: bool = False,
) -> Pipeline:
    """创建新架构的编译流水线.
    
    Args:
        tp: Tensor Parallelism 度数
        pp: Pipeline Parallelism 度数
        dp: Data Parallelism 度数
        num_microbatches: micro-batch 数量
        training: 是否训练模式
        gradient_checkpointing: 是否启用梯度检查点
        peak_tflops: 硬件峰值算力 (TFLOPS)
        memory_bandwidth: 内存带宽 (bytes/s)
        network_bandwidth: 网络带宽 (bytes/s)
        debug: 是否打印调试信息
    
    Returns:
        Pipeline
    """
    passes = []
    
    # 调试: 打印 Graph IR
    if debug:
        passes.append(PrintGraphPass("Graph IR"))
    
    # 1. ParallelPass: 标记并行策略
    passes.append(ParallelPass(
        tp=tp,
        pp=pp,
        dp=dp,
        tp_comm_type="ar",  # AllReduce
    ))
    
    # 2. ExpandPass: Block → Op
    passes.append(ExpandPass())
    
    # 3. SchedulePass: 计算 workload + timing
    # 参数与 Calculon H100 配置对齐:
    # - network_efficiency: 0.65 (NVLink 效率)
    # - all_reduce_offset: 1.0 (AllReduce 双向通信偏移)
    passes.append(SchedulePass(
        peak_tflops=peak_tflops,
        memory_bandwidth=memory_bandwidth,
        network_bandwidth=network_bandwidth,
        network_efficiency=0.65,  # NVLink 效率 (与 Calculon H100 配置对齐)
        network_latency=10e-6,  # 10µs 延迟
        compute_efficiency=0.95,  # 95% 计算效率
        all_reduce_offset=1.0,  # AllReduce 通信偏移量 (Calculon 模型)
    ))
    
    # 调试: 打印 Schedule IR
    if debug:
        passes.append(PrintSchedulePass("Schedule IR (after SchedulePass)"))
    
    # 4. OptimizerPass: 追加反向 Op (训练模式)
    if training:
        passes.append(OptimizerPass(
            optimizer_config=OptimizerConfig(
                optimizer_type="adam",
                master_weights=True,
                gradient_checkpointing=gradient_checkpointing,
                recompute_mode="full" if gradient_checkpointing else "none",
            ),
            training=True,
            memory_bandwidth=memory_bandwidth,
            peak_flops=peak_tflops * 1e12,
        ))
    
    # 5. PipelineSchedulePass: PP 调度 (PP > 1 时)
    if pp > 1 and num_microbatches > 1:
        passes.append(PipelineSchedulePass(
            config=PipelineConfig(
                mode=PPScheduleMode.ONE_F_ONE_B,
                num_microbatches=num_microbatches,
                num_stages=pp,
            ),
        ))
    
    # 6. TimelinePassV2: Op → Event
    passes.append(TimelinePassV2(track_memory=True))
    
    # 7. OverlapAnalysisPass: 重叠分析
    passes.append(OverlapAnalysisPass())
    
    # 8. SimulatePass: 最终评估
    passes.append(SimulatePass(
        peak_tflops=peak_tflops,
        training=training,
    ))
    
    return Pipeline(passes)


# ============================================================================
# Calculon 对比
# ============================================================================

def run_calculon(
    model_cfg: Dict[str, Any],
    execution_cfg: Dict[str, Any],
    system_cfg: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """运行 Calculon 仿真"""
    if not HAS_CALCULON:
        console.print("  [yellow]跳过[/yellow]: Calculon 未安装")
        return None
    
    
    try:
        with hp.scope(app=model_cfg, exe=execution_cfg) as ps:
            app = CalcModel.from_cfg(ps.app)
            exe = Execution(ps.exe)
            syst = System(system_cfg)
            
            model = Llm(app, logger)
            model.compile(syst, exe)
            model.run(syst)
            
            return model.get_stats_json(False)
    except Exception as e:
        console.print(f"  [red]错误[/red]: {e}")
        return None


def compare_results(
    model_name: str,
    model_cfg: Dict[str, Any],
    execution_cfg: Dict[str, Any],
    ir_result,
    calc_stats: Optional[Dict[str, Any]],
):
    """对比 IR 编译器与 Calculon 结果 (使用 Rich 表格)"""
    
    tp = execution_cfg.get("tensor_par", 1)
    pp = execution_cfg.get("pipeline_par", 1)
    dp = execution_cfg.get("data_par", 1)
    num_layers = model_cfg.get("num_blocks", 1)
    batch_size = execution_cfg.get("batch_size", 1)
    micro_batch_size = execution_cfg.get("microbatch_size", 1)
    num_microbatches = batch_size // (micro_batch_size * dp) if micro_batch_size * dp > 0 else 1
    num_gpus = tp * pp * dp
    
    ir_metrics = normalize_ir_metrics(ir_result)
    calc_metrics = normalize_calculon_stats(calc_stats) if calc_stats else {}
    
    ir_mem = ir_metrics.get("per_gpu", {}).get("memory", {})
    ir_time = ir_metrics.get("per_gpu", {}).get("time", {})
    ir_block = ir_metrics.get("block", {})
    ir_basis = ir_metrics.get("basis", {})
    
    calc_mem = calc_metrics.get("per_gpu", {}).get("memory", {})
    calc_time = calc_metrics.get("per_gpu", {}).get("time", {})
    calc_block = calc_metrics.get("block", {})
    calc_basis = calc_metrics.get("basis", {})
    
    calc_optim_raw = calc_stats.get("optimizer_space", None) if calc_stats else None
    
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
    mem_table.add_column("IR (V2)", justify="right", style="green", width=12)
    mem_table.add_column("Calculon", justify="right", style="yellow", width=12)
    mem_table.add_column("Diff", justify="right", width=10)
    
    ir_weight_basis = ir_basis.get("weights", "fp32")
    calc_weight_basis = calc_basis.get("weights", "fp32")
    ir_weight_bytes = ir_mem.get("weights_fp16_bytes", 0) if ir_weight_basis == "fp16" else ir_mem.get("weights_fp32_bytes", 0)
    calc_weight_bytes = calc_mem.get("weights_fp16_bytes", 0) if calc_weight_basis == "fp16" else calc_mem.get("weights_fp32_bytes", 0)
    
    mem_table.add_row("权重", format_bytes(ir_weight_bytes), format_bytes(calc_weight_bytes),
                      format_diff(ir_weight_bytes, calc_weight_bytes) if calc_weight_bytes else "-")
    mem_table.add_row("激活", format_bytes(ir_mem.get("activations_bytes", 0)),
                      format_bytes(calc_mem.get("activations_bytes", 0)),
                      format_diff(ir_mem.get("activations_bytes", 0), calc_mem.get("activations_bytes", 0)) if calc_mem.get("activations_bytes") else "-")
    mem_table.add_row("梯度", format_bytes(ir_mem.get("gradients_bytes", 0)), "-", "-")
    mem_table.add_row("优化器", format_bytes(ir_mem.get("optimizer_bytes", 0)),
                      format_bytes(calc_optim_raw) if calc_optim_raw else "-", "-")
    mem_table.add_row("合计", format_bytes(ir_mem.get("total_bytes", 0)),
                      format_bytes(calc_mem.get("total_bytes", 0)),
                      format_diff(ir_mem.get("total_bytes", 0), calc_mem.get("total_bytes", 0)) if calc_mem.get("total_bytes") else "-",
                      style="bold")
    
    console.print(mem_table)
    
    # === 总时间表格 ===
    time_table = Table(title="总时间", box=box.ROUNDED, show_header=True, header_style="bold magenta")
    time_table.add_column("指标", style="cyan", width=12)
    time_table.add_column("IR (V2)", justify="right", style="green", width=12)
    time_table.add_column("Calculon", justify="right", style="yellow", width=12)
    time_table.add_column("Diff", justify="right", width=10)
    
    time_table.add_row("迭代时间", format_time(ir_time.get("iteration_time", 0)),
                       format_time(calc_time.get("iteration_time", 0)),
                       format_diff(ir_time.get("iteration_time", 0), calc_time.get("iteration_time", 0)) if calc_time.get("iteration_time") else "-",
                       style="bold")
    time_table.add_row("前向", format_time(ir_time.get("forward_time", 0)), 
                       format_time(calc_time.get("forward_time", 0)) if calc_time.get("forward_time") else "-",
                       format_diff(ir_time.get("forward_time", 0), calc_time.get("forward_time", 0)) if calc_time.get("forward_time") else "-")
    time_table.add_row("反向", format_time(ir_time.get("backward_time", 0)),
                       format_time(calc_time.get("backward_time", 0)) if calc_time.get("backward_time") else "-",
                       format_diff(ir_time.get("backward_time", 0), calc_time.get("backward_time", 0)) if calc_time.get("backward_time") else "-")
    time_table.add_row("通信", format_time(ir_time.get("communication_time", 0)),
                       format_time(calc_time.get("communication_time", 0)) if calc_time.get("communication_time") else "-",
                       format_diff(ir_time.get("communication_time", 0), calc_time.get("communication_time", 0)) if calc_time.get("communication_time") else "-")
    time_table.add_row("Bubble", format_time(ir_time.get("bubble_time", 0)),
                       format_time(calc_time.get("bubble_time", 0)) if calc_time.get("bubble_time") else "-",
                       format_diff(ir_time.get("bubble_time", 0), calc_time.get("bubble_time", 0)) if calc_time.get("bubble_time") else "-")
    
    console.print(time_table)
    
    # === 单层内存表格 ===
    layer_mem_table = Table(title="单层内存", box=box.ROUNDED, show_header=True, header_style="bold magenta")
    layer_mem_table.add_column("指标", style="cyan", width=12)
    layer_mem_table.add_column("IR (V2)", justify="right", style="green", width=12)
    layer_mem_table.add_column("Calculon", justify="right", style="yellow", width=12)
    layer_mem_table.add_column("Diff", justify="right", width=10)
    
    ir_layer_weight = ir_block.get("memory", {}).get("weights_bytes", 0) if ir_block else 0
    calc_layer_weight = calc_block.get("memory", {}).get("weights_bytes", 0)
    layer_mem_table.add_row("权重", format_bytes(ir_layer_weight), format_bytes(calc_layer_weight),
                            format_diff(ir_layer_weight, calc_layer_weight) if calc_layer_weight else "-")
    
    ir_layer_act = ir_block.get("memory", {}).get("activations_bytes", 0) if ir_block else 0
    calc_layer_act = calc_block.get("memory", {}).get("activations_bytes", 0)
    layer_mem_table.add_row("激活", format_bytes(ir_layer_act), format_bytes(calc_layer_act),
                            format_diff(ir_layer_act, calc_layer_act) if calc_layer_act else "-")
    
    ir_layer_optim = ir_block.get("memory", {}).get("optimizer_bytes", 0) if ir_block else 0
    calc_layer_optim = calc_block.get("memory", {}).get("optimizer_bytes", 0)
    layer_mem_table.add_row("优化器", format_bytes(ir_layer_optim), format_bytes(calc_layer_optim),
                            format_diff(ir_layer_optim, calc_layer_optim) if ir_layer_optim and calc_layer_optim else "-")
    
    console.print(layer_mem_table)
    
    # === 单层时间表格 ===
    layer_time_table = Table(title="单层时间", box=box.ROUNDED, show_header=True, header_style="bold magenta")
    layer_time_table.add_column("指标", style="cyan", width=12)
    layer_time_table.add_column("IR (V2)", justify="right", style="green", width=12)
    layer_time_table.add_column("Calculon", justify="right", style="yellow", width=12)
    layer_time_table.add_column("Diff", justify="right", width=10)
    
    def add_time_row(name, ir_key, calc_key):
        ir_val = ir_block.get("time", {}).get(ir_key, 0) if ir_block else 0
        calc_val = calc_block.get("time", {}).get(calc_key, 0)
        layer_time_table.add_row(name, format_time(ir_val), format_time(calc_val),
                                  format_diff(ir_val, calc_val) if calc_val else "-")
    
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
        format_diff(ir_comm_fw, calc_block.get("time", {}).get("tp_comm_fw", 0)) if ir_comm_fw and calc_block.get("time", {}).get("tp_comm_fw") else "-",
    )
    layer_time_table.add_row(
        "TP通信(BW)",
        format_time(ir_comm_bw) if ir_comm_bw else "-",
        format_time(calc_block.get("time", {}).get("tp_comm_bw", 0)),
        format_diff(ir_comm_bw, calc_block.get("time", {}).get("tp_comm_bw", 0)) if ir_comm_bw and calc_block.get("time", {}).get("tp_comm_bw") else "-",
    )
    
    add_time_row("通信合计", "comm_time", "comm_time")
    
    ir_total = ir_block.get("time", {}).get("total_time", 0) if ir_block else 0
    calc_total = calc_block.get("time", {}).get("total_time", 0)
    layer_time_table.add_row("单层总计", format_time(ir_total), format_time(calc_total),
                              format_diff(ir_total, calc_total) if calc_total else "-", style="bold")
    
    console.print(layer_time_table)
    
    # === 效率指标 ===
    eff_table = Table(title="效率指标", box=box.ROUNDED, show_header=True, header_style="bold magenta")
    eff_table.add_column("指标", style="cyan", width=14)
    eff_table.add_column("IR (V2)", justify="right", style="green", width=12)
    eff_table.add_column("Calculon", justify="right", style="yellow", width=12)
    eff_table.add_column("Diff", justify="right", width=10)
    
    ir_mfu = ir_result.mfu if ir_result else 0
    calc_mfu = calc_stats.get("proc_mfu", 0) if calc_stats else 0
    eff_table.add_row("MFU", f"{ir_mfu:.1%}", f"{calc_mfu:.1%}" if calc_mfu else "-",
                      format_diff(ir_mfu, calc_mfu) if calc_mfu else "-")
    
    ir_throughput = ir_result.config.get("tokens_per_second", 0) if ir_result else 0
    eff_table.add_row("Tokens/s", format_num(ir_throughput), "-", "-")
    
    console.print(eff_table)
    
    return {"ir": ir_result, "calculon": calc_stats}


# ============================================================================
# 主程序
# ============================================================================

def main():
    """主程序入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="GPT-175B 训练仿真 (V2 新架构)")
    parser.add_argument("--model", default="gpt3-175B", help="模型配置名")
    parser.add_argument("--system", default="h100_80g_nvl8", help="系统配置名")
    parser.add_argument("--execution", default="3072_t4_p64_d12_mbs4_full", help="执行配置名")
    parser.add_argument("--layers", type=int, default=None, help="覆盖层数 (用于快速测试)")
    parser.add_argument("--debug", action="store_true", help="打印调试信息")
    parser.add_argument("--skip-calculon", action="store_true", help="跳过 Calculon 对比")
    parser.add_argument("-v", "--verbose", action="store_true", help="详细输出")
    args = parser.parse_args()
    
    # 加载配置
    cfg = load_configs(args.model, args.system, args.execution)
    model_cfg = cfg["model"]
    system_cfg = cfg["system"]
    execution_cfg = cfg["execution"]
    model_name = cfg["model_name"]
    
    # 提取参数
    hidden = model_cfg.get("hidden", 12288)
    feedforward = model_cfg.get("feedforward", hidden * 4)
    num_heads = model_cfg.get("attn_heads", 96)
    head_dim = model_cfg.get("attn_size", hidden // num_heads)
    num_layers = args.layers or model_cfg.get("num_blocks", 96)
    seq_len = model_cfg.get("seq_size", 2048)
    
    tp = execution_cfg.get("tensor_par", 4)
    pp = execution_cfg.get("pipeline_par", 48)
    dp = execution_cfg.get("data_par", 12)
    batch_size = execution_cfg.get("batch_size", 3072)
    micro_batch_size = execution_cfg.get("microbatch_size", 4)
    num_microbatches = batch_size // (micro_batch_size * dp)
    gradient_checkpointing = execution_cfg.get("activation_recompute", "none") != "none"
    optimizer_sharding = execution_cfg.get("optimizer_sharding", False)
    
    # 调整 PP 以匹配层数
    if num_layers % pp != 0:
        # 找到能整除层数的 PP
        for new_pp in [num_layers, num_layers // 2, num_layers // 4, num_layers // 8, 1]:
            if num_layers % new_pp == 0 and new_pp > 0:
                pp = new_pp
                break
        console.print(f"[yellow]调整 PP={pp} 以匹配层数 {num_layers}[/yellow]")
    
    # 更新执行配置
    execution_cfg_adj = dict(execution_cfg)
    execution_cfg_adj["pipeline_par"] = pp
    
    # 更新模型配置 (用于对比)
    model_cfg_adj = dict(model_cfg)
    model_cfg_adj["num_blocks"] = num_layers
    
    # Header
    console.print()
    console.print(Panel.fit(
        f"[bold cyan]GPT 模型训练仿真 (V2 新架构)[/bold cyan]\n\n"
        f"模型: [green]{args.model}[/green]  (H={hidden}, L={num_layers})\n"
        f"系统: [green]{args.system}[/green]\n"
        f"执行: [green]{args.execution}[/green]  (TP={tp}, PP={pp}, DP={dp})",
        border_style="blue"
    ))
    
    # 1. 构建图 (使用新 DSL)
    console.print("\n[bold cyan]▶ 构建计算图 (新 DSL)[/bold cyan]")
    graph = build_transformer_graph_v2(
        model_name=model_name,
        hidden=hidden,
        feedforward=feedforward,
        num_heads=num_heads,
        head_dim=head_dim,
        num_layers=num_layers,
        seq_len=seq_len,
        batch_size=micro_batch_size,
        tp=tp,
        pp=pp,
        dp=dp,
        gradient_checkpointing=gradient_checkpointing,
        optimizer_sharding=optimizer_sharding,
    )
    console.print(f"  结构: {graph}")
    console.print(f"  Block 总数: {graph.count_blocks()}")
    
    if args.verbose:
        console.print()
        from blueprinting.ir.dsl import print_graph
        print_graph(graph)
    
    # 2. 编译 (新架构)
    console.print("\n[bold cyan]▶ 编译 (新架构 Pipeline)[/bold cyan]")
    
    # 系统参数
    peak_tflops = system_cfg.get("peak_tflops", 1000)
    memory_bandwidth = system_cfg.get("memory_bandwidth_gbps", 3072) * 1e9
    
    pipeline = create_compiler_v2(
        tp=tp,
        pp=pp,
        dp=dp,
        num_microbatches=num_microbatches,
        training=True,
        gradient_checkpointing=gradient_checkpointing,
        peak_tflops=peak_tflops,
        memory_bandwidth=memory_bandwidth,
        debug=args.debug,
    )
    
    console.print(f"  Pipeline: {pipeline}")
    
    result = pipeline.run(graph)
    console.print("  [green]✓[/green] 完成")
    
    # 打印结果摘要
    console.print(f"\n  E2E Time: {result.e2e_time*1e3:.2f} ms")
    console.print(f"  Peak Memory: {result.peak_memory/1e9:.2f} GB")
    console.print(f"  Total FLOPs: {result.total_flops:.2e}")
    console.print(f"  MFU: {result.mfu:.1%}")
    
    # 3. Calculon 对比 (可选)
    calc_stats = None
    if not args.skip_calculon:
        console.print("\n[bold cyan]▶ Calculon 仿真[/bold cyan]")
        calc_stats = run_calculon(model_cfg_adj, execution_cfg_adj, system_cfg)
        if calc_stats:
            console.print("  [green]✓[/green] 完成")
    
    # 4. 对比结果
    console.print()
    console.rule("[bold]IR (V2) vs Calculon 对比[/bold]", style="blue")
    
    compare_results(model_name, model_cfg_adj, execution_cfg_adj, result, calc_stats)
    
    console.print()
    console.rule("[green]完成[/green]", style="green")


if __name__ == "__main__":
    main()

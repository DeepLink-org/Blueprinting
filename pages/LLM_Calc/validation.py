"""LLM 训练计算器 - 准确性校验页面"""

import json
import logging
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import streamlit as st
import hyperparameter as hp

from blueprinting.ui import setup_page, page_title, section_header, info_card
from blueprinting.validations.cases.seqsel_fig1 import seqsel_fig1
from blueprinting.validations.cases.seqsel_fig7 import seqsel_fig7
from blueprinting.validations.cases.seqsel_tab5 import seqsel_tab5
from blueprinting.ir import GraphIR
from blueprinting import Model, Execution
from calculon import System
from calculon.llm import Llm

# 从 ir_pipeline 导入构建函数
from pages.LLM_Calc.ir_pipeline import build_transformer_graph, create_compiler, extract_scope_params


# ============================================================================
# 页面初始化
# ============================================================================
setup_page(title="LLM训练计算器 - 准确性校验")

page_title(
    "模拟器准确性校验",
    subtitle="验证模拟结果与实际测量数据的一致性",
    icon="✅"
)


# ============================================================================
# 说明信息
# ============================================================================
info_card(
    "验证数据来源",
    "本页面展示模拟器与实际测量数据的对比验证结果。数据来源于 SeqSel 论文的公开数据集，用于评估模拟器的准确性。",
    icon="📊"
)

st.markdown("")


# ============================================================================
# IR vs Calculon 对比
# ============================================================================
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def load_configs(model: str, system: str, execution: str) -> Dict[str, Any]:
    """从名称加载配置（自动查找 data/ 目录）"""
    model_path = DATA_DIR / "models" / f"{model}.json"
    system_path = DATA_DIR / "systems" / f"{system}.json"
    execution_path = DATA_DIR / "examples" / f"{execution}.json"
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


def format_bytes(n: float) -> str:
    if not n:
        return "-"
    if n >= 1e12:
        return f"{n/1e12:.2f} TB"
    if n >= 1e9:
        return f"{n/1e9:.2f} GB"
    if n >= 1e6:
        return f"{n/1e6:.2f} MB"
    return f"{n:.0f} B"


def format_time(s: float) -> str:
    if not s:
        return "-"
    if s >= 1:
        return f"{s:.2f} s"
    if s >= 1e-3:
        return f"{s*1e3:.2f} ms"
    return f"{s*1e6:.2f} µs"


def pct_diff(a: float, b: float) -> float:
    if b == 0:
        return 0
    return (a - b) / b * 100


def format_diff(a: float, b: float) -> str:
    if not b:
        return "-"
    return f"{pct_diff(a, b):+.1f}%"


def normalize_ir_metrics(ir_result, num_layers: int = 1, pp: int = 1) -> Dict[str, Any]:
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

    layers_per_stage = num_layers // pp if pp > 0 else num_layers
    block_weights = weights_fp32 / layers_per_stage if layers_per_stage > 0 else 0
    block_activations = activations
    block_optimizer = optimizer_states / layers_per_stage if layers_per_stage > 0 else 0

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
                "optimizer_time": getattr(ir_tb, "optimizer", 0) if ir_tb else 0,
                "communication_time": ir_tb.communication if ir_tb else 0,
                "bubble_time": ir_tb.bubble if ir_tb else 0,
            },
        },
        "block": {
            "memory": {
                "weights_bytes": block_weights / 2,
                "activations_bytes": block_activations,
                "optimizer_bytes": block_optimizer,
            },
            "time": {
                "forward_time": block_fw,
                "agrad_time": block_bw / 2,
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


def run_calculon(model_cfg: Dict[str, Any], execution_cfg: Dict[str, Any], system_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """运行 Calculon 仿真"""
    with hp.scope(app=model_cfg, exe=execution_cfg) as ps:
        app = Model.from_cfg(ps.app)
        exe = Execution(ps.exe)
        syst = System(system_cfg)
        
        logger = logging.getLogger("calculon")
        model = Llm(app, logger)
        model.compile(syst, exe)
        model.run(syst)

        return model.get_stats_json(False)


with st.expander("🧪 IR vs Calculon 路径对比", expanded=True):
    model_dir = DATA_DIR / "models"
    system_dir = DATA_DIR / "systems"
    exec_dir = DATA_DIR / "examples"

    model_options = sorted([p.stem for p in model_dir.glob("*.json")])
    system_options = sorted([p.stem for p in system_dir.glob("*.json")])
    exec_options = sorted([p.stem for p in exec_dir.glob("*.json")])

    col1, col2, col3 = st.columns(3)
    with col1:
        model_name = st.selectbox("模型配置", model_options, index=model_options.index("gpt3-175B") if "gpt3-175B" in model_options else 0, key="val_model_select")
    with col2:
        system_name = st.selectbox("系统配置", system_options, index=system_options.index("h100_80g_nvl8") if "h100_80g_nvl8" in system_options else 0, key="val_system_select")
    with col3:
        exec_name = st.selectbox("执行配置", exec_options, index=exec_options.index("3072_t4_p64_d12_mbs4_full") if "3072_t4_p64_d12_mbs4_full" in exec_options else 0, key="val_exec_select")

    if st.button("🚀 运行对比", type="primary"):
        with st.spinner("加载配置..."):
            cfg = load_configs(model_name, system_name, exec_name)
            model_cfg = cfg["model"]
            system_cfg = cfg["system"]
            execution_cfg = cfg["execution"]

        with st.spinner("运行 IR 编译器..."):
            micro_batch_size = execution_cfg.get("microbatch_size", 1)
            graph = build_transformer_graph(model_cfg, model_name, micro_batch_size)
            compiler, system_params, parallel_params = create_compiler(
                execution_cfg, system_cfg, model_cfg.get("seq_size", 2048),
            )
            with hp.scope(system=system_params, parallel=parallel_params):
                ir_result = compiler.compile(graph)

        with st.spinner("运行 Calculon..."):
            calc_stats = run_calculon(model_cfg, execution_cfg, system_cfg)

        tp = execution_cfg["tensor_par"]
        pp = execution_cfg["pipeline_par"]
        dp = execution_cfg["data_par"]
        num_gpus = tp * pp * dp

        st.markdown(f"**配置**: {model_name} | TP={tp}, PP={pp}, DP={dp} | GPUs={num_gpus}")

        ir_metrics = normalize_ir_metrics(ir_result, num_layers=model_cfg.get("num_blocks", 1), pp=pp)
        calc_metrics = normalize_calculon_stats(calc_stats)

        ir_mem = ir_metrics.get("per_gpu", {}).get("memory", {})
        ir_time = ir_metrics.get("per_gpu", {}).get("time", {})
        ir_block = ir_metrics.get("block", {})

        calc_mem = calc_metrics.get("per_gpu", {}).get("memory", {})
        calc_time = calc_metrics.get("per_gpu", {}).get("time", {})
        calc_block = calc_metrics.get("block", {})

        st.markdown("#### 总内存 (per GPU)")
        total_mem_rows = [
            {"指标": "权重", "IR": format_bytes(ir_mem.get("weights_fp16_bytes", 0) or ir_mem.get("weights_fp32_bytes", 0)),
             "Calculon": format_bytes(calc_mem.get("weights_fp16_bytes", 0) or calc_mem.get("weights_fp32_bytes", 0)),
             "Diff": format_diff(ir_mem.get("weights_fp16_bytes", 0) or ir_mem.get("weights_fp32_bytes", 0),
                                 calc_mem.get("weights_fp16_bytes", 0) or calc_mem.get("weights_fp32_bytes", 0))},
            {"指标": "激活", "IR": format_bytes(ir_mem.get("activations_bytes", 0)),
             "Calculon": format_bytes(calc_mem.get("activations_bytes", 0)),
             "Diff": format_diff(ir_mem.get("activations_bytes", 0), calc_mem.get("activations_bytes", 0))},
            {"指标": "优化器", "IR": format_bytes(ir_mem.get("optimizer_bytes", 0)),
             "Calculon": format_bytes(calc_mem.get("optimizer_bytes", 0)),
             "Diff": format_diff(ir_mem.get("optimizer_bytes", 0), calc_mem.get("optimizer_bytes", 0))},
            {"指标": "合计", "IR": format_bytes(ir_mem.get("total_bytes", 0)),
             "Calculon": format_bytes(calc_mem.get("total_bytes", 0)),
             "Diff": format_diff(ir_mem.get("total_bytes", 0), calc_mem.get("total_bytes", 0))},
        ]
        st.dataframe(pd.DataFrame(total_mem_rows), hide_index=True, use_container_width=True)

        st.markdown("#### 总时间")
        total_time_rows = [
            {"指标": "迭代时间", "IR": format_time(ir_time.get("iteration_time", 0)),
             "Calculon": format_time(calc_time.get("iteration_time", 0)),
             "Diff": format_diff(ir_time.get("iteration_time", 0), calc_time.get("iteration_time", 0))},
            {"指标": "前向", "IR": format_time(ir_time.get("forward_time", 0)),
             "Calculon": "-", "Diff": "-"},
            {"指标": "反向", "IR": format_time(ir_time.get("backward_time", 0)),
             "Calculon": "-", "Diff": "-"},
            {"指标": "通信", "IR": format_time(ir_time.get("communication_time", 0)),
             "Calculon": "-", "Diff": "-"},
            {"指标": "Bubble", "IR": format_time(ir_time.get("bubble_time", 0)),
             "Calculon": "-", "Diff": "-"},
            {"指标": "优化器", "IR": format_time(ir_time.get("optimizer_time", 0)),
             "Calculon": "-", "Diff": "-"},
        ]
        st.dataframe(pd.DataFrame(total_time_rows), hide_index=True, use_container_width=True)

        st.markdown("#### 单层内存")
        block_mem_rows = [
            {"指标": "权重", "IR": format_bytes(ir_block.get("memory", {}).get("weights_bytes", 0)),
             "Calculon": format_bytes(calc_block.get("memory", {}).get("weights_bytes", 0)),
             "Diff": format_diff(ir_block.get("memory", {}).get("weights_bytes", 0),
                                 calc_block.get("memory", {}).get("weights_bytes", 0))},
            {"指标": "激活", "IR": format_bytes(ir_block.get("memory", {}).get("activations_bytes", 0)),
             "Calculon": format_bytes(calc_block.get("memory", {}).get("activations_bytes", 0)),
             "Diff": format_diff(ir_block.get("memory", {}).get("activations_bytes", 0),
                                 calc_block.get("memory", {}).get("activations_bytes", 0))},
            {"指标": "优化器", "IR": format_bytes(ir_block.get("memory", {}).get("optimizer_bytes", 0)),
             "Calculon": format_bytes(calc_block.get("memory", {}).get("optimizer_bytes", 0)),
             "Diff": format_diff(ir_block.get("memory", {}).get("optimizer_bytes", 0),
                                 calc_block.get("memory", {}).get("optimizer_bytes", 0))},
        ]
        st.dataframe(pd.DataFrame(block_mem_rows), hide_index=True, use_container_width=True)

        st.markdown("#### 单层时间")
        block_time = ir_block.get("time", {})
        calc_time_block = calc_block.get("time", {})
        ir_comm_fw = block_time.get("comm_fw", 0)
        ir_comm_bw = block_time.get("comm_bw", 0)
        block_time_rows = [
            {"指标": "前向", "IR": format_time(block_time.get("forward_time", 0)),
             "Calculon": format_time(calc_time_block.get("forward_time", 0)),
             "Diff": format_diff(block_time.get("forward_time", 0), calc_time_block.get("forward_time", 0))},
            {"指标": "激活梯度", "IR": format_time(block_time.get("agrad_time", 0)),
             "Calculon": format_time(calc_time_block.get("agrad_time", 0)),
             "Diff": format_diff(block_time.get("agrad_time", 0), calc_time_block.get("agrad_time", 0))},
            {"指标": "权重梯度", "IR": format_time(block_time.get("wgrad_time", 0)),
             "Calculon": format_time(calc_time_block.get("wgrad_time", 0)),
             "Diff": format_diff(block_time.get("wgrad_time", 0), calc_time_block.get("wgrad_time", 0))},
            {"指标": "计算合计", "IR": format_time(block_time.get("compute_time", 0)),
             "Calculon": format_time(calc_time_block.get("compute_time", 0)),
             "Diff": format_diff(block_time.get("compute_time", 0), calc_time_block.get("compute_time", 0))},
            {"指标": "TP通信(FW)", "IR": format_time(ir_comm_fw), "Calculon": format_time(calc_time_block.get("tp_comm_fw", 0)),
             "Diff": format_diff(ir_comm_fw, calc_time_block.get("tp_comm_fw", 0)) if ir_comm_fw else "-"},
            {"指标": "TP通信(BW)", "IR": format_time(ir_comm_bw), "Calculon": format_time(calc_time_block.get("tp_comm_bw", 0)),
             "Diff": format_diff(ir_comm_bw, calc_time_block.get("tp_comm_bw", 0)) if ir_comm_bw else "-"},
            {"指标": "通信合计", "IR": format_time(block_time.get("comm_time", 0)),
             "Calculon": format_time(calc_time_block.get("comm_time", 0)),
             "Diff": format_diff(block_time.get("comm_time", 0), calc_time_block.get("comm_time", 0))},
            {"指标": "单层总计", "IR": format_time(block_time.get("total_time", 0)),
             "Calculon": format_time(calc_time_block.get("total_time", 0)),
             "Diff": format_diff(block_time.get("total_time", 0), calc_time_block.get("total_time", 0))},
        ]
        st.dataframe(pd.DataFrame(block_time_rows), hide_index=True, use_container_width=True)


# ============================================================================
# SeqSel Figure 1 验证
# ============================================================================
with st.expander("📈 SeqSel Figure 1 - 基准吞吐量对比", expanded=True):
    st.markdown("**验证目标**: 基准配置下的吞吐量预测准确性")
    df_fig1 = seqsel_fig1(show=True)
    st.dataframe(df_fig1, hide_index=True, use_container_width=True)


# ============================================================================
# SeqSel Figure 7 验证
# ============================================================================
with st.expander("📊 SeqSel Figure 7 - 不同配置性能分析", expanded=True):
    st.markdown("**验证目标**: 不同并行配置下的性能预测")
    df_fig7 = seqsel_fig7(show=True)
    st.dataframe(df_fig7, hide_index=True, use_container_width=True)


# ============================================================================
# SeqSel Table 5 验证
# ============================================================================
with st.expander("📋 SeqSel Table 5 - 详细配置参数对比", expanded=True):
    st.markdown("**验证目标**: 详细配置参数的预测准确性")
    df_tab5 = seqsel_tab5(show=True)
    st.dataframe(df_tab5, hide_index=True, use_container_width=True)

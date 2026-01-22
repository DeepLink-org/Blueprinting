"""LLM 训练计算器 - IR 变换过程可视化页面"""

import html
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
import hyperparameter as hp

from blueprinting.ui import setup_page, page_title, section_header, info_card
from blueprinting.ir import (
    GraphIR,
    ScheduleIR,
    TimelineIR,
    SimulationResult,
    IRBuilder,
    Compiler,
    WorkloadPass,
    ParallelPass,
    SchedulePass,
    TimelinePass,
    OverlapAnalysisPass,
    EvaluatePass,
    OptimizerPass,
    OptimizerConfig,
)
from blueprinting import Model, Execution
from calculon import System
from calculon.llm import Llm


# ============================================================================
# 页面初始化
# ============================================================================
setup_page(title="LLM训练计算器 - IR 变换可视化")

page_title(
    "IR 变换过程可视化",
    subtitle="逐步渲染每个 Pass 的 IR 输出，并对比最终结果",
    icon="🧭",
)

info_card(
    "使用说明",
    "选择模型/系统/执行配置后运行，可查看每个编译 Pass 的 IR 快照，并在最后对比 Calculon 结果。",
    icon="🧩",
)


# ============================================================================
# 配置加载 & 工具函数
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


def _safe_value(val: Any) -> Any:
    if val is None:
        return "-"
    if isinstance(val, (int, float)):
        return val
    return str(val)


def _limit_df(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if df.empty:
        return df
    return df.head(max_rows).copy()


def _tree_to_html(tree_text: str) -> str:
    escaped = html.escape(tree_text or "")
    return f"<pre style='font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; line-height: 1.35;'>{escaped}</pre>"


def _schedule_tree_html(schedule: ScheduleIR, max_ops: int = 5) -> str:
    """Render ScheduleIR hierarchy as HTML using the structure, not text."""
    lines = [
        "<div style='font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px;'>",
        "<ul style='margin:0; padding-left:16px;'>",
        f"<li>ScheduleIR (stages={schedule.num_stages}, devices={schedule.num_devices})",
        "<ul style='margin:0; padding-left:16px;'>",
    ]

    for stage in schedule.iter_stages():
        stage_ops = list(stage.iter_ops())
        lines.append(
            f"<li>Stage {stage.stage_id} "
            f"<span style='color:#6b7280'>(ops={len(stage_ops)})</span>"
            "<ul style='margin:0; padding-left:16px;'>"
        )
        for device in stage.iter_devices():
            dev_label = (
                f"Device {device.device_id} "
                f"<span style='color:#6b7280'>(ops={len(device.ops)})</span>"
            )

            preview_parts = []
            for op in device.ops[:max_ops]:
                name = html.escape(op.name or "")
                op_type = html.escape(op.op_type or "")
                preview_parts.append(
                    f"{op_type}({name}) "
                    f"<span style='color:#6b7280'>(seq={op.event_seq})</span>"
                )
            preview = " · ".join(preview_parts)

            lines.append("<li>")
            lines.append("<details>")
            lines.append(
                "<summary>"
                f"{dev_label}"
                f"{' <span style=\"color:#9ca3af\">—</span> ' + preview if preview else ''}"
                "</summary>"
            )
            lines.append("<ul style='margin:0; padding-left:16px;'>")

            for op in device.ops[:max_ops]:
                name = html.escape(op.name or "")
                op_type = html.escape(op.op_type or "")
                lines.append(
                    f"<li>{op_type}({name}) "
                    f"<span style='color:#6b7280'>(seq={op.event_seq})</span></li>"
                )

            if len(device.ops) > max_ops:
                lines.append("<li>")
                lines.append("<details>")
                lines.append(
                    f"<summary><span style='color:#6b7280'>... ({len(device.ops) - max_ops} more)</span></summary>"
                )
                lines.append("<ul style='margin:0; padding-left:16px;'>")
                for op in device.ops[max_ops:]:
                    name = html.escape(op.name or "")
                    op_type = html.escape(op.op_type or "")
                    lines.append(
                        f"<li>{op_type}({name}) "
                        f"<span style='color:#6b7280'>(seq={op.event_seq})</span></li>"
                    )
                lines.append("</ul></details></li>")

            lines.append("</ul></details></li>")
        lines.append("</ul></li>")

    lines.extend(["</ul></li>", "</ul></div>"])
    return "".join(lines)


def _schedule_summary(schedule: ScheduleIR) -> str:
    stage_count = schedule.num_stages
    devices_per_stage = []
    distinct_devices = set()
    for stage in schedule.iter_stages():
        devices = list(stage.iter_devices())
        devices_per_stage.append(len(devices))
        for device in devices:
            distinct_devices.add(device.device_id)
    max_devices_per_stage = max(devices_per_stage) if devices_per_stage else 0
    distinct_device_count = len(distinct_devices)
    return (
        f"ScheduleIR: {stage_count} stages, "
        f"devices_per_stage={max_devices_per_stage}, "
        f"distinct_devices={distinct_device_count}"
    )


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
                "optimizer_time": ir_tb.optimizer if ir_tb else 0,
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
    builder.set_metadata("pp", execution.get("pipeline_par", 1))
    builder.set_metadata("dp", execution.get("data_par", 1))

    x = builder.add_input("input_ids", shape=[micro_batch_size, seq_len, hidden])
    prev = x

    for i in range(num_layers):
        with builder.block(f"layer{i}", "TransformerLayer", layer_idx=i):
            with builder.block("attn", "Attention"):
                norm = builder.rmsnorm("norm", prev, hidden, batch_seq=batch_seq)
                q = builder.linear("q_proj", norm.name + "_out", hidden, hidden, shard="tp_col", batch_seq=batch_seq)
                k = builder.linear("k_proj", norm.name + "_out", hidden, hidden, shard="tp_col", batch_seq=batch_seq)
                v = builder.linear("v_proj", norm.name + "_out", hidden, hidden, shard="tp_col", batch_seq=batch_seq)
                attn = builder.attention(
                    "mha",
                    [q.name + "_out", k.name + "_out", v.name + "_out"],
                    num_heads, head_dim, seq_len=seq_len, batch_size=micro_batch_size, shard="tp_col"
                )
                out_proj = builder.linear("out_proj", attn.name + "_out", hidden, hidden, shard="tp_row", batch_seq=batch_seq)
                res1 = builder.add("residual", [prev, out_proj.name + "_out"], num_elements=batch_seq * hidden)

            attn_out = res1.name + "_out"

            with builder.block("ffn", "FFN"):
                norm2 = builder.rmsnorm("norm", attn_out, hidden, batch_seq=batch_seq)
                fc1 = builder.linear("fc1", norm2.name + "_out", hidden, feedforward, shard="tp_col", batch_seq=batch_seq)
                act = builder.activation("act", fc1.name + "_out", "SiLU", num_elements=batch_seq * feedforward, shard="tp_col")
                fc2 = builder.linear("fc2", act.name + "_out", feedforward, hidden, shard="tp_row", batch_seq=batch_seq)
                res2 = builder.add("residual", [attn_out, fc2.name + "_out"], num_elements=batch_seq * hidden)

            prev = res2.name + "_out"

    with builder.block("final", "Output"):
        final_norm = builder.rmsnorm("norm", prev, hidden, batch_seq=batch_seq)

    builder.add_output(final_norm.name + "_out")
    return builder.build()


def create_compiler(execution: Dict[str, Any], system_path: Path, seq_len: int) -> Compiler:
    """创建 IR 编译流水线"""
    tp = execution["tensor_par"]
    pp = execution["pipeline_par"]
    dp = execution["data_par"]
    batch_size = execution["batch_size"]
    micro_batch_size = execution["microbatch_size"]
    gradient_checkpointing = execution.get("activation_recompute", "none") != "none"
    num_microbatches = batch_size // (micro_batch_size * dp)

    compiler = Compiler()
    compiler.add_pass(WorkloadPass(dtype_bytes=2))
    compiler.add_pass(ParallelPass(
        tp=tp, pp=pp, dp=dp,
        tp_comm_type=execution.get("tensor_par_comm_type", "ar"),
        sequence_parallel=execution.get("sequence_par", False),
    ))
    compiler.add_pass(SchedulePass(system_config=system_path, training=True))
    compiler.add_pass(OptimizerPass(
        OptimizerConfig(
            optimizer_type="adam",
            master_weights=True,
            gradient_checkpointing=gradient_checkpointing,
            recompute_mode="full" if gradient_checkpointing else "none",
            checkpoint_ratio=1.0 if gradient_checkpointing else 0.0,
            num_microbatches=num_microbatches,
        ),
        system_config={"memory_bandwidth_gbps": 3072, "peak_tflops": 1000},
    ))
    compiler.add_pass(TimelinePass())
    compiler.add_pass(OverlapAnalysisPass())
    compiler.add_pass(EvaluatePass(subs={"batch_seq": micro_batch_size * seq_len}, training=True))
    return compiler


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


def _graph_ops_table(graph: GraphIR, max_rows: int) -> pd.DataFrame:
    rows = []
    for path, op in graph.iter_ops_with_path():
        rows.append({
            "path": path,
            "op": op.op_type,
            "shard": op.shard or "-",
            "flops_fw": _safe_value(op.flops_fw),
            "flops_bw": _safe_value(op.flops_bw),
            "weight_bytes": _safe_value(op.weight_bytes),
            "activation_bytes": _safe_value(op.activation_bytes),
        })
        if len(rows) >= max_rows:
            break
    return pd.DataFrame(rows)


def _schedule_ops_table(schedule: ScheduleIR, max_rows: int) -> pd.DataFrame:
    rows = []
    for op in schedule.iter_ops_sorted():
        rows.append({
            "seq": op.event_seq,
            "device": op.device,
            "stage": op.stage,
            "stream": op.stream,
            "op": op.op_type,
            "name": op.name,
            "duration": _safe_value(op.duration),
        })
        if len(rows) >= max_rows:
            break
    return pd.DataFrame(rows)


def _timeline_device_summary(timeline: TimelineIR, device: int = 0) -> str:
    peak = timeline.peak_memory(device=device)
    makespan = timeline.makespan(device=device)
    compute = timeline.compute_time(device=device)
    comm = timeline.comm_time(device=device)

    def _fmt(val, unit_scale=1.0, unit=""):
        if isinstance(val, (int, float)):
            return f"{val * unit_scale:.2f}{unit}"
        return str(val)

    peak_str = f"{peak/1e9:.2f} GB" if isinstance(peak, (int, float)) else str(peak)
    makespan_str = f"{makespan*1e3:.2f} ms" if isinstance(makespan, (int, float)) else str(makespan)
    compute_str = f"{compute*1e3:.2f} ms" if isinstance(compute, (int, float)) else str(compute)
    comm_str = f"{comm*1e3:.2f} ms" if isinstance(comm, (int, float)) else str(comm)

    lines = [
        f"TimelineIR Summary (device={device})",
        f"  Events: {len(timeline.events)}",
        f"  Devices: {timeline.num_devices}",
        "",
        f"  Peak Memory: {peak_str}",
        f"  Makespan: {makespan_str}",
        f"  Compute: {compute_str}",
        f"  Comm: {comm_str}",
    ]
    return "\n".join(lines)


def _timeline_events_table(timeline: TimelineIR, max_rows: int, device: Optional[int] = None) -> pd.DataFrame:
    rows = []
    events = timeline.get_events(device=device)
    for seq, event in enumerate(events):
        rows.append({
            "event_seq": seq,
            "time": _safe_value(event.time),
            "type": event.event_type.value,
            "device_id": event.device,
            "stream": event.stream.value,
            "resource": event.resource_id,
            "size": _safe_value(event.size),
            "op_type": event.op_type or "-",
        })
        if len(rows) >= max_rows:
            break
    return pd.DataFrame(rows)


def snapshot_ir(
    name: str,
    ir: Any,
    max_rows: int = 120,
    timeline_device: int = 0,
    timeline_single_device: bool = True,
) -> Dict[str, Any]:
    """Create a render-friendly snapshot for an IR object."""
    snap: Dict[str, Any] = {"name": name, "ir_type": type(ir).__name__}

    if isinstance(ir, GraphIR):
        snap["summary"] = ir.summary()
        snap["tree"] = ir.tree(max_depth=3)
        snap["table"] = _limit_df(_graph_ops_table(ir, max_rows=max_rows), max_rows)
        snap["table_title"] = "Op 列表 (GraphIR)"
    elif isinstance(ir, ScheduleIR):
        snap["summary"] = _schedule_summary(ir)
        snap["tree"] = ir.tree(max_ops=4)
        snap["tree_html"] = _schedule_tree_html(ir, max_ops=4)
        snap["table"] = _limit_df(_schedule_ops_table(ir, max_rows=max_rows), max_rows)
        snap["table_title"] = "调度 Op 列表 (ScheduleIR)"
    elif isinstance(ir, TimelineIR):
        if timeline_single_device:
            snap["summary"] = _timeline_device_summary(ir, device=timeline_device)
        else:
            snap["summary"] = ir.summary()
        snap["tree"] = None
        table_device = timeline_device if timeline_single_device else None
        snap["table"] = _limit_df(_timeline_events_table(ir, max_rows=max_rows, device=table_device), max_rows)
        snap["table_title"] = "事件列表 (TimelineIR)"
        snap["metadata"] = ir.metadata
    elif isinstance(ir, SimulationResult):
        snap["summary"] = repr(ir)
        snap["tree"] = None
        snap["table"] = pd.DataFrame([ir.to_dict()])
        snap["table_title"] = "结果汇总 (SimulationResult)"
    else:
        snap["summary"] = str(ir)
        snap["tree"] = None

    return snap


def run_ir_pipeline_with_snapshots(
    graph: GraphIR,
    compiler: Compiler,
    max_rows: int = 120,
    timeline_device: int = 0,
    timeline_single_device: bool = True,
) -> Dict[str, Any]:
    """运行 IR 编译流程并返回各 Pass 的快照。"""
    snapshots: List[Dict[str, Any]] = []
    snapshots.append(snapshot_ir("InputGraph", graph, max_rows=max_rows))

    current: Any = graph
    for p in compiler.passes:
        current = p.run(current)
        snapshots.append(
            snapshot_ir(
                p.name,
                current,
                max_rows=max_rows,
                timeline_device=timeline_device,
                timeline_single_device=timeline_single_device,
            )
        )

    return {"final": current, "snapshots": snapshots}


# ============================================================================
# 页面主体
# ============================================================================
with st.expander("🧪 IR 变换路径可视化", expanded=True):
    model_dir = DATA_DIR / "models"
    system_dir = DATA_DIR / "systems"
    exec_dir = DATA_DIR / "examples"

    model_options = sorted([p.stem for p in model_dir.glob("*.json")])
    system_options = sorted([p.stem for p in system_dir.glob("*.json")])
    exec_options = sorted([p.stem for p in exec_dir.glob("*.json")])

    col1, col2, col3 = st.columns(3)
    with col1:
        model_name = st.selectbox(
            "模型配置",
            model_options,
            index=model_options.index("gpt3-175B") if "gpt3-175B" in model_options else 0,
        )
    with col2:
        system_name = st.selectbox(
            "系统配置",
            system_options,
            index=system_options.index("h100_80g_nvl8") if "h100_80g_nvl8" in system_options else 0,
        )
    with col3:
        exec_name = st.selectbox(
            "执行配置",
            exec_options,
            index=exec_options.index("3072_t4_p64_d12_mbs4_full") if "3072_t4_p64_d12_mbs4_full" in exec_options else 0,
        )

    show_raw_metadata = st.checkbox("显示 IR 元数据", value=False)
    max_rows = st.slider("每个表格最多显示行数", min_value=20, max_value=300, value=120, step=20)
    timeline_single_device = st.checkbox("Timeline 仅显示单卡", value=True)
    timeline_device = st.number_input("Timeline 设备编号", min_value=0, value=0, step=1)

    run_btn = st.button("🚀 运行可视化", type="primary")

if run_btn:
    with st.spinner("加载配置..."):
        cfg = load_configs(model_name, system_name, exec_name)
        model_cfg = cfg["model"]
        system_cfg = cfg["system"]
        execution_cfg = cfg["execution"]

    with st.spinner("构建 GraphIR..."):
        graph = build_transformer_graph(model_cfg, execution_cfg, model_name)

    with st.spinner("运行 IR 编译流程..."):
        compiler = create_compiler(execution_cfg, cfg["system_path"], model_cfg.get("seq_size", 2048))
        pipeline_result = run_ir_pipeline_with_snapshots(
            graph,
            compiler,
            max_rows=max_rows,
            timeline_device=int(timeline_device),
            timeline_single_device=timeline_single_device,
        )
        ir_result = pipeline_result["final"]
        snapshots = pipeline_result["snapshots"]

    with st.spinner("运行 Calculon..."):
        calc_stats = run_calculon(model_cfg, execution_cfg, system_cfg)

    st.session_state["ir_snapshots"] = snapshots
    st.session_state["ir_result"] = ir_result
    st.session_state["calc_stats"] = calc_stats
    st.session_state["cfg"] = cfg
    st.session_state["execution_cfg"] = execution_cfg


snapshots: Optional[List[Dict[str, Any]]] = st.session_state.get("ir_snapshots")
ir_result: Optional[SimulationResult] = st.session_state.get("ir_result")
calc_stats: Optional[Dict[str, Any]] = st.session_state.get("calc_stats")
cfg: Optional[Dict[str, Any]] = st.session_state.get("cfg")
execution_cfg: Optional[Dict[str, Any]] = st.session_state.get("execution_cfg")

if snapshots:
    section_header("IR Pass 输出快照", "按 Pass 顺序展示 IR 的变换过程")

    for idx, snap in enumerate(snapshots, start=1):
        title = f"{idx}. {snap['name']} → {snap['ir_type']}"
        with st.expander(title, expanded=False):
            st.markdown("**概要**")
            st.code(snap.get("summary") or "-", language="text")

            if snap.get("tree_html"):
                st.markdown("**结构树**")
                st.markdown(snap["tree_html"], unsafe_allow_html=True)
            elif snap.get("tree"):
                st.markdown("**结构树**")
                st.code(snap["tree"], language="text")

            table = snap.get("table")
            if table is not None and not table.empty:
                st.markdown(f"**{snap.get('table_title', '详情')}**")
                st.dataframe(table, hide_index=True, use_container_width=True)

            if show_raw_metadata and snap.get("metadata"):
                st.markdown("**元数据**")
                st.json(snap["metadata"])

    if ir_result and calc_stats and cfg and execution_cfg:
        section_header("IR vs Calculon 最终对比", "指标口径与 validation 页面保持一致")

        model_cfg = cfg["model"]
        tp = execution_cfg["tensor_par"]
        pp = execution_cfg["pipeline_par"]
        dp = execution_cfg["data_par"]
        num_gpus = tp * pp * dp

        st.markdown(f"**配置**: {cfg['model_name']} | TP={tp}, PP={pp}, DP={dp} | GPUs={num_gpus}")

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

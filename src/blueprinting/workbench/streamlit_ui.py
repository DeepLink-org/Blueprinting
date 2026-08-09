"""Reusable Streamlit components backed only by the Blueprinting service."""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from typing import Any

import pandas as pd
import streamlit as st

from blueprinting.analysis import CalibrationMode
from blueprinting.application import (
    AnalysisDiagnostic,
    AnalysisDraft,
    AnalysisOutcome,
    BlueprintingService,
    DiagnosticLevel,
    SweepReport,
    SweepRequest,
)

from .catalog import ConfigCatalog, default_catalog

_COMPILER_DTYPES = ("float16", "bfloat16", "float32", "float8")
_PARALLEL_OPTIONS = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128)


def _stretch(widget: Any) -> dict[str, Any]:
    """Use the current width API while retaining Streamlit 1.40 compatibility."""

    if "width" in inspect.signature(widget).parameters:
        return {"width": "stretch"}
    return {"use_container_width": True}


@dataclass(frozen=True)
class _PresetSelection:
    catalog: ConfigCatalog
    model_name: str
    execution_name: str
    hardware_name: str
    model_data: dict[str, Any]
    execution_data: dict[str, Any]
    hardware_data: dict[str, Any]

    @property
    def namespace(self) -> str:
        return f"{self.model_name}:{self.execution_name}:{self.hardware_name}"


def setup_workbench_page(title: str, icon: str) -> None:
    st.set_page_config(page_title=f"Blueprinting · {title}", page_icon=icon, layout="wide")
    st.title(f"{icon} {title}")


def _preferred_index(names: tuple[str, ...], preferred: str) -> int:
    return names.index(preferred) if preferred in names else 0


def _preset_selection(key_prefix: str) -> _PresetSelection:
    catalog = default_catalog()
    model_names = catalog.names("models")
    execution_names = catalog.names("examples")
    hardware_names = catalog.names("systems")
    with st.sidebar:
        st.header("分析输入")
        model_name = st.selectbox(
            "模型预设",
            model_names,
            index=_preferred_index(model_names, "gpt3-175B.json"),
            key=f"{key_prefix}.preset.model",
        )
        hardware_name = st.selectbox(
            "硬件证据",
            hardware_names,
            index=_preferred_index(hardware_names, "a100_80g.json"),
            key=f"{key_prefix}.preset.hardware",
        )
        execution_name = st.selectbox(
            "策略模板",
            execution_names,
            index=0,
            key=f"{key_prefix}.preset.execution",
        )
    return _PresetSelection(
        catalog=catalog,
        model_name=model_name,
        execution_name=execution_name,
        hardware_name=hardware_name,
        model_data=catalog.load("models", model_name),
        execution_data=catalog.load("examples", execution_name),
        hardware_data=catalog.load("systems", hardware_name),
    )


def _model_fields(selection: _PresetSelection, key_prefix: str) -> dict[str, Any]:
    model = selection.model_data
    namespace = f"{key_prefix}.{selection.namespace}.model"
    with st.expander("模型语义", expanded=False):
        hidden = st.number_input("Hidden size", min_value=1, value=int(model["hidden"]), key=f"{namespace}.hidden")
        feedforward = st.number_input(
            "Feed-forward size",
            min_value=1,
            value=int(model["feedforward"]),
            key=f"{namespace}.feedforward",
        )
        sequence = st.number_input(
            "Sequence length",
            min_value=1,
            value=int(model["seq_size"]),
            key=f"{namespace}.sequence",
        )
        heads = st.number_input(
            "Attention heads",
            min_value=1,
            value=int(model["attn_heads"]),
            key=f"{namespace}.heads",
        )
        head_size = st.number_input(
            "Attention head size",
            min_value=1,
            value=int(model["attn_size"]),
            key=f"{namespace}.head_size",
        )
        blocks = st.number_input(
            "Transformer blocks",
            min_value=1,
            value=int(model["num_blocks"]),
            key=f"{namespace}.blocks",
        )
    return {
        "hidden": int(hidden),
        "feedforward": int(feedforward),
        "seq_size": int(sequence),
        "attn_heads": int(heads),
        "attn_size": int(head_size),
        "num_blocks": int(blocks),
    }


def _supported_datatypes(selection: _PresetSelection) -> tuple[str, ...]:
    matrix = set(selection.hardware_data.get("matrix", {}))
    vector = set(selection.hardware_data.get("vector", {}))
    supported = tuple(item for item in _COMPILER_DTYPES if item in matrix and item in vector)
    if not supported:
        raise ValueError(f"硬件预设 {selection.hardware_name} 没有同时定义 matrix/vector datatype")
    return supported


def _strategy_fields(
    selection: _PresetSelection,
    key_prefix: str,
    *,
    include_parallelism: bool,
) -> tuple[dict[str, Any], tuple[int, int, int] | None]:
    execution = dict(selection.execution_data)
    namespace = f"{key_prefix}.{selection.namespace}.execution"
    parallelism = None
    if include_parallelism:
        st.markdown("**并行策略**")
        col1, col2, col3 = st.columns(3)
        with col1:
            tp = st.number_input(
                "Tensor parallel",
                min_value=1,
                value=int(execution["tensor_par"]),
                key=f"{namespace}.tp",
            )
        with col2:
            pp = st.number_input(
                "Pipeline parallel",
                min_value=1,
                value=int(execution["pipeline_par"]),
                key=f"{namespace}.pp",
            )
        with col3:
            dp = st.number_input(
                "Data parallel",
                min_value=1,
                value=int(execution["data_par"]),
                key=f"{namespace}.dp",
            )
        parallelism = (int(tp), int(pp), int(dp))
        st.caption(f"World size 由并行度推导：{int(tp) * int(pp) * int(dp):,}")

    st.markdown("**训练负载**")
    col1, col2 = st.columns(2)
    with col1:
        global_batch = st.number_input(
            "Global batch size",
            min_value=1,
            value=int(execution["batch_size"]),
            key=f"{namespace}.global_batch",
        )
    with col2:
        microbatch = st.number_input(
            "Microbatch size",
            min_value=1,
            value=int(execution["microbatch_size"]),
            key=f"{namespace}.microbatch",
        )

    dtypes = _supported_datatypes(selection)
    current_dtype = execution.get("datatype", dtypes[0])
    datatype = st.selectbox(
        "Datatype",
        dtypes,
        index=dtypes.index(current_dtype) if current_dtype in dtypes else 0,
        key=f"{namespace}.datatype",
    )
    recompute_options = ("none", "attn_only", "full")
    recompute = st.selectbox(
        "Activation recompute",
        recompute_options,
        index=recompute_options.index(execution.get("activation_recompute", "none")),
        key=f"{namespace}.recompute",
    )
    communication_options = ("ar", "rs_ag")
    current_communication = execution.get("tensor_par_comm_type", "ar")
    communication = st.selectbox(
        "TP communication",
        communication_options,
        index=communication_options.index(current_communication) if current_communication in communication_options else 0,
        key=f"{namespace}.communication",
    )
    interleaving = st.number_input(
        "Pipeline interleaving",
        min_value=1,
        value=int(execution.get("pipeline_interleaving", 1)),
        key=f"{namespace}.interleaving",
    )
    optimizer_sharding = st.checkbox(
        "Optimizer sharding",
        value=bool(execution.get("optimizer_sharding", False)),
        key=f"{namespace}.optimizer_sharding",
    )

    network_count = len(selection.hardware_data.get("networks", ()))
    network_options = tuple(range(network_count))
    if not network_options:
        raise ValueError(f"硬件预设 {selection.hardware_name} 没有网络层级")
    with st.expander("网络映射", expanded=False):
        tp_network = st.selectbox(
            "TP network tier",
            network_options,
            index=min(int(execution.get("tensor_par_net", 0)), network_count - 1),
            key=f"{namespace}.tp_network",
        )
        pp_network = st.selectbox(
            "PP network tier",
            network_options,
            index=min(int(execution.get("pipeline_par_net", 0)), network_count - 1),
            key=f"{namespace}.pp_network",
        )
        dp_network = st.selectbox(
            "DP network tier",
            network_options,
            index=min(int(execution.get("data_par_net", 0)), network_count - 1),
            key=f"{namespace}.dp_network",
        )

    execution.update(
        {
            "batch_size": int(global_batch),
            "microbatch_size": int(microbatch),
            "datatype": datatype,
            "activation_recompute": recompute,
            "tensor_par_comm_type": communication,
            "pipeline_interleaving": int(interleaving),
            "optimizer_sharding": bool(optimizer_sharding),
            "tensor_par_net": int(tp_network),
            "pipeline_par_net": int(pp_network),
            "data_par_net": int(dp_network),
            "attention_type": "multihead",
            "tensor_par_overlap": "none",
            "data_par_overlap": False,
            "weight_offload": False,
            "activations_offload": False,
            "optimizer_offload": False,
            "training": True,
        }
    )
    if parallelism is not None:
        execution.update(
            {
                "tensor_par": parallelism[0],
                "pipeline_par": parallelism[1],
                "data_par": parallelism[2],
                "num_procs": parallelism[0] * parallelism[1] * parallelism[2],
            }
        )
    return execution, parallelism


def _calibration_field(selection: _PresetSelection, key_prefix: str) -> CalibrationMode:
    namespace = f"{key_prefix}.{selection.namespace}.calibration"
    labels = {
        "系统证据曲线": CalibrationMode.SYSTEM_EVIDENCE,
        "理论峰值基线": CalibrationMode.PEAK_ONLY,
    }
    selected = st.radio(
        "估算证据",
        tuple(labels),
        horizontal=True,
        key=namespace,
    )
    return labels[selected]


def analysis_form(key_prefix: str, submit_label: str = "运行 Blueprinting 分析") -> AnalysisDraft | None:
    selection = _preset_selection(key_prefix)
    with st.sidebar, st.form(f"{key_prefix}.analysis_form"):
        model_data = _model_fields(selection, key_prefix)
        execution_data, _ = _strategy_fields(selection, key_prefix, include_parallelism=True)
        calibration = _calibration_field(selection, key_prefix)
        submitted = st.form_submit_button(submit_label, type="primary", **_stretch(st.form_submit_button))
    if not submitted:
        return None
    return AnalysisDraft.from_mappings(
        model_name=selection.model_name.removesuffix(".json"),
        model_data=model_data,
        execution_name=selection.execution_name,
        execution_data=execution_data,
        hardware_name=selection.hardware_name.removesuffix(".json"),
        hardware_data=selection.hardware_data,
        calibration_mode=calibration,
    )


def sweep_form(key_prefix: str) -> SweepRequest | None:
    selection = _preset_selection(key_prefix)
    execution = selection.execution_data
    with st.sidebar, st.form(f"{key_prefix}.sweep_form"):
        model_data = _model_fields(selection, key_prefix)
        execution_data, _ = _strategy_fields(selection, key_prefix, include_parallelism=False)
        calibration = _calibration_field(selection, key_prefix)
        st.markdown("**候选空间**")
        options = tuple(
            sorted(
                set(
                    _PARALLEL_OPTIONS
                    + (
                        int(execution["tensor_par"]),
                        int(execution["pipeline_par"]),
                        int(execution["data_par"]),
                    )
                )
            )
        )
        tp_values = st.multiselect(
            "Tensor parallel candidates",
            options,
            default=[int(execution["tensor_par"])],
            key=f"{key_prefix}.{selection.namespace}.sweep.tp",
        )
        pp_values = st.multiselect(
            "Pipeline parallel candidates",
            options,
            default=[int(execution["pipeline_par"])],
            key=f"{key_prefix}.{selection.namespace}.sweep.pp",
        )
        dp_values = st.multiselect(
            "Data parallel candidates",
            options,
            default=[int(execution["data_par"])],
            key=f"{key_prefix}.{selection.namespace}.sweep.dp",
        )
        count = len(tp_values) * len(pp_values) * len(dp_values)
        st.caption(f"候选数量：{count} / 128")
        submitted = st.form_submit_button(
            "探索策略空间",
            type="primary",
            **_stretch(st.form_submit_button),
        )
    if not submitted:
        return None
    if not tp_values or not pp_values or not dp_values:
        st.sidebar.error("TP、PP、DP 候选集合不能为空。")
        return None
    base_tp = int(execution["tensor_par"])
    base_pp = int(execution["pipeline_par"])
    base_dp = int(execution["data_par"])
    execution_data.update(
        {
            "tensor_par": base_tp,
            "pipeline_par": base_pp,
            "data_par": base_dp,
            "num_procs": base_tp * base_pp * base_dp,
        }
    )
    draft = AnalysisDraft.from_mappings(
        model_name=selection.model_name.removesuffix(".json"),
        model_data=model_data,
        execution_name=selection.execution_name,
        execution_data=execution_data,
        hardware_name=selection.hardware_name.removesuffix(".json"),
        hardware_data=selection.hardware_data,
        calibration_mode=calibration,
    )
    try:
        return SweepRequest(
            base=draft,
            tensor_parallel=tuple(int(item) for item in tp_values),
            pipeline_parallel=tuple(int(item) for item in pp_values),
            data_parallel=tuple(int(item) for item in dp_values),
        )
    except ValueError as error:
        st.sidebar.error(str(error))
        return None


@st.cache_data(show_spinner=False)
def cached_analysis(draft: AnalysisDraft) -> AnalysisOutcome:
    return BlueprintingService().analyze(draft)


@st.cache_data(show_spinner=False)
def cached_sweep(request: SweepRequest) -> SweepReport:
    return BlueprintingService().sweep(request)


def remember(key: str, value: Any) -> Any:
    st.session_state[key] = value
    return value


def recalled(key: str) -> Any | None:
    return st.session_state.get(key)


def format_seconds(value: float) -> str:
    if value >= 1:
        return f"{value:.3f} s"
    if value >= 1e-3:
        return f"{value * 1e3:.3f} ms"
    if value >= 1e-6:
        return f"{value * 1e6:.3f} µs"
    return f"{value * 1e9:.3f} ns"


def format_bytes(value: int | float) -> str:
    number = float(value)
    units = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")
    for unit in units:
        if abs(number) < 1024 or unit == units[-1]:
            return f"{number:.2f} {unit}"
        number /= 1024
    raise AssertionError("byte formatter did not terminate")


def format_count(value: int | float) -> str:
    number = float(value)
    for scale, suffix in ((1e15, "P"), (1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "K")):
        if abs(number) >= scale:
            return f"{number / scale:.2f}{suffix}"
    return f"{number:.0f}"


def render_diagnostics(diagnostics: tuple[AnalysisDiagnostic, ...]) -> None:
    for diagnostic in diagnostics:
        location = ".".join(diagnostic.path)
        suffix = f" · `{location}`" if location else ""
        message = f"**{diagnostic.code}**{suffix} — {diagnostic.message}"
        if diagnostic.hint:
            message += f"\n\n建议：{diagnostic.hint}"
        if diagnostic.level is DiagnosticLevel.ERROR:
            st.error(message)
        elif diagnostic.level is DiagnosticLevel.WARNING:
            st.warning(message)
        else:
            st.info(message)


def render_analysis_summary(outcome: AnalysisOutcome) -> None:
    render_diagnostics(outcome.diagnostics)
    report = outcome.report
    if report is None:
        st.caption(f"Request digest: `{outcome.request_digest}`")
        return

    if report.feasible:
        st.success("该候选在当前解析式内存模型下可放入单设备容量。")
    else:
        st.warning("该候选完成了分析，但不满足设备内存容量约束。")

    columns = st.columns(5)
    metrics = (
        ("迭代延迟", format_seconds(report.total_seconds)),
        ("Token/s", format_count(report.total_tokens_per_second)),
        ("Token/s/设备", format_count(report.tokens_per_second_per_device)),
        ("单设备内存", format_bytes(report.memory["total"])),
        ("主导项", report.bottleneck.replace("_", " ")),
    )
    for column, (label, value) in zip(columns, metrics):
        column.metric(label, value)

    tab_latency, tab_memory, tab_workload, tab_evidence = st.tabs(
        ["延迟分解", "内存分解", "工作量事实", "证据与边界"]
    )
    with tab_latency:
        latency = pd.DataFrame(
            ((name.replace("_", " "), value) for name, value in report.latency.items()),
            columns=("component", "seconds"),
        ).set_index("component")
        st.bar_chart(latency, horizontal=True)
        st.dataframe(
            latency.reset_index().assign(display=lambda frame: frame["seconds"].map(format_seconds)),
            hide_index=True,
            **_stretch(st.dataframe),
        )
    with tab_memory:
        memory = pd.DataFrame(
            (
                (name.replace("_", " "), value)
                for name, value in report.memory.items()
                if name not in {"total", "capacity"}
            ),
            columns=("component", "bytes"),
        ).set_index("component")
        st.bar_chart(memory, horizontal=True)
        used = report.memory["total"] / report.memory["capacity"]
        st.progress(min(float(used), 1.0), text=f"容量使用率 {used:.1%}")
    with tab_workload:
        workload = report.workload.to_dict()
        rows = [
            {"事实": "Portable tasks", "值": f"{workload['task_count']:,}"},
            {"事实": "Compute tasks", "值": f"{workload['compute_task_count']:,}"},
            {"事实": "Collective tasks", "值": f"{workload['collective_task_count']:,}"},
            {"事实": "Operations", "值": format_count(workload["operations"])},
            {"事实": "Read bytes", "值": format_bytes(workload["read_bytes"])},
            {"事实": "Write bytes", "值": format_bytes(workload["write_bytes"])},
            {"事实": "Message bytes", "值": format_bytes(workload["message_bytes"])},
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, **_stretch(st.dataframe))
    with tab_evidence:
        evidence = report.evidence.to_dict()
        st.json(evidence)
        st.markdown("**当前实现边界**")
        for limitation in report.limitations:
            st.markdown(f"- {limitation}")
        st.caption(
            f"Plan `{report.plan_digest}` · Evidence `{report.evidence_revision}` · "
            f"Request `{report.request_digest}`"
        )


def stage_dataframe(outcome: AnalysisOutcome) -> pd.DataFrame:
    if outcome.report is None:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "stage": stage.stage,
            "label": stage.label,
            "pass": stage.pass_name,
            "schema": stage.schema,
            "nodes": stage.node_count,
            "values/buffers": stage.value_count,
            "lowering_ms": stage.duration_ns / 1e6,
            "valid": stage.valid,
            "digest": stage.digest,
        }
        for stage in outcome.report.stages
    )


def task_dataframe(outcome: AnalysisOutcome) -> pd.DataFrame:
    if outcome.report is None:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "operation": task.operation,
            "phase": task.phase,
            "engine": task.engine,
            "kind": task.kind,
            "ops": task.operations,
            "read_bytes": task.read_bytes,
            "write_bytes": task.write_bytes,
            "message_bytes": task.message_bytes,
            "compute_s": task.compute_seconds,
            "memory_s": task.memory_seconds,
            "network_s": task.network_seconds,
            "total_s": task.total_seconds,
            "dependencies": len(task.dependencies),
            "concurrency_group": task.concurrency_group,
            "task_id": task.task_id,
        }
        for task in outcome.report.tasks
    )


def render_ir_explorer(outcome: AnalysisOutcome) -> None:
    render_diagnostics(outcome.diagnostics)
    report = outcome.report
    if report is None:
        return
    st.subheader("推导边界")
    st.dataframe(stage_dataframe(outcome), hide_index=True, **_stretch(st.dataframe))
    tabs = st.tabs([stage.label for stage in report.stages])
    for tab, stage in zip(tabs, report.stages):
        with tab:
            cols = st.columns(4)
            cols[0].metric("节点", stage.node_count)
            cols[1].metric("值 / Buffer", stage.value_count)
            cols[2].metric("Lowering 耗时", format_seconds(stage.duration_ns / 1e9))
            cols[3].metric("Verifier", "通过" if stage.valid else "失败")
            st.caption(f"`{stage.schema}` · `{stage.digest}`")
            render_diagnostics(stage.diagnostics)
            snapshot = json.loads(stage.snapshot_json)
            with st.expander("Canonical IR snapshot", expanded=False):
                st.json(snapshot)
            st.download_button(
                "下载 snapshot",
                data=json.dumps(snapshot, indent=2, ensure_ascii=False),
                file_name=f"{stage.stage}-{stage.digest[:12]}.json",
                mime="application/json",
                key=f"download.{stage.digest}",
            )
    st.subheader("Portable task audit")
    st.dataframe(task_dataframe(outcome), hide_index=True, **_stretch(st.dataframe))


def sweep_dataframe(report: SweepReport) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tp": case.tensor_parallel,
            "pp": case.pipeline_parallel,
            "dp": case.data_parallel,
            "world_size": case.world_size,
            "status": case.status,
            "feasible": case.feasible,
            "pareto": case.pareto,
            "latency_s": case.total_seconds,
            "memory_gib": case.memory_bytes / 1024**3 if case.memory_bytes is not None else None,
            "tokens_s_device": case.tokens_per_second_per_device,
            "bottleneck": case.bottleneck,
            "diagnostic": "; ".join(item.message for item in case.diagnostics),
            "request_digest": case.request_digest,
        }
        for case in report.cases
    )


def render_sweep(report: SweepReport) -> None:
    data = sweep_dataframe(report)
    columns = st.columns(4)
    columns[0].metric("候选", len(report.cases))
    columns[1].metric("成功", report.succeeded_count)
    columns[2].metric("可行", report.feasible_count)
    columns[3].metric("Pareto", int(data["pareto"].sum()))

    successful = data[(data["status"] == "success") & data["latency_s"].notna()]
    if not successful.empty:
        st.subheader("延迟—内存空间")
        st.scatter_chart(
            successful,
            x="memory_gib",
            y="latency_s",
            color="pareto",
            size="world_size",
        )
    st.subheader("全部候选")
    st.dataframe(
        data.sort_values(["pareto", "feasible", "latency_s"], ascending=[False, False, True]),
        hide_index=True,
        **_stretch(st.dataframe),
    )
    invalid = data[data["status"] != "success"]
    if not invalid.empty:
        with st.expander(f"无效候选诊断（{len(invalid)}）", expanded=False):
            st.dataframe(
                invalid[["tp", "pp", "dp", "diagnostic", "request_digest"]],
                hide_index=True,
                **_stretch(st.dataframe),
            )
    st.caption(f"Sweep request `{report.request_digest}`")

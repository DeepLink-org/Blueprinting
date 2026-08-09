"""NiceGUI architecture-exploration workbench.

The UI is a stateful, presentation-only client around :class:`BlueprintingService`.
It exposes one case through a point lens and a case set through a batch lens,
while reusing one configuration surface and immutable evaluation results.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from enum import Enum
from functools import partial
from typing import Any

from nicegui import run, ui

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
from .chrome_trace import perfetto_open_javascript, portable_projection_trace_json
from .evidence_lab import EvidenceLabPanel
from .float_analysis import FloatAnalysisPanel
from .nicegui_theme import METRIC_COLORS, WORKBENCH_CSS
from .presentation import (
    analysis_metrics,
    dependency_timeline_chart_options,
    format_bytes,
    format_count,
    format_seconds,
    latency_chart_options,
    memory_chart_options,
    sweep_chart_options,
    sweep_distribution_chart_options,
    sweep_rows,
    task_rows,
    time_breakdown_chart_options,
    time_breakdown_rows,
    timeline_summary,
)

_COMPILER_DTYPES = ("float16", "bfloat16", "float32", "float8")
_PARALLEL_OPTIONS = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128)
_CALIBRATION_LABELS = {
    "系统证据曲线": CalibrationMode.SYSTEM_EVIDENCE,
    "理论峰值基线": CalibrationMode.PEAK_ONLY,
}
_BATCH_STATUS_LABELS = {
    "all": "全部状态",
    "feasible": "容量可行",
    "infeasible": "容量不可行",
    "pareto": "非支配",
    "failed": "推导失败",
}
_BOTTLENECK_LABELS = {
    "forward": "前向计算",
    "backward": "反向计算",
    "optimizer": "优化器更新",
    "recompute": "激活重计算",
    "tensor_parallel": "张量并行通信",
    "pipeline_parallel": "流水并行通信",
    "data_parallel": "数据并行通信",
    "recommunication": "重通信",
    "pipeline_bubble": "流水空泡",
}


class WorkbenchMode(Enum):
    ANALYSIS = "analysis"
    SWEEP = "sweep"
    EVIDENCE = "evidence"
    FLOAT = "float"


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or value is None or int(value) <= 0:
        raise ValueError(f"{label} 必须是正整数")
    return int(value)


def _strip_json_suffix(name: str) -> str:
    return name.removesuffix(".json")


class ConfigurationPanel:
    """One movable configuration surface shared by both workbench modes."""

    def __init__(
        self,
        catalog: ConfigCatalog,
        *,
        on_change: Callable[[str], None],
    ) -> None:
        self.catalog = catalog
        self.on_change = on_change
        self._controls: list[Any] = []
        self.root: Any | None = None

    def build(self) -> None:
        model_names = self.catalog.names("models")
        execution_names = self.catalog.names("examples")
        hardware_names = self.catalog.names("systems")
        model_name = self.catalog.preferred("models", "gpt3-175B.json")
        execution_name = self.catalog.preferred("examples", execution_names[0])
        hardware_name = self.catalog.preferred("systems", "a100_80g.json")
        model = self.catalog.load("models", model_name)
        execution = self.catalog.load("examples", execution_name)
        hardware = self.catalog.load("systems", hardware_name)

        self.root = ui.column().classes("bp-config-root")
        with self.root:
            with ui.element("section").classes("bp-config-section"):
                self._section_heading("01", "工作负载", "选择模型基线；只在需要时展开并修改语义维度。")
                self.model_preset = self._select(
                    "模型预设",
                    model_names,
                    model_name,
                    self._load_model_preset,
                )
                with (
                    ui.expansion("模型语义参数", icon="schema", value=False).classes("bp-advanced mt-2"),
                    ui.element("div").classes("bp-form-grid pt-2"),
                ):
                    self.hidden = self._number("隐藏维度", model["hidden"])
                    self.feedforward = self._number("前馈维度", model["feedforward"])
                    self.sequence = self._number("序列长度", model["seq_size"])
                    self.heads = self._number("注意力头数", model["attn_heads"])
                    self.head_size = self._number("单头维度", model["attn_size"])
                    self.blocks = self._number("Transformer 层数", model["num_blocks"])

            with ui.element("section").classes("bp-config-section"):
                self._section_heading("02", "映射与目标", "定义执行策略、硬件证据目标和并行拓扑。")
                with ui.element("div").classes("bp-form-grid"):
                    self.execution_preset = self._select(
                        "策略模板",
                        execution_names,
                        execution_name,
                        self._load_execution_preset,
                    )
                    self.hardware_preset = self._select(
                        "硬件证据",
                        hardware_names,
                        hardware_name,
                        self._load_hardware_preset,
                    )
                with ui.element("div").classes("bp-form-grid bp-form-grid--three mt-1"):
                    self.tp = self._number("TP", execution["tensor_par"], on_change=self._notify_shared).mark(
                        "tp-input"
                    )
                    self.pp = self._number("PP", execution["pipeline_par"], on_change=self._notify_shared)
                    self.dp = self._number("DP", execution["data_par"], on_change=self._notify_shared)
                with ui.row().classes("bp-derived-row items-center justify-between"):
                    ui.label("派生设备规模")
                    self.world_size_label = ui.label().classes("bp-mono")
                with ui.element("div").classes("bp-form-grid bp-form-grid--three mt-1"):
                    self.global_batch = self._number("全局批量", execution["batch_size"])
                    self.microbatch = self._number("微批量", execution["microbatch_size"])
                    self.datatype = self._select(
                        "数据类型",
                        self._supported_datatypes(hardware_name),
                        self._initial_datatype(execution, hardware_name),
                    )
                with (
                    ui.expansion("执行策略细节", icon="account_tree", value=False).classes("bp-advanced mt-2"),
                    ui.column().classes("w-full gap-3 pt-2"),
                    ui.element("div").classes("bp-form-grid"),
                ):
                    self.recompute = self._select(
                        "激活重计算",
                        ("none", "attn_only", "full"),
                        execution.get("activation_recompute", "none"),
                    )
                    self.communication = self._select(
                        "TP 通信",
                        ("ar", "rs_ag"),
                        execution.get("tensor_par_comm_type", "ar"),
                    )
                    self.interleaving = self._number(
                        "流水交错数",
                        execution.get("pipeline_interleaving", 1),
                    )
                    self.optimizer_sharding = ui.checkbox(
                        "优化器分片",
                        value=bool(execution.get("optimizer_sharding", False)),
                        on_change=self._notify_shared,
                    ).props("dense")
                    self._controls.append(self.optimizer_sharding)

            self.sweep_section = ui.element("section").classes("bp-config-section bp-config-section--search")
            with self.sweep_section:
                self._section_heading("03", "CaseSet 组合空间", "组合 TP、PP、DP 候选；最多评估 128 个 Case。")
                options = self._parallel_options(execution)
                with ui.element("div").classes("bp-form-grid"):
                    self.tp_candidates = self._multi_select(
                        "TP 候选",
                        options,
                        [int(execution["tensor_par"])],
                    )
                    self.pp_candidates = self._multi_select(
                        "PP 候选",
                        options,
                        [int(execution["pipeline_par"])],
                    )
                    self.dp_candidates = self._multi_select(
                        "DP 候选",
                        options,
                        [int(execution["data_par"])],
                    )
                with ui.row().classes("bp-derived-row items-center justify-between"):
                    ui.label("候选组合")
                    self.candidate_count_label = ui.label().classes("bp-mono")

            with ui.element("section").classes("bp-config-section"):
                self._section_heading("04", "证据与高级映射", "选择估算来源，并按需指定通信网络层级。")
                ui.label("估算证据").classes("bp-section-copy")
                self.calibration = ui.radio(
                    list(_CALIBRATION_LABELS),
                    value="系统证据曲线",
                    on_change=self._notify_shared,
                ).props("dense inline")
                self._controls.append(self.calibration)
                with (
                    ui.expansion("网络映射", icon="hub", value=False).classes("bp-advanced mt-2"),
                    ui.element("div").classes("bp-form-grid bp-form-grid--three pt-2"),
                ):
                    network_options = tuple(range(len(hardware.get("networks", ()))))
                    if not network_options:
                        raise ValueError(f"硬件预设 {hardware_name} 没有网络层级")
                    self.tp_network = self._select(
                        "TP 网络层级",
                        network_options,
                        min(int(execution.get("tensor_par_net", 0)), len(network_options) - 1),
                    )
                    self.pp_network = self._select(
                        "PP 网络层级",
                        network_options,
                        min(int(execution.get("pipeline_par_net", 0)), len(network_options) - 1),
                    )
                    self.dp_network = self._select(
                        "DP 网络层级",
                        network_options,
                        min(int(execution.get("data_par_net", 0)), len(network_options) - 1),
                    )

        self._update_world_size()
        self._update_candidate_count()

    @staticmethod
    def _section_heading(index: str, title: str, copy: str) -> None:
        with ui.row().classes("w-full items-start gap-3 no-wrap mb-3"):
            ui.label(index).classes("bp-section-index")
            with ui.column().classes("gap-1"):
                ui.label(title).classes("bp-section-title")
                ui.label(copy).classes("bp-section-copy")

    def _select(
        self,
        label: str,
        options: tuple[Any, ...],
        value: Any,
        on_change: Callable[..., Any] | None = None,
    ) -> Any:
        control = (
            ui.select(list(options), label=label, value=value, on_change=on_change or self._notify_shared)
            .props("outlined dense options-dense")
            .classes("w-full")
        )
        self._controls.append(control)
        return control

    def _multi_select(self, label: str, options: tuple[int, ...], value: list[int]) -> Any:
        control = (
            ui.select(list(options), label=label, value=value, multiple=True, on_change=self._notify_sweep)
            .props("outlined dense use-chips options-dense")
            .classes("w-full")
        )
        self._controls.append(control)
        return control

    def _number(
        self,
        label: str,
        value: Any,
        *,
        on_change: Callable[..., Any] | None = None,
    ) -> Any:
        control = (
            ui.number(label, value=float(value), min=1, step=1, precision=0, on_change=on_change or self._notify_shared)
            .props("outlined dense")
            .classes("w-full")
        )
        self._controls.append(control)
        return control

    def _selection_names(self) -> tuple[str, str, str]:
        return str(self.model_preset.value), str(self.execution_preset.value), str(self.hardware_preset.value)

    def _supported_datatypes(self, hardware_name: str) -> tuple[str, ...]:
        hardware = self.catalog.load("systems", hardware_name)
        matrix = set(hardware.get("matrix", {}))
        vector = set(hardware.get("vector", {}))
        supported = tuple(item for item in _COMPILER_DTYPES if item in matrix and item in vector)
        if not supported:
            raise ValueError(f"硬件预设 {hardware_name} 没有同时定义 matrix/vector datatype")
        return supported

    def _initial_datatype(self, execution: dict[str, Any], hardware_name: str) -> str:
        supported = self._supported_datatypes(hardware_name)
        current = str(execution.get("datatype", supported[0]))
        return current if current in supported else supported[0]

    @staticmethod
    def _parallel_options(execution: dict[str, Any]) -> tuple[int, ...]:
        return tuple(
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

    def _load_model_preset(self, *_: Any) -> None:
        model = self.catalog.load("models", str(self.model_preset.value))
        for control, value in (
            (self.hidden, model["hidden"]),
            (self.feedforward, model["feedforward"]),
            (self.sequence, model["seq_size"]),
            (self.heads, model["attn_heads"]),
            (self.head_size, model["attn_size"]),
            (self.blocks, model["num_blocks"]),
        ):
            control.set_value(float(value))
        self._notify_shared()

    def _load_hardware_preset(self, *_: Any) -> None:
        hardware_name = str(self.hardware_preset.value)
        hardware = self.catalog.load("systems", hardware_name)
        datatypes = self._supported_datatypes(hardware_name)
        current_datatype = str(self.datatype.value)
        self.datatype.set_options(
            list(datatypes),
            value=current_datatype if current_datatype in datatypes else datatypes[0],
        )
        network_options = tuple(range(len(hardware.get("networks", ()))))
        if not network_options:
            raise ValueError(f"硬件预设 {hardware_name} 没有网络层级")
        for control in (self.tp_network, self.pp_network, self.dp_network):
            current = int(control.value or 0)
            control.set_options(list(network_options), value=min(current, len(network_options) - 1))
        self._notify_shared()

    def _load_execution_preset(self, *_: Any) -> None:
        execution = self.catalog.load("examples", str(self.execution_preset.value))
        for control, value in (
            (self.tp, execution["tensor_par"]),
            (self.pp, execution["pipeline_par"]),
            (self.dp, execution["data_par"]),
            (self.global_batch, execution["batch_size"]),
            (self.microbatch, execution["microbatch_size"]),
            (self.interleaving, execution.get("pipeline_interleaving", 1)),
        ):
            control.set_value(float(value))
        self.recompute.set_value(execution.get("activation_recompute", "none"))
        communication = execution.get("tensor_par_comm_type", "ar")
        self.communication.set_value(communication if communication in {"ar", "rs_ag"} else "ar")
        self.optimizer_sharding.set_value(bool(execution.get("optimizer_sharding", False)))

        supported = self._supported_datatypes(str(self.hardware_preset.value))
        datatype = str(execution.get("datatype", supported[0]))
        self.datatype.set_value(datatype if datatype in supported else supported[0])
        for control, field in (
            (self.tp_network, "tensor_par_net"),
            (self.pp_network, "pipeline_par_net"),
            (self.dp_network, "data_par_net"),
        ):
            options = tuple(control.options)
            value = min(int(execution.get(field, 0)), len(options) - 1)
            control.set_value(value)

        options = self._parallel_options(execution)
        self.tp_candidates.set_options(list(options), value=[int(execution["tensor_par"])])
        self.pp_candidates.set_options(list(options), value=[int(execution["pipeline_par"])])
        self.dp_candidates.set_options(list(options), value=[int(execution["data_par"])])
        self._update_candidate_count()
        self._notify_shared()

    def _notify_shared(self, *_: Any) -> None:
        self._update_world_size()
        self.on_change("shared")

    def _notify_sweep(self, *_: Any) -> None:
        self._update_candidate_count()
        self.on_change("sweep")

    def _update_world_size(self) -> None:
        values = (self.tp.value, self.pp.value, self.dp.value)
        if any(value is None for value in values):
            self.world_size_label.set_text("—")
            return
        self.world_size_label.set_text(f"{int(values[0]) * int(values[1]) * int(values[2]):,} devices")

    def candidate_count(self) -> int:
        return (
            len(self.tp_candidates.value or ())
            * len(self.pp_candidates.value or ())
            * len(self.dp_candidates.value or ())
        )

    def _update_candidate_count(self) -> None:
        count = self.candidate_count()
        self.candidate_count_label.set_text(f"{count} / 128")
        if count > 128 or count <= 0:
            self.candidate_count_label.classes(add="bp-warning", remove="bp-muted")
        else:
            self.candidate_count_label.classes(add="bp-muted", remove="bp-warning")

    def set_mode(self, mode: WorkbenchMode) -> None:
        self.sweep_section.set_visibility(mode is WorkbenchMode.SWEEP)

    def set_model_preset(self, value: str) -> None:
        if str(self.model_preset.value) == value:
            return
        self.model_preset.set_value(value)
        self._load_model_preset()

    def set_hardware_preset(self, value: str) -> None:
        if str(self.hardware_preset.value) == value:
            return
        self.hardware_preset.set_value(value)
        self._load_hardware_preset()

    def set_parallel_value(self, axis: str, value: int) -> None:
        controls = {"tp": self.tp, "pp": self.pp, "dp": self.dp}
        control = controls[axis]
        if int(control.value or 0) == value:
            return
        control.set_value(float(value))
        self._notify_shared()

    def load_point_parallelism(self, tensor_parallel: int, pipeline_parallel: int, data_parallel: int) -> None:
        """Load a Case selected from a CaseSet without mutating the CaseSet definition."""

        for control, value in (
            (self.tp, tensor_parallel),
            (self.pp, pipeline_parallel),
            (self.dp, data_parallel),
        ):
            control.set_value(float(value))
        self._update_world_size()

    def set_parallel_candidates(self, axis: str, values: tuple[int, ...]) -> None:
        controls = {
            "tp": self.tp_candidates,
            "pp": self.pp_candidates,
            "dp": self.dp_candidates,
        }
        control = controls[axis]
        if tuple(int(value) for value in (control.value or ())) == values:
            return
        control.set_value(list(values))
        self._notify_sweep()

    def set_calibration(self, value: str) -> None:
        if value not in _CALIBRATION_LABELS:
            raise ValueError("请选择有效的估算证据模式")
        if str(self.calibration.value) == value:
            return
        self.calibration.set_value(value)
        self._notify_shared()

    def summary(self) -> tuple[tuple[str, str], ...]:
        model_name, execution_name, hardware_name = self._selection_names()
        values = (self.tp.value, self.pp.value, self.dp.value)
        world_size = (
            "—" if any(value is None for value in values) else f"{int(values[0]) * int(values[1]) * int(values[2]):,}"
        )
        return (
            ("模型", _strip_json_suffix(model_name)),
            ("目标", _strip_json_suffix(hardware_name)),
            ("策略", _strip_json_suffix(execution_name)),
            ("并行度", f"TP {int(self.tp.value or 0)} · PP {int(self.pp.value or 0)} · DP {int(self.dp.value or 0)}"),
            ("设备数", world_size),
            ("证据", str(self.calibration.value)),
        )

    def draft(self) -> AnalysisDraft:
        model_name, execution_name, hardware_name = self._selection_names()
        hardware_data = self.catalog.load("systems", hardware_name)
        execution_data = self.catalog.load("examples", execution_name)
        tp = _positive_int(self.tp.value, "TP")
        pp = _positive_int(self.pp.value, "PP")
        dp = _positive_int(self.dp.value, "DP")
        model_data = {
            "hidden": _positive_int(self.hidden.value, "隐藏维度"),
            "feedforward": _positive_int(self.feedforward.value, "前馈维度"),
            "seq_size": _positive_int(self.sequence.value, "序列长度"),
            "attn_heads": _positive_int(self.heads.value, "注意力头数"),
            "attn_size": _positive_int(self.head_size.value, "单头维度"),
            "num_blocks": _positive_int(self.blocks.value, "Transformer 层数"),
        }
        execution_data.update(
            {
                "tensor_par": tp,
                "pipeline_par": pp,
                "data_par": dp,
                "num_procs": tp * pp * dp,
                "batch_size": _positive_int(self.global_batch.value, "全局批量"),
                "microbatch_size": _positive_int(self.microbatch.value, "微批量"),
                "datatype": str(self.datatype.value),
                "activation_recompute": str(self.recompute.value),
                "tensor_par_comm_type": str(self.communication.value),
                "pipeline_interleaving": _positive_int(self.interleaving.value, "流水交错数"),
                "optimizer_sharding": bool(self.optimizer_sharding.value),
                "tensor_par_net": int(self.tp_network.value),
                "pipeline_par_net": int(self.pp_network.value),
                "data_par_net": int(self.dp_network.value),
                "attention_type": "multihead",
                "tensor_par_overlap": "none",
                "data_par_overlap": False,
                "weight_offload": False,
                "activations_offload": False,
                "optimizer_offload": False,
                "training": True,
            }
        )
        calibration = _CALIBRATION_LABELS.get(str(self.calibration.value))
        if calibration is None:
            raise ValueError("请选择有效的估算证据模式")
        return AnalysisDraft.from_mappings(
            model_name=_strip_json_suffix(model_name),
            model_data=model_data,
            execution_name=execution_name,
            execution_data=execution_data,
            hardware_name=_strip_json_suffix(hardware_name),
            hardware_data=hardware_data,
            calibration_mode=calibration,
        )

    def sweep_request(self) -> SweepRequest:
        candidates = {
            "tensor_parallel": tuple(int(value) for value in (self.tp_candidates.value or ())),
            "pipeline_parallel": tuple(int(value) for value in (self.pp_candidates.value or ())),
            "data_parallel": tuple(int(value) for value in (self.dp_candidates.value or ())),
        }
        if any(not values for values in candidates.values()):
            raise ValueError("TP、PP、DP 候选集合不能为空")
        return SweepRequest(base=self.draft(), **candidates)

    def set_busy(self, busy: bool) -> None:
        for control in self._controls:
            control.disable() if busy else control.enable()


class BlueprintingWorkbench:
    """Per-client controller for the blueprint-editor workbench."""

    def __init__(
        self,
        *,
        catalog: ConfigCatalog | None = None,
        service_factory: Callable[[], BlueprintingService] = BlueprintingService,
    ) -> None:
        self.catalog = catalog or default_catalog()
        self.service = service_factory()
        self.mode = WorkbenchMode.ANALYSIS
        self.form: ConfigurationPanel | None = None
        self.analysis_outcome: AnalysisOutcome | None = None
        self.sweep_report: SweepReport | None = None
        self.sweep_base: AnalysisDraft | None = None
        self.analysis_stale = False
        self.sweep_stale = False
        self.busy = False
        self.local_error: str | None = None
        self._progress_completed = 0
        self._progress_total = 0
        self.setup_summary: Any | None = None
        self.stale_host: Any | None = None
        self.loading_progress: Any | None = None
        self.loading_progress_label: Any | None = None
        self.setup_action: Any | None = None
        self.drawer_action: Any | None = None
        self.sidebar_controls: Any | None = None
        self.sidebar_summary: Any | None = None
        self.sidebar_action_host: Any | None = None
        self.quick_controls: list[Any] = []
        self.quick_model: Any | None = None
        self.quick_hardware: Any | None = None
        self.quick_tp: Any | None = None
        self.quick_pp: Any | None = None
        self.quick_dp: Any | None = None
        self.quick_calibration: Any | None = None
        self._syncing_quick_controls = False
        self.batch_results_host: Any | None = None
        self.batch_status_filter: Any | None = None
        self.batch_tp_filter: Any | None = None
        self.batch_pp_filter: Any | None = None
        self.batch_dp_filter: Any | None = None
        self.batch_grid: Any | None = None
        self.evidence_panel: EvidenceLabPanel | None = None
        self.float_panel: FloatAnalysisPanel | None = None

    def build(self) -> None:
        ui.add_css(WORKBENCH_CSS)
        ui.colors(primary="#2563eb", secondary="#2563eb", positive="#15803d", warning="#b45309", negative="#b91c1c")
        ui.dark_mode(False)
        ui.page_title("Blueprinting · 硬件架构工作台")
        self._build_legacy_dialog()

        with ui.left_drawer(value=True, bordered=False) as self.sidebar:
            self.sidebar.props("width=288 breakpoint=980").classes("bp-sidebar")
            with ui.column().classes("bp-sidebar-shell w-full no-wrap"):
                with ui.row().classes("bp-sidebar-brand w-full items-center gap-3 no-wrap"):
                    ui.element("div").classes("bp-brand-mark")
                    with ui.column().classes("gap-0 min-w-0"):
                        ui.label("Blueprinting").classes("bp-brand-title")
                        ui.label("硬件架构探索工作台").classes("bp-brand-subtitle")
                ui.separator().classes("bp-sidebar-rule")
                ui.label("WORKBENCH LENS").classes("bp-sidebar-kicker")
                ui.label("观察尺度").classes("bp-sidebar-title")
                with ui.element("div").classes("bp-mode-switch w-full mt-2").mark("mode-switch") as self.mode_switch:
                    self.analysis_mode_tab = self._build_mode_button(
                        WorkbenchMode.ANALYSIS,
                        "单点剖析",
                        "query_stats",
                        "mode-analysis",
                    )
                    self.sweep_mode_tab = self._build_mode_button(
                        WorkbenchMode.SWEEP,
                        "批量探索",
                        "scatter_plot",
                        "mode-sweep",
                    )
                    self.evidence_mode_tab = self._build_mode_button(
                        WorkbenchMode.EVIDENCE,
                        "性能证据",
                        "speed",
                        "mode-evidence",
                    )
                    self.float_mode_tab = self._build_mode_button(
                        WorkbenchMode.FLOAT,
                        "浮点分析",
                        "calculate",
                        "mode-float",
                    )
                self._mode_buttons = {
                    WorkbenchMode.ANALYSIS: self.analysis_mode_tab,
                    WorkbenchMode.SWEEP: self.sweep_mode_tab,
                    WorkbenchMode.EVIDENCE: self.evidence_mode_tab,
                    WorkbenchMode.FLOAT: self.float_mode_tab,
                }
                self._sync_mode_buttons()
                ui.separator().classes("bp-sidebar-rule")
                self.sidebar_controls = ui.column().classes("w-full gap-2")
                self.sidebar_summary = ui.column().classes("bp-sidebar-summary w-full gap-2")
                self.sidebar_action_host = ui.column().classes("w-full mt-2")
                with ui.column().classes("bp-sidebar-footer w-full gap-2"):
                    with ui.row().classes("items-center gap-2"):
                        ui.element("span").classes("bp-service-dot")
                        ui.label("本地服务可用").classes("bp-sidebar-meta")
                    ui.label("Formal plan · Numerical evidence").classes("bp-sidebar-meta bp-mono")
                    ui.button("Legacy 工具", icon="history", on_click=self.legacy_dialog.open).props(
                        "flat no-caps align=left"
                    ).classes("bp-sidebar-legacy w-full")

        with (
            ui.dialog() as self.config_dialog,
            ui.card().classes("bp-config-dialog").mark("configuration-modal"),
            ui.column().classes("bp-dialog-shell w-full no-wrap"),
        ):
            with ui.row().classes("bp-dialog-heading w-full items-center no-wrap"):
                with ui.column().classes("gap-0"):
                    self.dialog_title = ui.label("Case 完整配置").classes("bp-section-title")
                    self.dialog_copy = ui.label("工作负载、映射与证据细节。 ").classes("bp-section-copy")
                ui.space()
                ui.button(icon="close", on_click=self.config_dialog.close).props("flat round dense")
            self.drawer_form_host = ui.column().classes("bp-dialog-form w-full")
            self.drawer_footer = ui.column().classes("bp-dialog-footer w-full gap-2")

        with ui.element("main").classes("bp-main"):
            with ui.row().classes("bp-mobile-bar w-full items-center gap-3 no-wrap"):
                ui.button(icon="menu", on_click=self._toggle_sidebar).props("flat round dense")
                ui.element("div").classes("bp-brand-mark bp-brand-mark--mobile")
                ui.label("Blueprinting").classes("bp-brand-title")
                ui.space()
                with ui.row().classes("items-center gap-2 no-wrap"):
                    ui.element("span").classes("bp-service-dot")
                    ui.label("本地服务").classes("bp-mobile-service")
            self.workspace = ui.column().classes("w-full gap-4")

        self.progress_timer = ui.timer(0.12, self._poll_progress, active=False, immediate=False)
        self._render_workspace()

    def _toggle_sidebar(self) -> None:
        if hasattr(self, "sidebar"):
            self.sidebar.toggle()

    def _build_legacy_dialog(self) -> None:
        with ui.dialog() as self.legacy_dialog, ui.card().classes("bp-card").style("width: 560px; max-width: 92vw"):
            with ui.row().classes("items-center gap-3"):
                ui.icon("inventory_2", size="26px", color="secondary")
                with ui.column().classes("gap-0"):
                    ui.label("Calculon / Streamlit Legacy").classes("bp-card-title")
                    ui.label("旧工具保持隔离，不参与 Blueprinting 主分析路径。 ").classes("bp-card-copy")
            ui.separator().classes("my-2")
            ui.label("需要旧 Calculon 工具时，请单独启动：").classes("text-sm")
            ui.code("uv run streamlit run streamlit_app.py", language="bash").classes("bp-code")
            with ui.row().classes("w-full justify-end gap-2"):
                ui.link("打开 localhost:8501", "http://127.0.0.1:8501", new_tab=True).classes("text-secondary")
                ui.button("关闭", on_click=self.legacy_dialog.close).props("flat no-caps")

    def _select_mode(self, mode: WorkbenchMode) -> None:
        if self.busy or mode is self.mode:
            return
        self.mode = mode
        self._sync_mode_buttons()
        self.local_error = None
        self.config_dialog.close()
        if self.form is not None and self.mode in {WorkbenchMode.ANALYSIS, WorkbenchMode.SWEEP}:
            self.form.set_mode(self.mode)
        self._render_sidebar_controls()
        self._render_workspace()

    def _build_mode_button(self, mode: WorkbenchMode, label: str, icon: str, marker: str) -> Any:
        button = (
            ui.element("button")
            .props("type=button")
            .classes("bp-mode-button")
            .mark(marker)
            .on("click", lambda _: self._select_mode(mode))
        )
        with button:
            ui.icon(icon).classes("bp-mode-icon")
            ui.label(label).classes("bp-mode-label")
        return button

    def _sync_mode_buttons(self) -> None:
        for mode, button in self._mode_buttons.items():
            if mode is self.mode:
                button.classes(add="bp-mode-button--active")
            else:
                button.classes(remove="bp-mode-button--active")

    def _set_mode_buttons_busy(self, busy: bool) -> None:
        for button in self._mode_buttons.values():
            if busy:
                button.props(add="disabled")
            else:
                button.props(remove="disabled")

    def _ensure_form(self, host: Any) -> None:
        if self.form is None:
            with host:
                self.form = ConfigurationPanel(self.catalog, on_change=self._configuration_changed)
                self.form.build()
        else:
            self.form.root.move(host)
        self.form.set_mode(self.mode)
        self._render_sidebar_controls()
        self._render_sidebar_summary()
        self._render_sidebar_action()

    def _detach_form(self) -> None:
        if self.form is not None and self.form.root is not None:
            self.form.root.move(self.drawer_form_host)

    def _render_sidebar_controls(self) -> None:
        if self.sidebar_controls is None:
            return
        self.sidebar_controls.clear()
        self.quick_controls = []
        self.quick_model = None
        self.quick_hardware = None
        self.quick_tp = None
        self.quick_pp = None
        self.quick_dp = None
        self.quick_calibration = None
        if self.mode is WorkbenchMode.EVIDENCE:
            with self.sidebar_controls:
                ui.label("EVIDENCE CONTROLS").classes("bp-sidebar-kicker")
                with ui.element("section").classes("bp-sidebar-controls-card"):
                    ui.label("只读证据目录").classes("bp-sidebar-title")
                    ui.label("Vidur Phi-2 · A100 · exact selectors").classes("bp-sidebar-meta")
                    ui.label("GEMM primitive 在主视图切换。 ").classes("bp-sidebar-meta")
            return
        if self.mode is WorkbenchMode.FLOAT:
            with self.sidebar_controls:
                ui.label("NUMERIC CONTROLS").classes("bp-sidebar-kicker")
                with ui.element("section").classes("bp-sidebar-controls-card"):
                    ui.label("浮点格式").classes("bp-sidebar-title")
                    ui.label("格式位宽、编码位和观察范围在主视图中交互调整。 ").classes("bp-sidebar-meta")
            return
        if self.form is None:
            return
        with self.sidebar_controls:
            ui.label("CASESET CONTROLS" if self.mode is WorkbenchMode.SWEEP else "CASE CONTROLS").classes(
                "bp-sidebar-kicker"
            )
            with ui.element("section").classes("bp-sidebar-controls-card"):
                with ui.row().classes("w-full items-center justify-between gap-2"):
                    ui.label("空间与过滤" if self.mode is WorkbenchMode.SWEEP else "当前 Case").classes(
                        "bp-sidebar-title"
                    )
                    ui.label("范围" if self.mode is WorkbenchMode.SWEEP else "具体值").classes("bp-sidebar-meta")

                model_options = {name: _strip_json_suffix(name) for name in self.catalog.names("models")}
                hardware_options = {name: _strip_json_suffix(name) for name in self.catalog.names("systems")}
                self.quick_model = (
                    ui.select(
                        model_options,
                        label="模型",
                        value=str(self.form.model_preset.value),
                        on_change=self._quick_model_changed,
                    )
                    .props("filled dense dark options-dense popup-content-class=bp-sidebar-menu")
                    .classes("bp-sidebar-control w-full")
                    .mark("quick-model")
                )
                self.quick_hardware = (
                    ui.select(
                        hardware_options,
                        label="目标",
                        value=str(self.form.hardware_preset.value),
                        on_change=self._quick_hardware_changed,
                    )
                    .props("filled dense dark options-dense popup-content-class=bp-sidebar-menu")
                    .classes("bp-sidebar-control w-full")
                    .mark("quick-hardware")
                )
                self.quick_controls.extend((self.quick_model, self.quick_hardware))

                ui.label("候选集合" if self.mode is WorkbenchMode.SWEEP else "并行拓扑").classes(
                    "bp-sidebar-field-label"
                )
                with ui.element("div").classes("bp-sidebar-parallel-grid"):
                    if self.mode is WorkbenchMode.SWEEP:
                        self.quick_tp = self._quick_candidate_select("TP", "tp", self.form.tp_candidates.value)
                        self.quick_pp = self._quick_candidate_select("PP", "pp", self.form.pp_candidates.value)
                        self.quick_dp = self._quick_candidate_select("DP", "dp", self.form.dp_candidates.value)
                    else:
                        self.quick_tp = self._quick_parallel_number("TP", "tp", self.form.tp.value)
                        self.quick_pp = self._quick_parallel_number("PP", "pp", self.form.pp.value)
                        self.quick_dp = self._quick_parallel_number("DP", "dp", self.form.dp.value)
                self.quick_controls.extend((self.quick_tp, self.quick_pp, self.quick_dp))

                self.quick_calibration = (
                    ui.select(
                        list(_CALIBRATION_LABELS),
                        label="估算证据",
                        value=str(self.form.calibration.value),
                        on_change=self._quick_calibration_changed,
                    )
                    .props("filled dense dark options-dense popup-content-class=bp-sidebar-menu")
                    .classes("bp-sidebar-control w-full")
                    .mark("quick-calibration")
                )
                self.quick_controls.append(self.quick_calibration)
                ui.button(
                    "定义 CaseSet" if self.mode is WorkbenchMode.SWEEP else "完整 Case 配置",
                    icon="tune",
                    on_click=self._open_configuration,
                ).props("outline dense no-caps").classes("bp-sidebar-full-config w-full").mark(
                    "open-full-configuration"
                )
        self._set_quick_controls_busy(self.busy)

    def _quick_parallel_number(self, label: str, axis: str, value: Any) -> Any:
        return (
            ui.number(
                label,
                value=float(value),
                min=1,
                step=1,
                precision=0,
                on_change=partial(self._quick_parallel_changed, axis),
            )
            .props("filled dense dark")
            .classes("bp-sidebar-control")
            .mark(f"quick-{axis}")
        )

    def _quick_candidate_select(self, label: str, axis: str, values: Any) -> Any:
        return (
            ui.select(
                list(_PARALLEL_OPTIONS),
                label=label,
                value=[int(value) for value in (values or ())],
                multiple=True,
                on_change=partial(self._quick_candidates_changed, axis),
            )
            .props("filled dense dark options-dense popup-content-class=bp-sidebar-menu")
            .classes("bp-sidebar-control")
            .mark(f"quick-{axis}-candidates")
        )

    def _quick_model_changed(self, event: Any) -> None:
        if self._syncing_quick_controls or self.form is None or event.value is None:
            return
        self.form.set_model_preset(str(event.value))

    def _quick_hardware_changed(self, event: Any) -> None:
        if self._syncing_quick_controls or self.form is None or event.value is None:
            return
        self.form.set_hardware_preset(str(event.value))

    def _quick_parallel_changed(self, axis: str, event: Any) -> None:
        if self._syncing_quick_controls or self.form is None or event.value is None:
            return
        self.form.set_parallel_value(axis, _positive_int(event.value, axis.upper()))

    def _quick_candidates_changed(self, axis: str, event: Any) -> None:
        if self._syncing_quick_controls or self.form is None:
            return
        values = tuple(int(value) for value in (event.value or ()))
        self.form.set_parallel_candidates(axis, values)

    def _quick_calibration_changed(self, event: Any) -> None:
        if self._syncing_quick_controls or self.form is None or event.value is None:
            return
        self.form.set_calibration(str(event.value))

    def _sync_sidebar_controls(self) -> None:
        if self.form is None or self.quick_model is None:
            return
        self._syncing_quick_controls = True
        try:
            self.quick_model.set_value(str(self.form.model_preset.value))
            self.quick_hardware.set_value(str(self.form.hardware_preset.value))
            if self.mode is WorkbenchMode.SWEEP:
                self.quick_tp.set_value(list(self.form.tp_candidates.value or ()))
                self.quick_pp.set_value(list(self.form.pp_candidates.value or ()))
                self.quick_dp.set_value(list(self.form.dp_candidates.value or ()))
            else:
                self.quick_tp.set_value(float(self.form.tp.value))
                self.quick_pp.set_value(float(self.form.pp.value))
                self.quick_dp.set_value(float(self.form.dp.value))
            self.quick_calibration.set_value(str(self.form.calibration.value))
        finally:
            self._syncing_quick_controls = False

    def _set_quick_controls_busy(self, busy: bool) -> None:
        for control in self.quick_controls:
            control.disable() if busy else control.enable()

    def _configuration_changed(self, scope: str) -> None:
        self.local_error = None
        if scope == "shared":
            if self.analysis_outcome is not None:
                self.analysis_stale = True
            if self.sweep_report is not None:
                self.sweep_stale = True
        elif self.sweep_report is not None:
            self.sweep_stale = True
        if self.setup_summary is not None:
            self._render_setup_summary()
        self._render_drawer_footer()
        self._render_stale_status()
        self._sync_sidebar_controls()
        self._render_sidebar_summary()
        self._render_sidebar_action()

    def _render_workspace(self) -> None:
        self._detach_form()
        self.workspace.clear()
        self.setup_summary = None
        self.stale_host = None
        self.loading_progress = None
        self.loading_progress_label = None
        self.setup_action = None
        self.batch_results_host = None
        self.batch_status_filter = None
        self.batch_tp_filter = None
        self.batch_pp_filter = None
        self.batch_dp_filter = None
        self.batch_grid = None
        with self.workspace:
            if self.busy:
                self._render_loading()
            elif self.mode is WorkbenchMode.EVIDENCE:
                self.evidence_panel = EvidenceLabPanel(self.catalog)
                self.evidence_panel.build()
            elif self.mode is WorkbenchMode.FLOAT:
                self.float_panel = FloatAnalysisPanel()
                self.float_panel.build()
            elif self.mode is WorkbenchMode.ANALYSIS and self.analysis_outcome is not None:
                self._render_analysis_result()
            elif self.mode is WorkbenchMode.SWEEP and self.sweep_report is not None:
                self._render_sweep_result()
            else:
                self._render_setup()
        self._render_sidebar_summary()
        self._render_sidebar_action()

    def _render_sidebar_summary(self) -> None:
        if self.sidebar_summary is None:
            return
        self.sidebar_summary.clear()
        if self.mode is WorkbenchMode.EVIDENCE:
            with self.sidebar_summary:
                ui.label("EVIDENCE STATUS").classes("bp-sidebar-kicker")
                ui.label("Pinned profile").classes("bp-sidebar-title")
                ui.label("20 typed records · measured + analytical").classes("bp-sidebar-meta")
                ui.label("只读 PoC").classes("bp-sidebar-state bp-sidebar-state--ready")
            return
        if self.mode is WorkbenchMode.FLOAT:
            with self.sidebar_summary:
                ui.label("NUMERIC STATUS").classes("bp-sidebar-kicker")
                ui.label("交互分析").classes("bp-sidebar-title")
                ui.label("当前视图不运行 workload derivation。 ").classes("bp-sidebar-meta")
            return
        with self.sidebar_summary:
            ui.label("CASESET STATUS" if self.mode is WorkbenchMode.SWEEP else "CASE STATUS").classes(
                "bp-sidebar-kicker"
            )
            with ui.row().classes("w-full items-center justify-between gap-2"):
                ui.label("候选集合" if self.mode is WorkbenchMode.SWEEP else "当前 Case").classes("bp-sidebar-title")
            if self.form is None:
                ui.label("配置载入后显示摘要。 ").classes("bp-sidebar-meta")
                return
            summary = dict(self.form.summary())
            with ui.element("div").classes("bp-sidebar-fact-grid"):
                self._sidebar_fact("设备", summary["设备数"])
                self._sidebar_fact(
                    "批量",
                    f"{int(self.form.global_batch.value or 0)} / {int(self.form.microbatch.value or 0)}",
                )
                self._sidebar_fact("数据", str(self.form.datatype.value))
                if self.mode is WorkbenchMode.SWEEP:
                    self._sidebar_fact("Case", f"{self.form.candidate_count()} / 128")
                else:
                    self._sidebar_fact(
                        "TP · PP · DP",
                        f"{int(self.form.tp.value or 0)} · {int(self.form.pp.value or 0)} · "
                        f"{int(self.form.dp.value or 0)}",
                    )
            stale = self.analysis_stale if self.mode is WorkbenchMode.ANALYSIS else self.sweep_stale
            has_result = (
                self.analysis_outcome is not None
                if self.mode is WorkbenchMode.ANALYSIS
                else self.sweep_report is not None
            )
            if self.busy:
                status_class, status_text = "bp-sidebar-state bp-sidebar-state--active", "正在运行"
            elif stale:
                status_class, status_text = "bp-sidebar-state bp-sidebar-state--warning", "结果需要更新"
            elif has_result:
                status_class, status_text = "bp-sidebar-state bp-sidebar-state--ready", "结果已就绪"
            else:
                status_class, status_text = "bp-sidebar-state", "等待运行"
            ui.label(status_text).classes(status_class)

    def _render_sidebar_action(self) -> None:
        if self.sidebar_action_host is None:
            return
        self.sidebar_action_host.clear()
        if self.mode in {WorkbenchMode.EVIDENCE, WorkbenchMode.FLOAT}:
            return
        if self.form is None:
            return

        has_result = (
            self.analysis_outcome is not None if self.mode is WorkbenchMode.ANALYSIS else self.sweep_report is not None
        )
        stale = self.analysis_stale if self.mode is WorkbenchMode.ANALYSIS else self.sweep_stale
        if self.mode is WorkbenchMode.ANALYSIS:
            label = "更新当前 Case" if stale else ("重新剖析" if has_result else "剖析当前 Case")
            marker = "sidebar-run-analysis"
            callback = self.run_analysis
        else:
            label = "更新 CaseSet" if stale else ("重新评估" if has_result else "评估全部 Case")
            marker = "sidebar-run-sweep"
            callback = self.run_sweep

        with self.sidebar_action_host:
            action = (
                ui.button(label, icon="play_arrow", on_click=callback)
                .props("unelevated no-caps")
                .classes("bp-sidebar-primary w-full")
                .mark(marker)
            )
            if self.busy or (self.mode is WorkbenchMode.SWEEP and not 0 < self.form.candidate_count() <= 128):
                action.disable()

    @staticmethod
    def _sidebar_fact(label: str, value: str) -> None:
        with ui.column().classes("bp-sidebar-fact gap-0"):
            ui.label(label).classes("bp-sidebar-label")
            ui.label(value).classes("bp-sidebar-value")

    def _render_setup(self) -> None:
        if self.mode is WorkbenchMode.ANALYSIS:
            self._workspace_heading(
                "POINT LENS",
                "单点剖析",
                "聚焦一个 Case，解释可行性、解析任务贡献、资源约束与证据边界。",
            )
        else:
            self._workspace_heading(
                "BATCH LENS",
                "批量探索",
                "把一组 Case 作为整体，观察分布、上下界、可行边界并逐步收缩候选空间。",
            )
        self._ensure_form(self.drawer_form_host)
        with ui.element("section").classes("bp-evidence-surface bp-setup-evidence"):
            with ui.element("div").classes("bp-evidence-section"):
                self.setup_summary = ui.column().classes("w-full gap-3")
            with ui.element("div").classes("bp-evidence-section"):
                if self.mode is WorkbenchMode.ANALYSIS:
                    self._card_heading("这个视图回答什么", "从结果下钻到任务和证据，不要求先完成固定步骤。")
                    focus_items = (
                        ("01", "是否可行", "检查单设备容量约束和结构化诊断。"),
                        ("02", "时间花在哪里", "按 phase 展开解析任务贡献与延迟组成。"),
                        ("03", "结论能相信到哪里", "核对 evidence revision、限制与 canonical audit。"),
                    )
                else:
                    self._card_heading("这个视图回答什么", "从总体分布逐步过滤，再进入任意单点继续剖析。")
                    focus_items = (
                        ("01", "空间长什么样", "观察 Case 分布、成功率和上下界。"),
                        ("02", "边界在哪里", "识别容量不可行区域和延迟—内存非支配集。"),
                        ("03", "哪个点值得展开", "筛选并选择 Case，切换到单点剖析。"),
                    )
                with ui.element("div").classes("bp-focus-grid"):
                    for index, title, copy in focus_items:
                        self._setup_focus(index, title, copy)
        self._render_setup_summary()

    @staticmethod
    def _setup_focus(index: str, title: str, copy: str) -> None:
        with ui.element("div").classes("bp-focus-card"):
            ui.label(index).classes("bp-kicker bp-mono")
            ui.label(title).classes("bp-card-title mt-1")
            ui.label(copy).classes("bp-card-copy mt-1")

    def _render_setup_summary(self) -> None:
        if self.setup_summary is None or self.form is None:
            return
        self.setup_summary.clear()
        summary = dict(self.form.summary())
        with self.setup_summary:
            with ui.row().classes("w-full items-end justify-between gap-3"):
                with ui.column().classes("gap-0"):
                    ui.label("CURRENT CASESET" if self.mode is WorkbenchMode.SWEEP else "CURRENT CASE").classes(
                        "bp-kicker"
                    )
                    ui.label("批量候选定义" if self.mode is WorkbenchMode.SWEEP else "当前分析对象").classes(
                        "bp-result-title mt-1"
                    )
                ui.label(
                    f"{self.form.candidate_count()} cases"
                    if self.mode is WorkbenchMode.SWEEP
                    else f"{summary['设备数']} devices"
                ).classes("bp-context-chip bp-mono")
            with ui.element("div").classes("bp-case-summary-grid"):
                self._case_summary_item("工作负载", summary["模型"])
                self._case_summary_item("目标", summary["目标"])
                if self.mode is WorkbenchMode.SWEEP:
                    dimensions = (
                        f"{len(self.form.tp_candidates.value or ())} × "
                        f"{len(self.form.pp_candidates.value or ())} × "
                        f"{len(self.form.dp_candidates.value or ())}"
                    )
                    self._case_summary_item("搜索维度", dimensions)
                else:
                    self._case_summary_item("映射", summary["并行度"])
                self._case_summary_item("Evidence", summary["证据"])
            if self.local_error:
                with ui.row().classes("bp-inline-error items-start gap-2 no-wrap mt-2"):
                    ui.icon("error", size="18px")
                    ui.label(self.local_error).classes("text-xs")
            with ui.row().classes("bp-setup-boundary items-start gap-2 no-wrap"):
                ui.icon("info", size="17px", color="secondary")
                boundary = (
                    "当前显示解析任务贡献，不是事件级 Timeline。"
                    if self.mode is WorkbenchMode.ANALYSIS
                    else "当前批量空间仅覆盖所选 TP / PP / DP 组合，不代表通用硬件蓝图搜索。"
                )
                ui.label(boundary).classes("bp-card-copy")
            label = "剖析当前 Case" if self.mode is WorkbenchMode.ANALYSIS else "评估全部 Case"
            marker = "run-analysis" if self.mode is WorkbenchMode.ANALYSIS else "run-sweep"
            callback = self.run_analysis if self.mode is WorkbenchMode.ANALYSIS else self.run_sweep
            with ui.row().classes("bp-setup-actions w-full items-center justify-end gap-2"):
                ui.button("更多设置", icon="tune", on_click=self._open_configuration).props(
                    "outline dense no-caps"
                ).classes("bp-secondary-action").mark("setup-full-configuration")
                self.setup_action = (
                    ui.button(label, icon="play_arrow", on_click=callback)
                    .props("unelevated no-caps")
                    .classes("bp-primary-action")
                    .mark(marker)
                )
            if self.mode is WorkbenchMode.SWEEP and not 0 < self.form.candidate_count() <= 128:
                self.setup_action.disable()
                ui.label("候选组合必须在 1–128 之间。 ").classes("bp-card-copy bp-warning")

    @staticmethod
    def _case_summary_item(label: str, value: str) -> None:
        with ui.column().classes("bp-case-summary-item gap-1"):
            ui.label(label).classes("bp-summary-label")
            ui.label(value).classes("bp-case-summary-value bp-mono")

    def _open_configuration(self) -> None:
        if self.form is None:
            return
        if self.mode is WorkbenchMode.ANALYSIS:
            self.dialog_title.set_text("Case 完整配置")
            self.dialog_copy.set_text("编辑当前工作负载、映射、证据与高级约束。")
        else:
            self.dialog_title.set_text("CaseSet 定义")
            self.dialog_copy.set_text("编辑共享基线、候选范围与批量评估证据。")
        self.form.root.move(self.drawer_form_host)
        self.form.set_mode(self.mode)
        self._render_drawer_footer()
        self.config_dialog.open()

    def _render_drawer_footer(self) -> None:
        if not hasattr(self, "drawer_footer"):
            return
        self.drawer_footer.clear()
        self.drawer_action = None
        if self.form is None:
            return
        with self.drawer_footer:
            if self.local_error:
                with ui.row().classes("bp-inline-error items-start gap-2 no-wrap"):
                    ui.icon("error", size="18px")
                    ui.label(self.local_error).classes("text-xs")
            if self.mode is WorkbenchMode.SWEEP:
                with ui.row().classes("w-full justify-between"):
                    ui.label("Case 组合").classes("bp-summary-label")
                    ui.label(f"{self.form.candidate_count()} / 128").classes("bp-summary-value bp-mono")
            has_result = (
                self.analysis_outcome is not None
                if self.mode is WorkbenchMode.ANALYSIS
                else self.sweep_report is not None
            )
            if self.mode is WorkbenchMode.ANALYSIS:
                label = "重新剖析当前 Case" if has_result else "剖析当前 Case"
                marker = "rerun-analysis" if has_result else "drawer-run-analysis"
            else:
                label = "重新评估 CaseSet" if has_result else "评估全部 Case"
                marker = "rerun-sweep" if has_result else "drawer-run-sweep"
            callback = self.run_analysis if self.mode is WorkbenchMode.ANALYSIS else self.run_sweep
            self.drawer_action = (
                ui.button(label, icon="refresh", on_click=callback)
                .props("unelevated no-caps")
                .classes("bp-primary-action w-full")
                .mark(marker)
            )
            if self.mode is WorkbenchMode.SWEEP and not 0 < self.form.candidate_count() <= 128:
                self.drawer_action.disable()

    async def run_analysis(self) -> None:
        if self.form is None:
            return
        try:
            draft = self.form.draft()
        except (ValueError, TypeError) as error:
            self.local_error = str(error)
            self._render_setup_summary()
            self._render_drawer_footer()
            return

        self.local_error = None
        self.busy = True
        self.form.set_busy(True)
        self._set_quick_controls_busy(True)
        self._set_mode_buttons_busy(True)
        self.config_dialog.close()
        self._render_workspace()
        try:
            outcome = await run.io_bound(self.service.analyze, draft)
            if outcome is None:
                raise RuntimeError("分析服务没有返回结果")
            self.analysis_outcome = outcome
            self.analysis_stale = False
        except Exception as error:  # pragma: no cover - NiceGUI safety boundary
            self.local_error = f"未预期的界面错误：{error}"
            self.analysis_stale = self.analysis_outcome is not None
            ui.notify(self.local_error, type="negative", position="bottom-right", close_button=True)
        finally:
            self.busy = False
            self.form.set_busy(False)
            self._set_quick_controls_busy(False)
            self._set_mode_buttons_busy(False)
            self._render_workspace()

    async def run_sweep(self) -> None:
        if self.form is None:
            return
        try:
            request = self.form.sweep_request()
        except (ValueError, TypeError) as error:
            self.local_error = str(error)
            self._render_setup_summary()
            self._render_drawer_footer()
            return

        self.local_error = None
        self._progress_completed = 0
        self._progress_total = request.candidate_count
        self.busy = True
        self.form.set_busy(True)
        self._set_quick_controls_busy(True)
        self._set_mode_buttons_busy(True)
        self.config_dialog.close()
        self.progress_timer.activate()
        self._render_workspace()

        def on_progress(completed: int, total: int) -> None:
            self._progress_completed = completed
            self._progress_total = total

        try:
            report = await run.io_bound(self.service.sweep, request, on_progress)
            if report is None:
                raise RuntimeError("批量评估服务没有返回结果")
            self.sweep_report = report
            self.sweep_base = request.base
            self.sweep_stale = False
            self._progress_completed = request.candidate_count
        except Exception as error:  # pragma: no cover - NiceGUI safety boundary
            self.local_error = f"未预期的界面错误：{error}"
            self.sweep_stale = self.sweep_report is not None
            ui.notify(self.local_error, type="negative", position="bottom-right", close_button=True)
        finally:
            self._poll_progress()
            self.progress_timer.deactivate()
            self.busy = False
            self.form.set_busy(False)
            self._set_quick_controls_busy(False)
            self._set_mode_buttons_busy(False)
            self._render_workspace()

    def _render_loading(self) -> None:
        if self.mode is WorkbenchMode.ANALYSIS:
            self._workspace_heading(
                "POINT LENS",
                "正在剖析当前 Case",
                "Case 定义已冻结；正在推导任务、验证约束并解析硬件证据。",
            )
            label = "ModelIR → DistributedTaskIR → PortablePlanIR"
        else:
            self._workspace_heading(
                "BATCH LENS",
                "正在评估 CaseSet",
                "每个 TP / PP / DP Case 都会独立推导；失败项保留诊断并参与分布统计。",
            )
            label = f"{self._progress_total} cases"
        with (
            ui.element("section").classes("bp-loading-panel"),
            ui.column().classes("items-center gap-3").style("width: min(520px, 100%)"),
        ):
            with ui.element("div").classes("bp-loading-glyph"):
                ui.spinner("grid", size="30px", color="secondary")
            ui.label("评估进行中").classes("bp-result-title")
            ui.label(label).classes("bp-card-copy bp-mono")
            if self.mode is WorkbenchMode.SWEEP:
                self.loading_progress = ui.linear_progress(value=0, show_value=False, color="secondary").classes(
                    "w-full mt-2"
                )
                self.loading_progress_label = ui.label("0 / 0 cases").classes("bp-card-copy bp-mono")

    def _poll_progress(self) -> None:
        if self.loading_progress is None or self.loading_progress_label is None:
            return
        total = self._progress_total
        value = min(self._progress_completed / total, 1.0) if total > 0 else 0
        self.loading_progress.set_value(value)
        self.loading_progress_label.set_text(f"{self._progress_completed} / {total} cases")

    def _render_result_context(
        self,
        title: str,
        copy: str,
        chips: tuple[str, ...],
    ) -> None:
        with ui.element("section").classes("bp-result-context"):
            with ui.row().classes("bp-result-context-row w-full items-center gap-3"):
                with ui.column().classes("gap-0 min-w-0"):
                    ui.label(title).classes("bp-result-title")
                    ui.label(copy).classes("bp-card-copy")
                with ui.row().classes("bp-result-chips gap-1"):
                    for chip in chips:
                        ui.label(chip).classes("bp-context-chip")
                ui.space()
                ui.button("配置", icon="tune", on_click=self._open_configuration).props(
                    "outline dense no-caps"
                ).classes("bp-secondary-action bp-result-config no-wrap").mark("edit-configuration")
            self.stale_host = ui.column().classes("w-full")
        self._render_stale_status()

    def _render_stale_status(self) -> None:
        if self.stale_host is None:
            return
        self.stale_host.clear()
        stale = self.analysis_stale if self.mode is WorkbenchMode.ANALYSIS else self.sweep_stale
        if stale:
            with self.stale_host, ui.row().classes("bp-stale-banner items-center gap-2 no-wrap"):
                ui.icon("sync_problem", size="18px")
                ui.label("配置已经变化；当前页面仍显示上一次结果。重新运行后才会替换。 ").classes("text-xs")

    def _render_analysis_result(self) -> None:
        outcome = self.analysis_outcome
        assert outcome is not None
        report = outcome.report
        if report is None:
            self._render_result_context(
                "当前 Case 未生成计划",
                "配置验证或推导阶段返回了结构化诊断。",
                (f"request {outcome.request_digest[:16]}",),
            )
            self._render_diagnostics(outcome.diagnostics)
            return

        with ui.column().classes("bp-result-header w-full gap-2"):
            self._render_result_context(
                f"{report.model_name} × {report.hardware_name}",
                "从迭代总量下钻到 phase、subsystem、operation 与完整 portable task graph。",
                (
                    f"world {report.world_size}",
                    report.calibration_mode,
                    f"plan {report.plan_digest[:12]}",
                ),
            )
            with (
                ui.tabs()
                .props("dense no-caps indicator-color=primary active-color=primary")
                .classes("bp-result-tabs") as tabs
            ):
                conclusion_tab = ui.tab("conclusion", "时间剖析").mark("tab-conclusion")
                workload_tab = ui.tab("workload", "任务与工作量").mark("tab-workload")
                derivation_tab = ui.tab("derivation", "技术审计").mark("tab-derivation")
        with ui.tab_panels(tabs, value=conclusion_tab, animated=False, keep_alive=True).classes(
            "bp-result-panels w-full"
        ):
            with ui.tab_panel(conclusion_tab), ui.column().classes("w-full gap-4 pt-3"):
                self._render_analysis_conclusion(outcome)
            with ui.tab_panel(workload_tab), ui.column().classes("w-full gap-4 pt-3"):
                self._render_workload(outcome)
            with ui.tab_panel(derivation_tab), ui.column().classes("w-full gap-4 pt-3"):
                self._render_derivation(outcome)

    def _render_analysis_conclusion(self, outcome: AnalysisOutcome) -> None:
        report = outcome.report
        assert report is not None
        evidence_surface = ui.element("section").classes("bp-evidence-surface")

        with evidence_surface, ui.element("div").classes("bp-evidence-section"):
            self._render_diagnostics(outcome.diagnostics)
            status_class = "bp-status-banner" if report.feasible else "bp-status-banner bp-status-banner--warning"
            status_icon = "check_circle" if report.feasible else "warning"
            status_text = (
                "当前配置满足单设备内存容量约束。"
                if report.feasible
                else "分析已完成，但当前配置超过单设备内存容量约束。"
            )
            with ui.row().classes(f"{status_class} items-center gap-2"):
                ui.icon(status_icon, size="18px")
                ui.label(status_text).classes("text-sm")

            with ui.element("div").classes("bp-metric-grid"):
                for metric in analysis_metrics(report):
                    with ui.element("div").classes("bp-metric").style(f"--metric-color: {METRIC_COLORS[metric.tone]}"):
                        ui.label(metric.label).classes("bp-metric-label")
                        ui.label(metric.value).classes("bp-metric-value")
                        ui.label(metric.detail).classes("bp-metric-detail")

            bottleneck = _BOTTLENECK_LABELS.get(report.bottleneck, report.bottleneck.replace("_", " "))
            with ui.element("div").classes("bp-insight"):
                ui.label("当前主导项").classes("bp-kicker")
                ui.label(bottleneck).classes("bp-result-title mt-1")
                ui.label("这是当前解析式延迟分解中的最大组成项，不等同于事件仿真的 critical path。 ").classes(
                    "bp-card-copy mt-1"
                )

        with evidence_surface, ui.element("section").classes("bp-evidence-section"):
            with ui.row().classes("bp-chain-header w-full items-start justify-between gap-3"):
                with ui.column().classes("gap-0"):
                    ui.label("自顶向下时间分解").classes("bp-card-title")
                    ui.label(
                        "Iteration → 时间项 → phase / subsystem → source layer → operation；点击矩形继续下钻。 "
                    ).classes("bp-card-copy")
                ui.label("ITERATION HIERARCHY").classes("bp-fidelity-tag bp-mono")
            ui.echart(time_breakdown_chart_options(report), renderer="canvas").classes("w-full bp-time-treemap")
            with ui.row().classes("bp-time-method items-start gap-2 no-wrap"):
                ui.icon("functions", size="17px", color="secondary")
                ui.label(
                    "第一级严格使用 iteration estimate；有 task 证据的时间项按 block 内贡献比例继续分解，pipeline bubble、PP 与 DP 等调度项在当前层级保持为不可再分的解析项。"
                ).classes("bp-card-copy")
            breakdown_rows = [
                {
                    **row,
                    "share_iteration_pct": round(float(row["share_iteration"]) * 100, 4),
                    "share_parent_pct": round(float(row["share_parent"]) * 100, 4),
                }
                for row in time_breakdown_rows(report)
            ]
            with (
                ui.expansion("查看完整层次明细", icon="account_tree", value=False).classes(
                    "bp-time-details bp-card w-full"
                ),
                ui.column().classes("w-full gap-2 pt-2"),
            ):
                ui.aggrid(
                    {
                        "columnDefs": [
                            {"headerName": "Level", "field": "level", "width": 82},
                            {"headerName": "Path", "field": "path", "pinned": "left", "minWidth": 360},
                            {"headerName": "Seconds", "field": "seconds", "type": "numericColumn"},
                            {
                                "headerName": "% Iteration",
                                "field": "share_iteration_pct",
                                "type": "numericColumn",
                            },
                            {"headerName": "% Parent", "field": "share_parent_pct", "type": "numericColumn"},
                            {"headerName": "Kind", "field": "category", "minWidth": 150},
                        ],
                        "rowData": breakdown_rows,
                        "defaultColDef": {"sortable": True, "filter": True, "resizable": True},
                        "pagination": True,
                        "paginationPageSize": 25,
                    },
                    theme="quartz",
                    auto_size_columns=False,
                ).classes("w-full bp-grid").style("height: 430px")

        projection = timeline_summary(report)
        trace_filename = f"portable-projection-{report.plan_digest[:12]}.json"
        trace_json = portable_projection_trace_json(report)
        with evidence_surface, ui.element("section").classes("bp-evidence-section"):
            with ui.row().classes("bp-chain-header w-full items-start justify-between gap-3"):
                with ui.column().classes("gap-0"):
                    ui.label("Portable task timeline").classes("bp-card-title")
                    ui.label(
                        "完整绘制当前 PortablePlan block 的全部 task；开始时间仅由 dependency 推导，支持横向缩放。 "
                    ).classes("bp-card-copy")
                with ui.row().classes("gap-1"):
                    ui.label("BLOCK SCOPE").classes("bp-fidelity-tag bp-mono")
                    ui.label("DEPENDENCY PROJECTION").classes("bp-fidelity-tag bp-mono")
            with ui.row().classes("items-center gap-2"):
                ui.button("在 Perfetto 中打开", icon="open_in_new").props("outline dense no-caps").classes(
                    "bp-secondary-action"
                ).mark("open-perfetto").on(
                    "click",
                    js_handler=perfetto_open_javascript(
                        trace_json,
                        title=f"Blueprinting portable projection · {report.plan_digest[:12]}",
                        filename=trace_filename,
                    ),
                )
            with ui.row().classes("bp-timeline-legend items-center gap-4"):
                for engine, label in (("matrix", "Matrix"), ("vector", "Vector"), ("collective", "Collective")):
                    with ui.row().classes("items-center gap-1"):
                        ui.element("span").classes(f"bp-engine-dot bp-engine-dot--{engine}")
                        ui.label(label).classes("bp-summary-label")
            ui.echart(dependency_timeline_chart_options(report), renderer="canvas").classes("w-full bp-timeline-chart")
            with ui.element("div").classes("bp-chain-stats"):
                for label, value in (
                    ("Projected span", format_seconds(projection.span_seconds)),
                    ("Portable tasks", f"{projection.task_count:,}"),
                    ("Dependencies", f"{projection.dependency_count:,}"),
                    ("Phase lanes", f"{projection.lane_count:,}"),
                ):
                    with ui.row().classes("items-center gap-2"):
                        ui.label(label).classes("bp-summary-label")
                        ui.label(value).classes("bp-card-copy bp-mono")
            with ui.row().classes("bp-time-scope items-start gap-2 no-wrap"):
                ui.icon("info", size="17px", color="secondary")
                ui.label(
                    "该时间轴覆盖完整 portable block task graph，但尚未展开 block multiplicity、microbatch、pipeline stage、queue、overlap 与 contention；这些信息需要 ConcretePlan 和 SimulationTraceIR。"
                ).classes("bp-card-copy")

        with (
            evidence_surface,
            ui.element("div").classes("bp-evidence-section"),
            ui.element("div").classes("bp-chart-grid"),
        ):
            with ui.element("section").classes("bp-evidence-block"):
                self._card_heading("迭代一级时间项", "与上方层次视图的第一级一致；蓝色条为当前最大组成项。")
                ui.echart(latency_chart_options(report), renderer="svg").classes("w-full h-80")
            with ui.element("section").classes("bp-evidence-block"):
                self._card_heading(
                    "单设备内存构成",
                    f"总量 {format_bytes(report.memory['total'])} / 容量 {format_bytes(report.memory['capacity'])}",
                )
                ui.echart(memory_chart_options(report), renderer="svg").classes("w-full h-80")

        with (
            evidence_surface,
            ui.element("div").classes("bp-evidence-section"),
            ui.element("div").classes("bp-detail-grid bp-detail-grid--wide"),
        ):
            with ui.element("section").classes("bp-evidence-block"):
                self._card_heading("证据与适用边界", report.evidence_revision)
                evidence = report.evidence
                for label, value in (
                    ("证据模式", report.calibration_mode),
                    ("Matrix peak", f"{format_count(evidence['matrix_peak_ops_per_second'])} op/s"),
                    ("Vector peak", f"{format_count(evidence['vector_peak_ops_per_second'])} op/s"),
                    ("Memory peak", f"{format_bytes(evidence['memory_peak_bytes_per_second'])}/s"),
                    ("网络层级", str(evidence["network_tiers"])),
                ):
                    self._fact_row(label, value)
                ui.label("实现边界").classes("bp-card-title mt-3")
                for limitation in report.limitations:
                    with ui.row().classes("items-start gap-2 no-wrap"):
                        ui.icon("subdirectory_arrow_right", size="16px", color="grey-6")
                        ui.label(limitation).classes("bp-card-copy")
            with ui.element("section").classes("bp-evidence-block"):
                self._card_heading("结果身份", "用于复现与问题定位")
                for label, value in (
                    ("Request", report.request_digest),
                    ("Session", report.session_fingerprint),
                    ("Plan", report.plan_digest),
                    ("Schema", report.schema),
                ):
                    ui.label(label).classes("bp-summary-label mt-2")
                    ui.label(value).classes("bp-card-copy bp-mono break-all")

    def _render_workload(self, outcome: AnalysisOutcome) -> None:
        report = outcome.report
        assert report is not None
        workload = report.workload
        evidence_surface = ui.element("section").classes("bp-evidence-surface")
        facts = (
            ("Portable tasks", f"{workload['task_count']:,}", "primary"),
            (
                "Compute / collective",
                f"{workload['compute_task_count']:,} / {workload['collective_task_count']:,}",
                "cyan",
            ),
            ("Operations", format_count(workload["operations"]), "violet"),
            (
                "Read / write",
                f"{format_bytes(workload['read_bytes'])} / {format_bytes(workload['write_bytes'])}",
                "amber",
            ),
        )
        with (
            evidence_surface,
            ui.element("div").classes("bp-evidence-section"),
            ui.element("div").classes("bp-metric-grid"),
        ):
            for label, value, tone in facts:
                with ui.element("div").classes("bp-metric").style(f"--metric-color: {METRIC_COLORS[tone]}"):
                    ui.label(label).classes("bp-metric-label")
                    ui.label(value).classes("bp-metric-value")
                    detail = (
                        format_bytes(workload["message_bytes"])
                        if label == "Compute / collective"
                        else "exact workload fact"
                    )
                    ui.label(f"Messages {detail}" if label == "Compute / collective" else detail).classes(
                        "bp-metric-detail"
                    )

        with evidence_surface, ui.element("section").classes("bp-evidence-section"):
            self._card_heading("Portable task audit", f"{len(report.tasks):,} 个任务；预测列来自当前 evidence view。")
            ui.aggrid(
                {
                    "columnDefs": [
                        {"headerName": "Operation", "field": "operation", "pinned": "left", "minWidth": 220},
                        {"headerName": "Phase", "field": "phase"},
                        {"headerName": "Engine", "field": "engine"},
                        {"headerName": "Kind", "field": "kind"},
                        {"headerName": "Ops", "field": "ops", "type": "numericColumn"},
                        {"headerName": "Read B", "field": "read_bytes", "type": "numericColumn"},
                        {"headerName": "Write B", "field": "write_bytes", "type": "numericColumn"},
                        {"headerName": "Message B", "field": "message_bytes", "type": "numericColumn"},
                        {"headerName": "Compute s", "field": "compute_s", "type": "numericColumn"},
                        {"headerName": "Memory s", "field": "memory_s", "type": "numericColumn"},
                        {"headerName": "Network s", "field": "network_s", "type": "numericColumn"},
                        {"headerName": "Total s", "field": "total_s", "type": "numericColumn"},
                        {"headerName": "Deps", "field": "dependencies", "type": "numericColumn"},
                        {"headerName": "Concurrency", "field": "concurrency_group", "minWidth": 150},
                    ],
                    "rowData": task_rows(outcome),
                    "defaultColDef": {"sortable": True, "filter": True, "resizable": True},
                    "pagination": True,
                    "paginationPageSize": 25,
                },
                theme="quartz",
                auto_size_columns=False,
            ).classes("w-full bp-grid").style("height: 520px")

    def _render_derivation(self, outcome: AnalysisOutcome) -> None:
        report = outcome.report
        assert report is not None
        evidence_surface = ui.element("section").classes("bp-evidence-surface")
        with evidence_surface, ui.element("div").classes("bp-evidence-section"):
            ui.label("Canonical derivation checkpoints").classes("bp-kicker")
            ui.label("每一层结果在提交前经过 verifier，并保留父 digest 与不可变快照。 ").classes("bp-card-copy")
            with ui.element("div").classes("bp-stage-flow mt-3"):
                for index, stage in enumerate(report.stages):
                    with ui.element("div").classes("bp-stage"):
                        with ui.row().classes("w-full items-center justify-between"):
                            ui.label(f"0{index + 1}").classes("bp-kicker bp-mono")
                            ui.icon(
                                "verified" if stage.valid else "error",
                                color="positive" if stage.valid else "negative",
                                size="18px",
                            )
                        ui.label(stage.label).classes("bp-card-title mt-2")
                        ui.label(stage.schema).classes("bp-card-copy bp-mono")
                        ui.label(f"{stage.node_count:,} nodes · {stage.value_count:,} values/buffers").classes(
                            "bp-card-copy mt-2"
                        )
                        ui.label(format_seconds(stage.duration_ns / 1e9)).classes("bp-card-copy bp-mono text-secondary")

        with evidence_surface, ui.element("div").classes("bp-evidence-section bp-derivation-details"):
            for stage in report.stages:
                with (
                    ui.expansion(
                        stage.label,
                        caption=f"{stage.pass_name} · {stage.digest[:16]}",
                        icon="verified" if stage.valid else "error",
                        value=False,
                    ).classes("bp-card w-full"),
                    ui.column().classes("w-full gap-3 pt-2"),
                ):
                    with ui.row().classes("gap-2"):
                        for value in (
                            stage.schema,
                            f"{stage.node_count} nodes",
                            f"{stage.value_count} values",
                            format_seconds(stage.duration_ns / 1e9),
                        ):
                            ui.label(value).classes("bp-data-chip bp-mono")
                    if stage.parent_digests:
                        ui.label(f"Parents · {' · '.join(stage.parent_digests)}").classes(
                            "bp-card-copy bp-mono break-all"
                        )
                    self._render_diagnostics(stage.diagnostics)
                    pretty_snapshot = json.dumps(json.loads(stage.snapshot_json), ensure_ascii=False, indent=2)
                    ui.code(pretty_snapshot, language="json").classes("bp-code")
                    ui.button(
                        "下载 canonical snapshot",
                        icon="download",
                        on_click=partial(self._download_snapshot, stage.stage, stage.digest, pretty_snapshot),
                    ).props("outline dense no-caps").classes("bp-secondary-action")

    @staticmethod
    def _download_snapshot(stage: str, digest: str, content: str) -> None:
        ui.download(
            content.encode("utf-8"),
            filename=f"{stage}-{digest[:12]}.json",
            media_type="application/json",
        )

    def _render_sweep_result(self) -> None:
        report = self.sweep_report
        assert report is not None
        base = self.sweep_base
        chips = [f"{len(report.cases)} cases", f"request {report.request_digest[:12]}"]
        title = "TP / PP / DP CaseSet"
        if base is not None:
            title = f"{base.model_name} × {base.hardware_name}"
            chips.insert(0, base.calibration_mode.value)
        self._render_result_context(
            title,
            "观察整批 Case 的分布、上下界和可行边界；选择任意一行可继续单点剖析。",
            tuple(chips),
        )
        rows = sweep_rows(report)
        with ui.element("section").classes("bp-evidence-surface"):
            with ui.element("div").classes("bp-evidence-section"):
                with ui.row().classes("w-full items-end justify-between gap-3"):
                    self._card_heading("CaseSet 过滤器", "过滤即时作用于分布、上下界和候选表，不会重新运行分析。")
                    ui.button("清除过滤", icon="filter_alt_off", on_click=self._clear_batch_filters).props(
                        "flat dense no-caps"
                    ).classes("bp-filter-clear")
                with ui.element("div").classes("bp-batch-filter-grid"):
                    self.batch_status_filter = (
                        ui.select(
                            _BATCH_STATUS_LABELS,
                            label="状态",
                            value="all",
                            on_change=self._batch_filter_changed,
                        )
                        .props("outlined dense options-dense")
                        .classes("bp-batch-filter")
                        .mark("batch-status-filter")
                    )
                    axes = {
                        "tp": sorted({int(row["tp"]) for row in rows}),
                        "pp": sorted({int(row["pp"]) for row in rows}),
                        "dp": sorted({int(row["dp"]) for row in rows}),
                    }
                    self.batch_tp_filter = self._batch_axis_filter("TP", axes["tp"], "batch-tp-filter")
                    self.batch_pp_filter = self._batch_axis_filter("PP", axes["pp"], "batch-pp-filter")
                    self.batch_dp_filter = self._batch_axis_filter("DP", axes["dp"], "batch-dp-filter")
            self.batch_results_host = ui.column().classes("w-full gap-0")
        self._render_filtered_sweep_content()

    def _batch_axis_filter(self, label: str, options: list[int], marker: str) -> Any:
        return (
            ui.select(
                options,
                label=label,
                value=[],
                multiple=True,
                on_change=self._batch_filter_changed,
            )
            .props("outlined dense use-chips options-dense")
            .classes("bp-batch-filter")
            .mark(marker)
        )

    def _batch_filter_changed(self, *_: Any) -> None:
        self._render_filtered_sweep_content()

    def _clear_batch_filters(self) -> None:
        if self.batch_status_filter is None:
            return
        self.batch_status_filter.set_value("all")
        for control in (self.batch_tp_filter, self.batch_pp_filter, self.batch_dp_filter):
            if control is not None:
                control.set_value([])
        self._render_filtered_sweep_content()

    def _filtered_sweep_rows(self) -> list[dict[str, Any]]:
        report = self.sweep_report
        if report is None:
            return []
        rows = sweep_rows(report)
        status = str(self.batch_status_filter.value) if self.batch_status_filter is not None else "all"
        selected_axes = {
            "tp": set(self.batch_tp_filter.value or ()) if self.batch_tp_filter is not None else set(),
            "pp": set(self.batch_pp_filter.value or ()) if self.batch_pp_filter is not None else set(),
            "dp": set(self.batch_dp_filter.value or ()) if self.batch_dp_filter is not None else set(),
        }

        def visible(row: dict[str, Any]) -> bool:
            if status == "feasible" and not (row["status"] == "success" and row["feasible"]):
                return False
            if status == "infeasible" and not (row["status"] == "success" and not row["feasible"]):
                return False
            if status == "pareto" and not row["pareto"]:
                return False
            if status == "failed" and row["status"] == "success":
                return False
            return all(not values or row[axis] in values for axis, values in selected_axes.items())

        return [row for row in rows if visible(row)]

    def _render_filtered_sweep_content(self) -> None:
        report = self.sweep_report
        if report is None or self.batch_results_host is None:
            return
        rows = self._filtered_sweep_rows()
        self.batch_results_host.clear()
        self.batch_grid = None
        with self.batch_results_host:
            successful = [row for row in rows if row["status"] == "success"]
            latency_values = [float(row["latency_s"]) for row in successful if row["latency_s"] is not None]
            memory_values = [float(row["memory_gib"]) for row in successful if row["memory_gib"] is not None]
            visible_feasible = sum(bool(row["feasible"]) for row in successful)
            visible_pareto = sum(bool(row["pareto"]) for row in successful)
            summary = (
                ("可见 Case", f"{len(rows)} / {len(report.cases)}", "primary", "当前过滤结果"),
                ("容量可行", str(visible_feasible), "cyan", f"非支配 {visible_pareto}"),
                ("延迟上下界", self._range_text(latency_values, format_seconds), "violet", "成功 Case"),
                (
                    "内存上下界",
                    self._range_text(memory_values, lambda value: f"{value:.2f} GiB"),
                    "amber",
                    "单设备",
                ),
            )
            with (
                ui.element("section").classes("bp-evidence-section"),
                ui.element("div").classes("bp-metric-grid"),
            ):
                for label, value, tone, detail in summary:
                    with ui.element("div").classes("bp-metric").style(f"--metric-color: {METRIC_COLORS[tone]}"):
                        ui.label(label).classes("bp-metric-label")
                        ui.label(value).classes("bp-metric-value bp-metric-value--range")
                        ui.label(detail).classes("bp-metric-detail")

            if successful:
                with (
                    ui.element("section").classes("bp-evidence-section"),
                    ui.element("div").classes("bp-batch-chart-grid"),
                ):
                    with ui.element("section").classes("bp-evidence-block"):
                        self._card_heading(
                            "延迟—内存空间",
                            "越靠左下越优；非支配标记仅针对原始 CaseSet 的延迟与单设备内存。",
                        )
                        ui.echart(sweep_chart_options(report, rows), renderer="svg").classes("w-full bp-batch-chart")
                    with ui.element("section").classes("bp-evidence-block"):
                        self._card_heading("可见分布", "直方图只统计当前过滤后成功完成的 Case。")
                        ui.echart(sweep_distribution_chart_options(report, rows), renderer="svg").classes(
                            "w-full bp-batch-chart"
                        )
            else:
                with ui.element("section").classes("bp-evidence-section"):
                    self._empty_state(
                        "当前过滤没有可绘制的 Case",
                        "调整状态或并行度过滤器；失败 Case 仍保留在候选表中。",
                        "filter_alt_off",
                    )

            sorted_rows = sorted(
                rows,
                key=lambda row: (
                    not bool(row["pareto"]),
                    not bool(row["feasible"]),
                    row["latency_s"] is None,
                    row["latency_s"] if row["latency_s"] is not None else float("inf"),
                ),
            )
            with ui.element("section").classes("bp-evidence-section"):
                with ui.row().classes("w-full items-end justify-between gap-3"):
                    self._card_heading("可见 Case", "选择一行后，可直接切换到单点模式重新剖析该 Case。")
                    ui.button("单点剖析所选", icon="query_stats", on_click=self._open_selected_batch_case).props(
                        "outline dense no-caps"
                    ).classes("bp-secondary-action").mark("batch-open-point")
                self.batch_grid = (
                    ui.aggrid(
                        {
                            "columnDefs": [
                                {"headerName": "TP", "field": "tp", "pinned": "left", "width": 76},
                                {"headerName": "PP", "field": "pp", "width": 76},
                                {"headerName": "DP", "field": "dp", "width": 76},
                                {"headerName": "World", "field": "world_size", "type": "numericColumn"},
                                {"headerName": "Status", "field": "status"},
                                {"headerName": "Feasible", "field": "feasible"},
                                {"headerName": "Non-dominated", "field": "pareto", "minWidth": 145},
                                {"headerName": "Latency s", "field": "latency_s", "type": "numericColumn"},
                                {"headerName": "Memory GiB", "field": "memory_gib", "type": "numericColumn"},
                                {
                                    "headerName": "Token/s/device",
                                    "field": "tokens_s_device",
                                    "type": "numericColumn",
                                },
                                {"headerName": "Bottleneck", "field": "bottleneck", "minWidth": 150},
                                {
                                    "headerName": "Diagnostic",
                                    "field": "diagnostic",
                                    "minWidth": 280,
                                    "tooltipField": "diagnostic",
                                },
                            ],
                            "rowData": sorted_rows,
                            "rowSelection": "single",
                            "defaultColDef": {"sortable": True, "filter": True, "resizable": True},
                            "pagination": True,
                            "paginationPageSize": 25,
                        },
                        theme="quartz",
                        auto_size_columns=False,
                    )
                    .classes("w-full bp-grid")
                    .style("height: 500px")
                )

            invalid = [row for row in rows if row["status"] != "success"]
            if invalid:
                with (
                    ui.element("section").classes("bp-evidence-section"),
                    ui.expansion(f"失败 Case 诊断（{len(invalid)}）", icon="warning", value=False).classes(
                        "bp-card w-full"
                    ),
                    ui.column().classes("w-full gap-2 pt-2"),
                ):
                    for row in invalid:
                        with ui.element("div").classes("bp-diagnostic"):
                            ui.label(f"TP{row['tp']} / PP{row['pp']} / DP{row['dp']}").classes("bp-card-title bp-mono")
                            ui.label(row["diagnostic"] or "未提供诊断").classes("bp-card-copy")

    @staticmethod
    def _range_text(values: list[float], formatter: Callable[[float], str]) -> str:
        if not values:
            return "—"
        lower = min(values)
        upper = max(values)
        if lower == upper:
            return formatter(lower)
        return f"{formatter(lower)} – {formatter(upper)}"

    async def _open_selected_batch_case(self) -> None:
        if self.batch_grid is None or self.form is None:
            return
        row = await self.batch_grid.get_selected_row()
        if row is None:
            ui.notify("请先在候选表中选择一个 Case。", type="warning", position="bottom-right")
            return
        self.form.load_point_parallelism(int(row["tp"]), int(row["pp"]), int(row["dp"]))
        self.mode = WorkbenchMode.ANALYSIS
        self._sync_mode_buttons()
        self.form.set_mode(self.mode)
        self.local_error = None
        self._render_sidebar_controls()
        await self.run_analysis()

    def _render_diagnostics(self, diagnostics: tuple[AnalysisDiagnostic, ...]) -> None:
        colors = {
            DiagnosticLevel.ERROR: "#b91c1c",
            DiagnosticLevel.WARNING: "#b45309",
            DiagnosticLevel.INFO: "#2563eb",
        }
        icons = {
            DiagnosticLevel.ERROR: "error",
            DiagnosticLevel.WARNING: "warning",
            DiagnosticLevel.INFO: "info",
        }
        for diagnostic in diagnostics:
            with (
                ui.element("div").classes("bp-diagnostic").style(f"--diagnostic-color: {colors[diagnostic.level]}"),
                ui.row().classes("items-start gap-3 no-wrap"),
            ):
                ui.icon(icons[diagnostic.level], color=colors[diagnostic.level], size="19px")
                with ui.column().classes("gap-1"):
                    location = ".".join(diagnostic.path)
                    title = diagnostic.code + (f" · {location}" if location else "")
                    ui.label(title).classes("bp-card-title bp-mono")
                    ui.label(diagnostic.message).classes("bp-card-copy")
                    if diagnostic.hint:
                        ui.label(f"建议：{diagnostic.hint}").classes("bp-card-copy text-secondary")

    @staticmethod
    def _workspace_heading(kicker: str, title: str, copy: str) -> None:
        with ui.column().classes("bp-workspace-heading gap-0"):
            ui.label(kicker).classes("bp-kicker")
            ui.label(title).classes("bp-page-title")
            ui.label(copy).classes("bp-page-copy")

    @staticmethod
    def _card_heading(title: str, copy: str) -> None:
        with ui.column().classes("gap-1 mb-3"):
            ui.label(title).classes("bp-card-title")
            ui.label(copy).classes("bp-card-copy")

    @staticmethod
    def _fact_row(label: str, value: str) -> None:
        with ui.row().classes("bp-fact-row items-center justify-between gap-4"):
            ui.label(label).classes("bp-summary-label")
            ui.label(value).classes("bp-summary-value bp-mono")

    @staticmethod
    def _empty_state(title: str, copy: str, icon: str) -> None:
        with ui.element("div").classes("bp-empty"), ui.column().classes("items-center gap-2"):
            ui.icon(icon, size="34px", color="secondary")
            ui.label(title).classes("bp-result-title")
            ui.label(copy).classes("bp-card-copy")


def create_workbench_root(
    *,
    catalog: ConfigCatalog | None = None,
    service_factory: Callable[[], BlueprintingService] = BlueprintingService,
) -> Callable[[], None]:
    """Create an injectable NiceGUI root callable for runtime and tests."""

    def root() -> None:
        BlueprintingWorkbench(catalog=catalog, service_factory=service_factory).build()

    return root


workbench_root = create_workbench_root()

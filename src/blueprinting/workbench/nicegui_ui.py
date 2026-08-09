"""NiceGUI architecture-exploration workbench.

The workbench is deliberately a thin client around :class:`BlueprintingService`.
It owns per-browser UI state and presentation only; it never assembles lowering
passes or computes hardware estimates itself.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from functools import partial
from typing import Any

from nicegui import run, ui

from blueprinting.application import (
    AnalysisDiagnostic,
    AnalysisDraft,
    AnalysisOutcome,
    BlueprintingService,
    DiagnosticLevel,
    SweepReport,
    SweepRequest,
)
from blueprinting.compiler.analysis import CalibrationMode

from .catalog import ConfigCatalog, default_catalog
from .nicegui_theme import METRIC_COLORS, WORKBENCH_CSS
from .presentation import (
    analysis_metrics,
    format_bytes,
    format_count,
    format_seconds,
    latency_chart_options,
    memory_chart_options,
    stage_rows,
    sweep_chart_options,
    sweep_rows,
    task_rows,
)

_COMPILER_DTYPES = ("float16", "bfloat16", "float32", "float8")
_PARALLEL_OPTIONS = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128)
_CALIBRATION_LABELS = {
    "系统证据曲线": CalibrationMode.SYSTEM_EVIDENCE,
    "理论峰值基线": CalibrationMode.PEAK_ONLY,
}


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or value is None or int(value) <= 0:
        raise ValueError(f"{label} 必须是正整数")
    return int(value)


def _strip_json_suffix(name: str) -> str:
    return name.removesuffix(".json")


class ConfigurationPanel:
    """Own the editable configuration controls for one browser tab."""

    def __init__(
        self,
        catalog: ConfigCatalog,
        *,
        on_analyze: Callable[[], Awaitable[None]],
        on_sweep: Callable[[], Awaitable[None]],
    ) -> None:
        self.catalog = catalog
        self.on_analyze = on_analyze
        self.on_sweep = on_sweep
        self._controls: list[Any] = []

    def build(self) -> None:
        model_names = self.catalog.names("models")
        execution_names = self.catalog.names("examples")
        hardware_names = self.catalog.names("systems")
        model_name = self.catalog.preferred("models", "gpt3-175B.json")
        execution_name = self.catalog.preferred("examples", execution_names[0])
        hardware_name = self.catalog.preferred("systems", "a100_80g.json")

        ui.label("Exploration inputs").classes("bp-drawer-title")
        ui.label("配置只描述问题；推导与估算由应用服务完成。").classes("text-xs bp-muted leading-relaxed")

        with ui.column().classes("bp-form-section gap-2"):
            self.model_preset = self._select("模型预设", model_names, model_name, self._reload_presets)
            self.hardware_preset = self._select("硬件证据", hardware_names, hardware_name, self._reload_presets)
            self.execution_preset = self._select("策略模板", execution_names, execution_name, self._reload_presets)

        model = self.catalog.load("models", model_name)
        execution = self.catalog.load("examples", execution_name)
        hardware = self.catalog.load("systems", hardware_name)

        with (
            ui.expansion("模型语义", icon="schema", value=False).classes("bp-form-section"),
            ui.column().classes("w-full gap-2 pt-2"),
        ):
            self.hidden = self._number("Hidden size", model["hidden"])
            self.feedforward = self._number("Feed-forward size", model["feedforward"])
            self.sequence = self._number("Sequence length", model["seq_size"])
            self.heads = self._number("Attention heads", model["attn_heads"])
            self.head_size = self._number("Attention head size", model["attn_size"])
            self.blocks = self._number("Transformer blocks", model["num_blocks"])

        with (
            ui.expansion("执行策略", icon="account_tree", value=True).classes("bp-form-section"),
            ui.column().classes("w-full gap-2 pt-2"),
        ):
            with ui.row().classes("w-full gap-2 no-wrap"):
                self.tp = self._number("TP", execution["tensor_par"], on_change=self._update_world_size)
                self.pp = self._number("PP", execution["pipeline_par"], on_change=self._update_world_size)
                self.dp = self._number("DP", execution["data_par"], on_change=self._update_world_size)
            self.world_size_label = ui.label().classes("text-xs bp-muted bp-mono")
            self.global_batch = self._number("Global batch", execution["batch_size"])
            self.microbatch = self._number("Microbatch", execution["microbatch_size"])
            self.datatype = self._select(
                "Datatype",
                self._supported_datatypes(hardware_name),
                self._initial_datatype(execution, hardware_name),
            )
            self.recompute = self._select(
                "Activation recompute",
                ("none", "attn_only", "full"),
                execution.get("activation_recompute", "none"),
            )
            self.communication = self._select(
                "TP communication",
                ("ar", "rs_ag"),
                execution.get("tensor_par_comm_type", "ar"),
            )
            self.interleaving = self._number("Pipeline interleaving", execution.get("pipeline_interleaving", 1))
            self.optimizer_sharding = ui.checkbox(
                "Optimizer sharding",
                value=bool(execution.get("optimizer_sharding", False)),
            ).props("dense")
            self._controls.append(self.optimizer_sharding)

        with (
            ui.expansion("网络映射", icon="hub", value=False).classes("bp-form-section"),
            ui.column().classes("w-full gap-2 pt-2"),
        ):
            network_options = tuple(range(len(hardware.get("networks", ()))))
            self.tp_network = self._select(
                "TP network tier",
                network_options,
                min(int(execution.get("tensor_par_net", 0)), len(network_options) - 1),
            )
            self.pp_network = self._select(
                "PP network tier",
                network_options,
                min(int(execution.get("pipeline_par_net", 0)), len(network_options) - 1),
            )
            self.dp_network = self._select(
                "DP network tier",
                network_options,
                min(int(execution.get("data_par_net", 0)), len(network_options) - 1),
            )

        with (
            ui.expansion("策略搜索空间", icon="travel_explore", value=False).classes("bp-form-section"),
            ui.column().classes("w-full gap-2 pt-2"),
        ):
            options = self._parallel_options(execution)
            self.tp_candidates = self._multi_select(
                "TP candidates", options, [int(execution["tensor_par"])], self._update_candidate_count
            )
            self.pp_candidates = self._multi_select(
                "PP candidates", options, [int(execution["pipeline_par"])], self._update_candidate_count
            )
            self.dp_candidates = self._multi_select(
                "DP candidates", options, [int(execution["data_par"])], self._update_candidate_count
            )
            self.candidate_count_label = ui.label().classes("text-xs bp-muted bp-mono")

        with ui.column().classes("bp-form-section gap-2"):
            ui.label("估算证据").classes("text-xs bp-muted")
            self.calibration = ui.radio(
                list(_CALIBRATION_LABELS),
                value="系统证据曲线",
            ).props("dense")
            self._controls.append(self.calibration)

        self.analysis_button = (
            ui.button("运行单点分析", icon="play_arrow", on_click=self.on_analyze)
            .props("unelevated no-caps")
            .classes("w-full h-11")
            .mark("run-analysis")
        )
        self.sweep_button = (
            ui.button("探索策略空间", icon="scatter_plot", on_click=self.on_sweep, color="secondary")
            .props("outline no-caps")
            .classes("w-full h-11")
            .mark("run-sweep")
        )
        with ui.column().classes("w-full gap-1"):
            self.busy_row = ui.row().classes("items-center gap-2")
            with self.busy_row:
                ui.spinner("dots", size="22px", color="secondary")
                self.busy_label = ui.label("正在执行分析…").classes("text-xs bp-muted")
            self.progress = ui.linear_progress(value=0, show_value=False, color="secondary").classes("w-full")
            self.progress_label = ui.label().classes("text-xs bp-muted bp-mono")
        self.busy_row.set_visibility(False)
        self.progress.set_visibility(False)
        self.progress_label.set_visibility(False)
        self._update_world_size()
        self._update_candidate_count()

    def _select(
        self,
        label: str,
        options: tuple[Any, ...],
        value: Any,
        on_change: Callable[..., Any] | None = None,
    ) -> Any:
        control = (
            ui.select(list(options), label=label, value=value, on_change=on_change)
            .props("outlined dense options-dense")
            .classes("w-full")
        )
        self._controls.append(control)
        return control

    def _multi_select(
        self,
        label: str,
        options: tuple[int, ...],
        value: list[int],
        on_change: Callable[..., Any],
    ) -> Any:
        control = (
            ui.select(list(options), label=label, value=value, multiple=True, on_change=on_change)
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
            ui.number(label, value=float(value), min=1, step=1, precision=0, on_change=on_change)
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

    def _reload_presets(self, *_: Any) -> None:
        model_name, execution_name, hardware_name = self._selection_names()
        model = self.catalog.load("models", model_name)
        execution = self.catalog.load("examples", execution_name)
        hardware = self.catalog.load("systems", hardware_name)

        for control, value in (
            (self.hidden, model["hidden"]),
            (self.feedforward, model["feedforward"]),
            (self.sequence, model["seq_size"]),
            (self.heads, model["attn_heads"]),
            (self.head_size, model["attn_size"]),
            (self.blocks, model["num_blocks"]),
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

        datatypes = self._supported_datatypes(hardware_name)
        self.datatype.set_options(list(datatypes), value=self._initial_datatype(execution, hardware_name))
        network_options = tuple(range(len(hardware.get("networks", ()))))
        if not network_options:
            raise ValueError(f"硬件预设 {hardware_name} 没有网络层级")
        for control, field in (
            (self.tp_network, "tensor_par_net"),
            (self.pp_network, "pipeline_par_net"),
            (self.dp_network, "data_par_net"),
        ):
            control.set_options(
                list(network_options),
                value=min(int(execution.get(field, 0)), len(network_options) - 1),
            )

        options = self._parallel_options(execution)
        self.tp_candidates.set_options(list(options), value=[int(execution["tensor_par"])])
        self.pp_candidates.set_options(list(options), value=[int(execution["pipeline_par"])])
        self.dp_candidates.set_options(list(options), value=[int(execution["data_par"])])
        self._update_world_size()
        self._update_candidate_count()

    def _update_world_size(self, *_: Any) -> None:
        values = (self.tp.value, self.pp.value, self.dp.value)
        if any(value is None for value in values):
            self.world_size_label.set_text("World size: —")
            return
        world_size = int(values[0]) * int(values[1]) * int(values[2])
        self.world_size_label.set_text(f"World size: {world_size:,}")

    def candidate_count(self) -> int:
        return (
            len(self.tp_candidates.value or ())
            * len(self.pp_candidates.value or ())
            * len(self.dp_candidates.value or ())
        )

    def _update_candidate_count(self, *_: Any) -> None:
        count = self.candidate_count()
        self.candidate_count_label.set_text(f"Candidates: {count} / 128")
        if count > 128:
            self.candidate_count_label.classes(add="text-negative", remove="bp-muted")
        else:
            self.candidate_count_label.classes(add="bp-muted", remove="text-negative")

    def draft(self) -> AnalysisDraft:
        model_name, execution_name, hardware_name = self._selection_names()
        hardware_data = self.catalog.load("systems", hardware_name)
        execution_data = self.catalog.load("examples", execution_name)
        tp = _positive_int(self.tp.value, "TP")
        pp = _positive_int(self.pp.value, "PP")
        dp = _positive_int(self.dp.value, "DP")
        model_data = {
            "hidden": _positive_int(self.hidden.value, "Hidden size"),
            "feedforward": _positive_int(self.feedforward.value, "Feed-forward size"),
            "seq_size": _positive_int(self.sequence.value, "Sequence length"),
            "attn_heads": _positive_int(self.heads.value, "Attention heads"),
            "attn_size": _positive_int(self.head_size.value, "Attention head size"),
            "num_blocks": _positive_int(self.blocks.value, "Transformer blocks"),
        }
        execution_data.update(
            {
                "tensor_par": tp,
                "pipeline_par": pp,
                "data_par": dp,
                "num_procs": tp * pp * dp,
                "batch_size": _positive_int(self.global_batch.value, "Global batch"),
                "microbatch_size": _positive_int(self.microbatch.value, "Microbatch"),
                "datatype": str(self.datatype.value),
                "activation_recompute": str(self.recompute.value),
                "tensor_par_comm_type": str(self.communication.value),
                "pipeline_interleaving": _positive_int(self.interleaving.value, "Pipeline interleaving"),
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

    def set_busy(self, busy: bool, label: str = "正在执行分析…") -> None:
        for control in self._controls:
            control.disable() if busy else control.enable()
        self.analysis_button.disable() if busy else self.analysis_button.enable()
        self.sweep_button.disable() if busy else self.sweep_button.enable()
        self.busy_label.set_text(label)
        self.busy_row.set_visibility(busy)

    def set_progress(self, completed: int, total: int, *, visible: bool) -> None:
        self.progress.set_visibility(visible)
        self.progress_label.set_visibility(visible)
        if total <= 0:
            self.progress.set_value(0)
            self.progress_label.set_text("")
            return
        self.progress.set_value(min(completed / total, 1.0))
        self.progress_label.set_text(f"{completed} / {total} candidates")


class BlueprintingWorkbench:
    """Per-client NiceGUI workbench and asynchronous application controller."""

    def __init__(
        self,
        *,
        catalog: ConfigCatalog | None = None,
        service_factory: Callable[[], BlueprintingService] = BlueprintingService,
    ) -> None:
        self.catalog = catalog or default_catalog()
        self.service = service_factory()
        self.analysis_outcome: AnalysisOutcome | None = None
        self.sweep_report: SweepReport | None = None
        self._progress_completed = 0
        self._progress_total = 0

    def build(self) -> None:
        ui.add_css(WORKBENCH_CSS)
        ui.colors(primary="#7c68ff", secondary="#22d3ee", accent="#f59e0b", dark="#070a12")
        ui.dark_mode(True)
        ui.page_title("Blueprinting · Architecture Workbench")
        self._build_legacy_dialog()

        with ui.header(elevated=False).classes("bp-header items-center no-wrap"):
            ui.button(icon="tune", on_click=self._toggle_drawer).props("flat round dense").classes("lt-md")
            with ui.row().classes("items-center gap-3 no-wrap"):
                with ui.element("div").classes("bp-brand-mark"):
                    ui.icon("architecture", size="22px", color="secondary")
                with ui.column().classes("gap-0"):
                    ui.label("Blueprinting").classes("bp-brand-title")
                    ui.label("Hardware architecture workbench").classes("bp-brand-subtitle")

            with (
                ui.tabs()
                .props("dense no-caps indicator-color=secondary active-color=white")
                .classes("self-stretch q-ml-lg") as self.tabs
            ):
                self.overview_tab = ui.tab("overview", "分析总览", icon="dashboard").mark("tab-overview")
                self.ir_tab = ui.tab("ir", "IR 推导审计", icon="schema").mark("tab-ir")
                self.sweep_tab = ui.tab("sweep", "策略空间", icon="scatter_plot").mark("tab-sweep")
            ui.space()
            ui.chip("service ready", icon="check_circle", color="transparent", text_color="positive").props(
                "dense outline"
            ).classes("gt-sm bp-mono")
            ui.button("Legacy", icon="history", on_click=self.legacy_dialog.open).props("flat dense no-caps")

        with ui.left_drawer(value=True, bordered=False) as self.drawer:
            self.drawer.props("width=318 breakpoint=900").classes("bp-drawer")
            with ui.column().classes("w-full gap-3"):
                self.form = ConfigurationPanel(
                    self.catalog,
                    on_analyze=self.run_analysis,
                    on_sweep=self.run_sweep,
                )
                self.form.build()
            self.progress_timer = ui.timer(0.12, self._poll_progress, active=False, immediate=False)

        with ui.element("main").classes("bp-main"):
            self._render_hero()
            with ui.tab_panels(self.tabs, value=self.overview_tab, animated=True, keep_alive=True).classes("w-full"):
                with ui.tab_panel(self.overview_tab):
                    self.overview_content = ui.column().classes("w-full gap-5")
                with ui.tab_panel(self.ir_tab):
                    self.ir_content = ui.column().classes("w-full gap-5")
                with ui.tab_panel(self.sweep_tab):
                    self.sweep_content = ui.column().classes("w-full gap-5")

        self._render_overview()
        self._render_ir_audit()
        self._render_sweep()

    def _build_legacy_dialog(self) -> None:
        with ui.dialog() as self.legacy_dialog, ui.card().classes("bp-card p-6").style("width: 560px; max-width: 92vw"):
            with ui.row().classes("items-center gap-3"):
                ui.icon("inventory_2", size="28px", color="secondary")
                with ui.column().classes("gap-0"):
                    ui.label("Calculon / Streamlit Legacy").classes("text-lg font-semibold")
                    ui.label("旧界面保持隔离，不参与 Blueprinting 分析路径。").classes("text-xs bp-muted")
            ui.separator().classes("my-2")
            ui.label("需要旧 Calculon 或浮点工具时，单独启动：").classes("text-sm")
            ui.code("uv run streamlit run streamlit_app.py", language="bash").classes("bp-code")
            with ui.row().classes("w-full justify-end gap-2"):
                ui.link("打开 localhost:8501", "http://127.0.0.1:8501", new_tab=True).classes("text-secondary")
                ui.button("关闭", on_click=self.legacy_dialog.close).props("flat no-caps")

    def _toggle_drawer(self) -> None:
        if hasattr(self, "drawer"):
            self.drawer.toggle()

    def _render_hero(self) -> None:
        with ui.element("section").classes("bp-hero"):
            ui.label("FORMAL EXPLORATION · EVIDENCE DRIVEN").classes("bp-kicker")
            ui.label("从工作负载语义走向可审计的硬件蓝图").classes("bp-hero-title")
            ui.label(
                "以类型化 IR、可验证 lowering 和版本化性能证据，把模型、并行策略与候选硬件映射成可比较的执行计划。"
            ).classes("bp-hero-copy")
            with ui.row().classes("gap-2 mt-4"):
                for label in ("ModelIR", "DistributedTaskIR", "PortablePlanIR", "Evidence", "Pareto"):
                    ui.label(label).classes("bp-chip")

    async def run_analysis(self) -> None:
        try:
            draft = self.form.draft()
            self.form.set_busy(True, "正在推导 canonical IR 并估算…")
            ui.notify("分析已开始", type="info", position="bottom-right")
            outcome = await run.io_bound(self.service.analyze, draft)
            if outcome is None:
                raise RuntimeError("分析服务没有返回结果")
            self.analysis_outcome = outcome
            self._render_overview()
            self._render_ir_audit()
            self.tabs.set_value(self.overview_tab)
            if outcome.ok:
                ui.notify("分析完成", type="positive", position="bottom-right")
            else:
                ui.notify("分析返回结构化诊断", type="warning", position="bottom-right")
        except (ValueError, TypeError) as error:
            ui.notify(str(error), type="negative", position="bottom-right", close_button=True)
        except Exception as error:  # pragma: no cover - NiceGUI safety boundary
            ui.notify(f"未预期的界面错误：{error}", type="negative", position="bottom-right", close_button=True)
        finally:
            self.form.set_busy(False)

    async def run_sweep(self) -> None:
        try:
            request = self.form.sweep_request()
            self._progress_completed = 0
            self._progress_total = request.candidate_count
            self.form.set_progress(0, request.candidate_count, visible=True)
            self.form.set_busy(True, f"正在探索 {request.candidate_count} 个候选…")
            self.progress_timer.activate()

            def on_progress(completed: int, total: int) -> None:
                self._progress_completed = completed
                self._progress_total = total

            report = await run.io_bound(self.service.sweep, request, on_progress)
            if report is None:
                raise RuntimeError("策略搜索服务没有返回结果")
            self.sweep_report = report
            self._progress_completed = request.candidate_count
            self._render_sweep()
            self.tabs.set_value(self.sweep_tab)
            ui.notify(
                f"策略搜索完成：{report.succeeded_count}/{len(report.cases)} 成功",
                type="positive",
                position="bottom-right",
            )
        except (ValueError, TypeError) as error:
            ui.notify(str(error), type="negative", position="bottom-right", close_button=True)
        except Exception as error:  # pragma: no cover - NiceGUI safety boundary
            ui.notify(f"未预期的界面错误：{error}", type="negative", position="bottom-right", close_button=True)
        finally:
            self._poll_progress()
            self.progress_timer.deactivate()
            self.form.set_busy(False)

    def _poll_progress(self) -> None:
        if hasattr(self, "form"):
            self.form.set_progress(self._progress_completed, self._progress_total, visible=self._progress_total > 0)

    def _render_overview(self) -> None:
        self.overview_content.clear()
        with self.overview_content:
            self._section_heading(
                "Architecture analysis",
                "分析总览",
                "单点分析保持 workload、mapping、portable plan 与 hardware evidence 的边界可见。",
            )
            outcome = self.analysis_outcome
            if outcome is None:
                self._empty_state(
                    "等待第一张硬件蓝图",
                    "从左侧选择模型、硬件证据和执行策略，然后运行单点分析。",
                    "route",
                )
                self._render_capability_cards()
                return

            self._render_diagnostics(outcome.diagnostics)
            report = outcome.report
            if report is None:
                ui.label(f"Request {outcome.request_digest}").classes("bp-mono text-xs bp-muted")
                return

            status_class = "bp-status" if report.feasible else "bp-status bp-status--warning"
            status_icon = "check_circle" if report.feasible else "warning"
            status_text = (
                "该候选满足当前单设备内存容量约束。"
                if report.feasible
                else "该候选完成了分析，但不满足单设备内存容量约束。"
            )
            with ui.row().classes(f"{status_class} items-center gap-2"):
                ui.icon(status_icon, size="18px")
                ui.label(status_text).classes("text-sm")

            with ui.row().classes("w-full gap-3"):
                for metric in analysis_metrics(report):
                    with ui.element("div").classes("bp-metric").style(f"--metric-color: {METRIC_COLORS[metric.tone]}"):
                        ui.label(metric.label).classes("bp-metric-label")
                        ui.label(metric.value).classes("bp-metric-value")
                        ui.label(metric.detail).classes("bp-metric-detail")

            with ui.row().classes("w-full gap-4 items-stretch"):
                with ui.card().classes("bp-card p-4").style("flex: 1 1 560px"):
                    self._card_heading("延迟分解", "结构化 latency components，而不是后验调平系数。")
                    ui.echart(latency_chart_options(report), renderer="svg").classes("w-full h-80")
                with ui.card().classes("bp-card p-4").style("flex: 1 1 560px"):
                    self._card_heading(
                        "内存分解",
                        f"{format_bytes(report.memory['total'])} / {format_bytes(report.memory['capacity'])}",
                    )
                    ui.echart(memory_chart_options(report), renderer="svg").classes("w-full h-80")

            with ui.row().classes("w-full gap-4 items-stretch"):
                with ui.card().classes("bp-card p-5").style("flex: 1 1 420px"):
                    self._card_heading("Portable workload facts", "IR 推导出的工作量事实")
                    workload = report.workload.to_dict()
                    facts = (
                        ("Portable tasks", f"{workload['task_count']:,}"),
                        (
                            "Compute / Collective",
                            f"{workload['compute_task_count']:,} / {workload['collective_task_count']:,}",
                        ),
                        ("Operations", format_count(workload["operations"])),
                        (
                            "Read / Write",
                            f"{format_bytes(workload['read_bytes'])} / {format_bytes(workload['write_bytes'])}",
                        ),
                        ("Messages", format_bytes(workload["message_bytes"])),
                    )
                    for label, value in facts:
                        with ui.row().classes("w-full justify-between items-center py-1"):
                            ui.label(label).classes("text-xs bp-muted")
                            ui.label(value).classes("text-sm bp-mono")
                with ui.card().classes("bp-card p-5").style("flex: 2 1 620px"):
                    self._card_heading("Evidence & implementation boundary", report.evidence_revision)
                    with ui.row().classes("w-full gap-2"):
                        ui.badge(report.calibration_mode, color="primary")
                        ui.badge(report.hardware_name, color="secondary", text_color="dark")
                        ui.badge(f"world {report.world_size}", color="grey-8")
                    for limitation in report.limitations:
                        with ui.row().classes("items-start gap-2 no-wrap"):
                            ui.icon("subdirectory_arrow_right", size="16px", color="grey-6")
                            ui.label(limitation).classes("text-xs bp-muted leading-relaxed")
                    ui.label(f"Plan {report.plan_digest[:16]} · Request {report.request_digest[:16]}").classes(
                        "text-xs bp-mono bp-muted mt-2"
                    )

    def _render_capability_cards(self) -> None:
        capabilities = (
            ("形式化推导", "ModelIR → DistributedTaskIR → PortablePlanIR", "schema", "#7c68ff"),
            ("证据估算", "计算、访存、通信与容量约束保持来源可见", "query_stats", "#22d3ee"),
            ("策略探索", "保留失败候选并生成延迟—内存 Pareto 前沿", "scatter_plot", "#f59e0b"),
        )
        with ui.row().classes("w-full gap-3"):
            for title, copy, icon, color in capabilities:
                with ui.card().classes("bp-card p-5").style("flex: 1 1 260px"):
                    ui.icon(icon, color=color, size="24px")
                    ui.label(title).classes("font-semibold mt-2")
                    ui.label(copy).classes("text-xs bp-muted leading-relaxed")

    def _render_ir_audit(self) -> None:
        self.ir_content.clear()
        with self.ir_content:
            self._section_heading(
                "Derivation audit",
                "IR 推导审计",
                "检查每一层 canonical checkpoint、Verifier、lineage digest 与 Portable task workload。",
            )
            outcome = self.analysis_outcome
            if outcome is None:
                self._empty_state("尚无推导记录", "运行单点分析后，这里会复用同一结果进行逐层审计。", "schema")
                return
            self._render_diagnostics(outcome.diagnostics)
            report = outcome.report
            if report is None:
                return

            with ui.row().classes("w-full gap-3 items-stretch"):
                for index, stage in enumerate(report.stages):
                    with ui.element("div").classes("bp-stage"):
                        with ui.row().classes("w-full items-center justify-between"):
                            ui.label(f"0{index + 1}").classes("bp-kicker bp-mono")
                            ui.icon("verified", color="positive" if stage.valid else "negative", size="18px")
                        ui.label(stage.label).classes("font-semibold mt-2")
                        ui.label(stage.schema).classes("text-xs bp-mono bp-muted")
                        ui.label(f"{stage.node_count:,} nodes · {stage.value_count:,} values/buffers").classes(
                            "text-xs bp-muted mt-2"
                        )
                        ui.label(format_seconds(stage.duration_ns / 1e9)).classes("text-xs bp-mono text-secondary")
                    if index < len(report.stages) - 1:
                        ui.icon("arrow_forward", color="grey-7", size="20px").classes("self-center gt-sm")

            ui.aggrid(
                {
                    "columnDefs": [
                        {"headerName": "Stage", "field": "label", "pinned": "left", "minWidth": 150},
                        {"headerName": "Pass", "field": "pass", "minWidth": 210},
                        {"headerName": "Schema", "field": "schema", "minWidth": 190},
                        {"headerName": "Nodes", "field": "nodes", "type": "numericColumn"},
                        {"headerName": "Values", "field": "values", "type": "numericColumn"},
                        {"headerName": "Lowering ms", "field": "lowering_ms", "type": "numericColumn"},
                        {"headerName": "Valid", "field": "valid"},
                        {"headerName": "Digest", "field": "digest", "minWidth": 260},
                    ],
                    "rowData": stage_rows(outcome),
                    "defaultColDef": {"sortable": True, "filter": True, "resizable": True},
                    "domLayout": "autoHeight",
                },
                theme="quartz",
            ).classes("w-full bp-grid")

            for stage in report.stages:
                with (
                    ui.expansion(
                        stage.label,
                        caption=f"{stage.pass_name} · {stage.digest[:16]}",
                        icon="verified" if stage.valid else "error",
                        value=False,
                    ).classes("bp-card w-full"),
                    ui.column().classes("w-full gap-3 p-3"),
                ):
                    with ui.row().classes("gap-2"):
                        ui.badge(stage.schema, color="grey-8")
                        ui.badge(f"{stage.node_count} nodes", color="primary")
                        ui.badge(f"{stage.value_count} values", color="secondary", text_color="dark")
                        ui.badge(format_seconds(stage.duration_ns / 1e9), color="grey-8")
                    self._render_diagnostics(stage.diagnostics)
                    pretty_snapshot = json.dumps(json.loads(stage.snapshot_json), ensure_ascii=False, indent=2)
                    ui.code(pretty_snapshot, language="json").classes("bp-code")
                    ui.button(
                        "下载 canonical snapshot",
                        icon="download",
                        on_click=partial(self._download_snapshot, stage.stage, stage.digest, pretty_snapshot),
                    ).props("outline dense no-caps")

            self._card_heading("Portable task audit", f"{len(report.tasks):,} tasks")
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

    @staticmethod
    def _download_snapshot(stage: str, digest: str, content: str) -> None:
        ui.download(
            content.encode("utf-8"),
            filename=f"{stage}-{digest[:12]}.json",
            media_type="application/json",
        )

    def _render_sweep(self) -> None:
        self.sweep_content.clear()
        with self.sweep_content:
            self._section_heading(
                "Design-space exploration",
                "策略空间探索",
                "批量推导 TP/PP/DP 候选，保留失败原因，并标记延迟—内存 Pareto 前沿。",
            )
            report = self.sweep_report
            if report is None:
                self._empty_state(
                    "尚未探索配置空间",
                    "在左侧展开策略搜索空间，选择候选集合后开始探索；最多接受 128 个候选。",
                    "scatter_plot",
                )
                return

            rows = sweep_rows(report)
            pareto_count = sum(bool(row["pareto"]) for row in rows)
            summary = (
                ("候选", len(report.cases), "#7c68ff"),
                ("成功", report.succeeded_count, "#22d3ee"),
                ("可行", report.feasible_count, "#34d399"),
                ("Pareto", pareto_count, "#f59e0b"),
            )
            with ui.row().classes("w-full gap-3"):
                for label, value, color in summary:
                    with ui.element("div").classes("bp-metric").style(f"--metric-color: {color}"):
                        ui.label(label).classes("bp-metric-label")
                        ui.label(str(value)).classes("bp-metric-value")
                        ui.label("strategy candidates").classes("bp-metric-detail")

            successful = [row for row in rows if row["status"] == "success" and row["latency_s"] is not None]
            if successful:
                with ui.card().classes("bp-card p-4"):
                    self._card_heading("延迟—内存空间", "亮色点为非支配解；失败候选不会被静默丢弃。")
                    ui.echart(sweep_chart_options(report), renderer="svg").classes("w-full h-96")

            sorted_rows = sorted(
                rows,
                key=lambda row: (
                    not bool(row["pareto"]),
                    not bool(row["feasible"]),
                    row["latency_s"] is None,
                    row["latency_s"] if row["latency_s"] is not None else float("inf"),
                ),
            )
            self._card_heading("全部候选", f"Request {report.request_digest[:16]}")
            ui.aggrid(
                {
                    "columnDefs": [
                        {"headerName": "TP", "field": "tp", "pinned": "left", "width": 80},
                        {"headerName": "PP", "field": "pp", "width": 80},
                        {"headerName": "DP", "field": "dp", "width": 80},
                        {"headerName": "World", "field": "world_size", "type": "numericColumn"},
                        {"headerName": "Status", "field": "status"},
                        {"headerName": "Feasible", "field": "feasible"},
                        {"headerName": "Pareto", "field": "pareto"},
                        {"headerName": "Latency s", "field": "latency_s", "type": "numericColumn"},
                        {"headerName": "Memory GiB", "field": "memory_gib", "type": "numericColumn"},
                        {"headerName": "Token/s/device", "field": "tokens_s_device", "type": "numericColumn"},
                        {"headerName": "Bottleneck", "field": "bottleneck", "minWidth": 150},
                        {
                            "headerName": "Diagnostic",
                            "field": "diagnostic",
                            "minWidth": 280,
                            "tooltipField": "diagnostic",
                        },
                    ],
                    "rowData": sorted_rows,
                    "defaultColDef": {"sortable": True, "filter": True, "resizable": True},
                    "pagination": True,
                    "paginationPageSize": 25,
                },
                theme="quartz",
                auto_size_columns=False,
            ).classes("w-full bp-grid").style("height: 520px")

            invalid = [row for row in rows if row["status"] != "success"]
            if invalid:
                with ui.expansion(
                    f"无效候选诊断（{len(invalid)}）",
                    icon="warning",
                    value=False,
                ).classes("bp-card w-full"):
                    for row in invalid:
                        with ui.element("div").classes("bp-diagnostic"):
                            ui.label(f"TP{row['tp']} / PP{row['pp']} / DP{row['dp']}").classes(
                                "text-xs font-semibold bp-mono"
                            )
                            ui.label(row["diagnostic"] or "未提供诊断").classes("text-xs bp-muted")

    def _render_diagnostics(self, diagnostics: tuple[AnalysisDiagnostic, ...]) -> None:
        colors = {
            DiagnosticLevel.ERROR: "#fb7185",
            DiagnosticLevel.WARNING: "#f59e0b",
            DiagnosticLevel.INFO: "#22d3ee",
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
                    ui.label(title).classes("text-xs font-semibold bp-mono")
                    ui.label(diagnostic.message).classes("text-xs bp-muted")
                    if diagnostic.hint:
                        ui.label(f"建议：{diagnostic.hint}").classes("text-xs text-secondary")

    @staticmethod
    def _section_heading(kicker: str, title: str, copy: str) -> None:
        with ui.column().classes("gap-1"):
            ui.label(kicker).classes("bp-kicker")
            ui.label(title).classes("bp-section-title")
            ui.label(copy).classes("bp-section-copy")

    @staticmethod
    def _card_heading(title: str, copy: str) -> None:
        with ui.column().classes("gap-1 mb-2"):
            ui.label(title).classes("font-semibold")
            ui.label(copy).classes("text-xs bp-muted")

    @staticmethod
    def _empty_state(title: str, copy: str, icon: str) -> None:
        with (
            ui.element("div").classes("bp-empty"),
            ui.column().classes("items-center gap-2 p-8"),
        ):
            ui.icon(icon, size="36px", color="primary")
            ui.label(title).classes("text-lg font-semibold")
            ui.label(copy).classes("max-w-xl text-xs bp-muted leading-relaxed")


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

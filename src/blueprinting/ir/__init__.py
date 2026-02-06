"""IR module for blueprinting compiler.

GraphIR (Block-level) → ScheduleIR (Op-level) → TimelineIR (Event-level) → SimulationResult
"""

import contextlib

from blueprinting.core import SymMax, SymMin, eval_lazy, sym_max, sym_min

from .compiler import Compiler
from .dsl import build_transformer_model
from .estimate import SymbolicEstimate
from .passes import (
    ExpandPass,
    OptimizerConfig,
    OptimizerPass,
    OverlapAnalysisPass,
    ParallelPass,
    Pass,
    Pipeline,
    PipelineConfig,
    PipelineSchedulePass,
    PPScheduleMode,
    PrintGraphPass,
    PrintResultPass,
    PrintSchedulePass,
    PrintTimelinePass,
    SchedulePass,
    SimulatePass,
    SymbolicEstimatePass,
    TimelinePass,
)
from .render import GraphRenderMixin, ScheduleRenderMixin, SimulationResultRenderMixin, TimelineRenderMixin
from .result import SimulationResult
from .system import SystemConfig, load_system_config
from .types import (
    BlockNode,
    DeviceSchedule,
    EventType,
    GraphIR,
    MemorySnapshot,
    NodeType,
    ScheduledOp,
    ScheduleIR,
    StageSchedule,
    StreamState,
    StreamType,
    TensorLifetime,
    TensorRef,
    TimelineEvent,
    TimelineIR,
)


def _bind_render_methods():
    """将 Render Mixin 的方法动态绑定到 IR 类型."""
    render_methods = ("to_terminal", "to_tree_html", "to_table", "_repr_html_")
    for name in render_methods:
        if hasattr(GraphRenderMixin, name):
            setattr(GraphIR, name, getattr(GraphRenderMixin, name))
        if hasattr(ScheduleRenderMixin, name):
            setattr(ScheduleIR, name, getattr(ScheduleRenderMixin, name))
        if hasattr(TimelineRenderMixin, name):
            setattr(TimelineIR, name, getattr(TimelineRenderMixin, name))
    for name in ("to_terminal", "_repr_html_"):
        if hasattr(SimulationResultRenderMixin, name):
            setattr(SimulationResult, name, getattr(SimulationResultRenderMixin, name))


with contextlib.suppress(ImportError):
    _bind_render_methods()

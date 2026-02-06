"""IR module for blueprinting compiler.

Hierarchical IR Structure:

GraphIR (model-side view):
    Module → Block → Op
    - ModuleNode: Top-level container (the model)
    - BlockNode: Named blocks (TransformerLayer, Attention, FFN)
    - OpNode: Leaf operations (Linear, RMSNorm, etc.)
    - Data flow expressed through tensor name binding

ScheduleIR (execution-side view):
    Stage → Device → ScheduledOp
    - StageSchedule: Pipeline stage grouping
    - DeviceSchedule: Per-device operation assignment
    - ScheduledOp: Operation with timing and event sequence
    - Sorted by event_seq → TimelineIR

TimelineIR (simulation view):
    Flat event stream sorted by time
    - Fine-grained events for precise simulation
"""

from .dsl import build_transformer_model
from .compiler import Compiler
# 所有 IR 类型从 types.py 导入
from .types import BlockNode, GraphIR, NodeType, TensorRef
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
                      TimelinePass,
)
from .estimate import SymbolicEstimate
from .result import SimulationResult
# DeviceSchedule, ScheduledOp, StageSchedule, TensorLifetime 已迁移到 types.py
from .types import DeviceSchedule, ScheduledOp, StageSchedule, TensorLifetime
# 符号化计算基础设施（从 core 模块重新导出）
from blueprinting.core import (
    SymMax,
    SymMin,
    sym_max,
    sym_min,
    eval_lazy,
)
from .system import SystemConfig, load_system_config
# 统一从 types.py 导入所有 IR 类型
from .types import (
    EventType,
    StreamType,
    TimelineEvent,
    MemorySnapshot,
    StreamState,
    ScheduleIR,
    TimelineIR,
)

# Render module - Mixins
from .render import (
    GraphRenderMixin,
    ScheduleRenderMixin,
    TimelineRenderMixin,
    SimulationResultRenderMixin,
)


# ==============================================================================
# 动态绑定 Render Mixin 方法到 IR 类型
# ==============================================================================
def _bind_render_methods():
    """将 Render Mixin 的方法动态绑定到 IR 类型."""
    render_methods = ('to_terminal', 'to_tree_html', 'to_table', '_repr_html_')
    
    # 绑定 GraphIR 的渲染方法
    for name in render_methods:
        if hasattr(GraphRenderMixin, name):
            setattr(GraphIR, name, getattr(GraphRenderMixin, name))
    
    # 绑定 ScheduleIR 的渲染方法
    for name in render_methods:
        if hasattr(ScheduleRenderMixin, name):
            setattr(ScheduleIR, name, getattr(ScheduleRenderMixin, name))
    
    # 绑定 TimelineIR 的渲染方法
    for name in render_methods:
        if hasattr(TimelineRenderMixin, name):
            setattr(TimelineIR, name, getattr(TimelineRenderMixin, name))
    
    # 绑定 SimulationResult 的渲染方法
    for name in ('to_terminal', '_repr_html_'):
        if hasattr(SimulationResultRenderMixin, name):
            setattr(SimulationResult, name, getattr(SimulationResultRenderMixin, name))


# 模块加载时绑定渲染方法
try:
    _bind_render_methods()
except ImportError:
    # 如果 render 模块不可用，忽略
    pass

__all__ = [
    # Graph IR
    "NodeType",
    "TensorRef",
    "BlockNode",
    "GraphIR",
    # Schedule IR - hierarchical
    "TensorLifetime",
    "ScheduledOp",
    "DeviceSchedule",
    "StageSchedule",
    "ScheduleIR",
    # Timeline IR
    "TimelineIR",
    "TimelineEvent",
    "EventType",
    "StreamType",
    "MemorySnapshot",
    "StreamState",
    # Symbolic computation
    "SymMax",
    "SymMin",
    "sym_max",
    "sym_min",
    "eval_lazy",
    # Result
    "SimulationResult",
    # DSL
    "build_transformer_model",
    # Compiler
    "Compiler",
    # System Config
    "SystemConfig",
    "load_system_config",
    # Passes
    "Pass",
    "Pipeline",
    "ExpandPass",
    "ParallelPass",
    "SchedulePass",
    "TimelinePass",
    "TimelinePass",
    "SimulatePass",
    "OverlapAnalysisPass",
    "OptimizerPass",
    "OptimizerConfig",
    "PipelineSchedulePass",
    "PipelineConfig",
    "PPScheduleMode",
    "PrintGraphPass",
    "PrintSchedulePass",
    "PrintTimelinePass",
    "PrintResultPass",
    "SymbolicEstimatePass",
    # Estimate
    "SymbolicEstimate",
    # Render Mixins
    "GraphRenderMixin",
    "ScheduleRenderMixin",
    "TimelineRenderMixin",
    "SimulationResultRenderMixin",
]

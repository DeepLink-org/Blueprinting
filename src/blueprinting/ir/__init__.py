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

from .builder import IRBuilder, build_transformer_layer, build_transformer_model
from .compiler import Compiler
from .graph import BlockNode, GraphIR, ModuleNode, NodeType, OpNode, TensorRef
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
                      TimelinePass,
                      TimelinePassV2,
)
from .result import SimulationResult
from .schedule import DeviceSchedule, ScheduledOp, ScheduleIR, StageSchedule, TensorLifetime
from .symmax import SymMax, clear_expr_cache, eval_lazy, get_cache_stats, sym_max
from .system import SystemConfig, load_system_config
from .timeline import EventType, StreamType, TimelineEvent, TimelineIR

__all__ = [
    # Graph IR - hierarchical
    "NodeType",
    "TensorRef",
    "OpNode",
    "BlockNode",
    "ModuleNode",
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
    # Symbolic Max
    "SymMax",
    "sym_max",
    "eval_lazy",
    "clear_expr_cache",
    "get_cache_stats",
    # Result
    "SimulationResult",
    # Builder
    "IRBuilder",
    "build_transformer_layer",
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
    "TimelinePassV2",
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
]

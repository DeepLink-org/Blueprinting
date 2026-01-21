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

from .graph import (
    NodeType,
    TensorRef,
    OpNode,
    BlockNode,
    ModuleNode,
    GraphIR,
)
from .schedule import (
    TensorLifetime,
    ScheduledOp,
    DeviceSchedule,
    StageSchedule,
    ScheduleIR,
)
from .timeline import TimelineIR, TimelineEvent, EventType, StreamType
from .symmax import SymMax, sym_max, eval_lazy, clear_expr_cache, get_cache_stats
from .result import SimulationResult
from .builder import IRBuilder, build_transformer_layer, build_transformer_model
from .compiler import Compiler
from .system import SystemConfig, load_system_config
from .passes import (
    Pass,
    WorkloadPass,
    ParallelPass,
    SchedulePass,
    TimelinePass,
    OverlapAnalysisPass,
    EvaluatePass,
    OptimizerPass,
    OptimizerConfig,
)

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
    "WorkloadPass",
    "ParallelPass",
    "SchedulePass",
    "TimelinePass",
    "OverlapAnalysisPass",
    "EvaluatePass",
    "OptimizerPass",
    "OptimizerConfig",
]

"""IR module for blueprinting compiler."""

from .graph import OpNode, GraphIR, TensorRef
from .schedule import ScheduledOp, ScheduleIR
from .timeline import TimelineIR, TimelineEvent, EventType, StreamType
from .symmax import SymMax, sym_max, eval_lazy, clear_expr_cache, get_cache_stats
from .result import SimulationResult
from .builder import IRBuilder
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
    # Graph IR
    "OpNode",
    "GraphIR",
    "TensorRef",
    # Schedule IR
    "ScheduledOp",
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

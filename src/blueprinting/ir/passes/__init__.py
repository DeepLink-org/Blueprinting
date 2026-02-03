"""Compiler passes for IR transformation.

Pass 流程:

    GraphIR (Block)
         │
         ↓ ParallelPass (标记并行策略)
         │
         ↓ ExpandPass (Block → Op)
         │
    ScheduleIR (Op, 无 workload)
         │
         ↓ SchedulePass (计算 workload + timing)
         │
    ScheduleIR (Op, 有 workload)
         │
         ↓ OptimizerPass (追加反向 Op，训练时)
         │
         ↓ PipelineSchedulePass (PP 调度，PP>1 时)
         │
         ↓ TimelinePass (Op → Event)
         │
    TimelineIR (Event)
         │
         ↓ SimulatePass (评估)
         │
    SimulationResult
"""

from .base import Pass, Pipeline
from .debug import PrintGraphPass, PrintResultPass, PrintSchedulePass, PrintTimelinePass
from .expand import ExpandPass
from .optimizer import OptimizerConfig, OptimizerPass
from .overlap import OverlapAnalysisPass
from .parallel import ParallelPass
from .pipeline import PipelineConfig, PipelineSchedulePass, PPScheduleMode
from .schedule import SchedulePass
from .timeline_v2 import SimulatePass, TimelinePassV2

# Alias for backward compatibility
TimelinePass = TimelinePassV2

__all__ = [
    # Base
    "Pass",
    "Pipeline",
    # Debug
    "PrintGraphPass",
    "PrintSchedulePass",
    "PrintTimelinePass",
    "PrintResultPass",
    # Core passes
    "ExpandPass",
    "SchedulePass",
    "ParallelPass",
    "OptimizerPass",
    "OptimizerConfig",
    "PipelineSchedulePass",
    "PipelineConfig",
    "PPScheduleMode",
    "TimelinePass",
    "TimelinePassV2",
    "SimulatePass",
    "OverlapAnalysisPass",
]

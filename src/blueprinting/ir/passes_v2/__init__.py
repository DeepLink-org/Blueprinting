"""Passes v2 - New three-layer IR architecture passes.

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
from .debug import (
    PrintGraphPass,
    PrintSchedulePass,
    PrintTimelinePass,
    PrintResultPass,
)
from .expand import ExpandPass
from .schedule import SchedulePass
from .parallel import ParallelPass
from .optimizer import OptimizerPass, OptimizerConfig
from .pipeline import PipelineSchedulePass, PipelineConfig, PPScheduleMode
from .timeline_v2 import TimelinePassV2, SimulatePass
from .overlap import OverlapAnalysisPass

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
    "TimelinePassV2",
    "SimulatePass",
    "OverlapAnalysisPass",
]

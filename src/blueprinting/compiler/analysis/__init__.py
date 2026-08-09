"""Exact workload analyses and evidence-backed cost models."""

from .cost_model import (
    BlockEstimate,
    CalibrationMode,
    HardwareProfile,
    IterationEstimate,
    IterationMemory,
    estimate_block,
    estimate_iteration,
)
from .transformer_workload import (
    BlockMemoryFacts,
    EngineKind,
    PhaseWork,
    PrimitiveInvocation,
    TrainingPhase,
    compile_transformer_block,
)

__all__ = [
    "BlockEstimate",
    "BlockMemoryFacts",
    "CalibrationMode",
    "EngineKind",
    "HardwareProfile",
    "IterationEstimate",
    "IterationMemory",
    "PhaseWork",
    "PrimitiveInvocation",
    "TrainingPhase",
    "compile_transformer_block",
    "estimate_block",
    "estimate_iteration",
]

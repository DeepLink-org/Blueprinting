"""SimulationResult - Output of the IR compiler.

This module defines the result data structures returned
by the compiler after evaluation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union


@dataclass
class MemoryBreakdown:
    """Breakdown of memory usage.

    Attributes:
        weights: Weight tensor memory (per device)
        activations: Activation memory (peak)
        gradients: Gradient memory
        optimizer_states: Optimizer state memory (for Adam: 2*weights)
        total: Total memory usage
    """

    weights: Union[int, float] = 0
    activations: Union[int, float] = 0
    gradients: Union[int, float] = 0
    optimizer_states: Union[int, float] = 0

    @property
    def total(self) -> Union[int, float]:
        return self.weights + self.activations + self.gradients + self.optimizer_states

    def __repr__(self) -> str:
        return (
            f"MemoryBreakdown(weights={self.weights/1e9:.2f}GB, "
            f"activations={self.activations/1e9:.2f}GB, "
            f"gradients={self.gradients/1e9:.2f}GB, "
            f"optimizer={self.optimizer_states/1e9:.2f}GB, "
            f"total={self.total/1e9:.2f}GB)"
        )


@dataclass
class TimeBreakdown:
    """Breakdown of execution time.

    Attributes:
        forward: Forward pass time
        backward: Backward pass time
        optimizer: Optimizer step time
        communication: Communication time (exposed)
        bubble: Pipeline bubble time
        total: Total end-to-end time
    """

    forward: Union[int, float] = 0
    backward: Union[int, float] = 0
    optimizer: Union[int, float] = 0
    communication: Union[int, float] = 0
    bubble: Union[int, float] = 0

    @property
    def total(self) -> Union[int, float]:
        return (
            self.forward
            + self.backward
            + self.optimizer
            + self.communication
            + self.bubble
        )

    def __repr__(self) -> str:
        return (
            f"TimeBreakdown(forward={self.forward*1e3:.2f}ms, "
            f"backward={self.backward*1e3:.2f}ms, "
            f"optimizer={self.optimizer*1e3:.2f}ms, "
            f"comm={self.communication*1e3:.2f}ms, "
            f"bubble={self.bubble*1e3:.2f}ms, "
            f"total={self.total*1e3:.2f}ms)"
        )


@dataclass
class SimulationResult:
    """Result of compiling and simulating a model.

    Attributes:
        peak_memory: Peak memory usage in bytes
        e2e_time: End-to-end time in seconds
        memory_breakdown: Detailed memory breakdown
        time_breakdown: Detailed time breakdown

    Derived metrics:
        throughput: Tokens per second (if configured)
        mfu: Model FLOPs Utilization

    Metadata:
        config: Configuration used for simulation
        warnings: Any warnings generated
    """

    # Primary results
    peak_memory: Union[int, float] = 0
    e2e_time: Union[int, float] = 0

    # Detailed breakdowns
    memory_breakdown: Optional[MemoryBreakdown] = None
    time_breakdown: Optional[TimeBreakdown] = None

    # Derived metrics
    total_flops: Union[int, float] = 0
    achieved_flops: Union[int, float] = 0
    throughput: Optional[float] = None  # tokens/second

    # Metadata
    config: Dict[str, Any] = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    @property
    def mfu(self) -> float:
        """Model FLOPs Utilization.

        MFU = achieved_flops / peak_flops
        """
        if self.achieved_flops == 0 or self.e2e_time == 0:
            return 0.0
        peak_flops = self.config.get("peak_tflops", 0) * 1e12
        if peak_flops == 0:
            return 0.0
        return self.achieved_flops / (peak_flops * self.e2e_time)

    @property
    def memory_utilization(self) -> float:
        """Memory utilization ratio."""
        capacity = self.config.get("memory_capacity", 0)
        if capacity == 0:
            return 0.0
        return self.peak_memory / capacity

    def is_feasible(self) -> bool:
        """Check if the configuration fits in memory."""
        capacity = self.config.get("memory_capacity", float("inf"))
        return self.peak_memory <= capacity

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "peak_memory_bytes": self.peak_memory,
            "peak_memory_gb": self.peak_memory / 1e9,
            "e2e_time_seconds": self.e2e_time,
            "e2e_time_ms": self.e2e_time * 1e3,
            "total_flops": self.total_flops,
            "throughput": self.throughput,
            "mfu": self.mfu,
            "memory_utilization": self.memory_utilization,
            "is_feasible": self.is_feasible(),
            "memory_breakdown": (
                {
                    "weights_gb": (
                        self.memory_breakdown.weights / 1e9
                        if self.memory_breakdown
                        else 0
                    ),
                    "activations_gb": (
                        self.memory_breakdown.activations / 1e9
                        if self.memory_breakdown
                        else 0
                    ),
                    "gradients_gb": (
                        self.memory_breakdown.gradients / 1e9
                        if self.memory_breakdown
                        else 0
                    ),
                    "optimizer_gb": (
                        self.memory_breakdown.optimizer_states / 1e9
                        if self.memory_breakdown
                        else 0
                    ),
                }
                if self.memory_breakdown
                else None
            ),
            "time_breakdown": (
                {
                    "forward_ms": (
                        self.time_breakdown.forward * 1e3 if self.time_breakdown else 0
                    ),
                    "backward_ms": (
                        self.time_breakdown.backward * 1e3 if self.time_breakdown else 0
                    ),
                    "optimizer_ms": (
                        self.time_breakdown.optimizer * 1e3
                        if self.time_breakdown
                        else 0
                    ),
                    "communication_ms": (
                        self.time_breakdown.communication * 1e3
                        if self.time_breakdown
                        else 0
                    ),
                    "bubble_ms": (
                        self.time_breakdown.bubble * 1e3 if self.time_breakdown else 0
                    ),
                }
                if self.time_breakdown
                else None
            ),
            "warnings": self.warnings,
        }

    def __repr__(self) -> str:
        mem_gb = self.peak_memory / 1e9
        time_ms = self.e2e_time * 1e3
        return (
            f"SimulationResult(\n"
            f"  peak_memory={mem_gb:.2f} GB,\n"
            f"  e2e_time={time_ms:.2f} ms,\n"
            f"  mfu={self.mfu:.1%},\n"
            f"  feasible={self.is_feasible()}\n"
            f")"
        )

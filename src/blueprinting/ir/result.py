"""SimulationResult - Output of the IR compiler.

Defines result data structures returned by the compiler after evaluation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union


@dataclass
class MemoryBreakdown:
    """Breakdown of memory usage (per-GPU).

    Attributes:
        weights: Weight tensor memory
        activations: Activation memory (peak)
        gradients: Gradient memory
        optimizer_states: Optimizer state memory
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
        communication: Communication time (exposed)
        bubble: Pipeline bubble time
        recompute: Activation recomputation time (gradient checkpointing)
    """

    forward: Union[int, float] = 0
    backward: Union[int, float] = 0
    communication: Union[int, float] = 0
    bubble: Union[int, float] = 0
    recompute: Union[int, float] = 0

    @property
    def compute(self) -> Union[int, float]:
        """计算时间 = forward + backward + recompute."""
        return self.forward + self.backward + self.recompute

    @property
    def total(self) -> Union[int, float]:
        return (
            self.forward
            + self.backward
            + self.recompute
            + self.communication
            + self.bubble
        )

    def __repr__(self) -> str:
        return (
            f"TimeBreakdown(forward={self.forward*1e3:.2f}ms, "
            f"backward={self.backward*1e3:.2f}ms, "
            f"recompute={self.recompute*1e3:.2f}ms, "
            f"comm={self.communication*1e3:.2f}ms, "
            f"bubble={self.bubble*1e3:.2f}ms, "
            f"total={self.total*1e3:.2f}ms)"
        )


@dataclass
class BlockMetrics:
    """单层（Block）指标 - 单个 Transformer 层的内存和时间指标."""

    weights: Union[int, float] = 0
    activations: Union[int, float] = 0
    optimizer_states: Union[int, float] = 0

    forward_time: float = 0
    backward_time: float = 0
    communication_time: float = 0

    comm_fw: float = 0
    comm_bw: float = 0

    @property
    def compute_time(self) -> float:
        return self.forward_time + self.backward_time

    @property
    def total_time(self) -> float:
        return self.forward_time + self.backward_time + self.communication_time


@dataclass
class SimulationResult:
    """模拟结果 - 编译器最终输出.

    Attributes:
        peak_memory: 峰值内存 (bytes, per-GPU)
        e2e_time: 端到端时间 (seconds)
        memory_breakdown: 内存分解
        time_breakdown: 时间分解 (per-layer per-microbatch)
        total_time_breakdown: 总时间分解 (整个迭代)
        block_metrics: 单层指标
        total_flops: 总计算量
        config: 配置信息
        timeline: TimelineIR 引用 (用于导出 trace)
        estimate: SymbolicEstimate (可选)
    """

    peak_memory: Union[int, float] = 0
    e2e_time: Union[int, float] = 0
    memory_breakdown: Optional[MemoryBreakdown] = None
    time_breakdown: Optional[TimeBreakdown] = None
    total_time_breakdown: Optional[TimeBreakdown] = None
    block_metrics: Optional[BlockMetrics] = None
    total_flops: Union[int, float] = 0
    config: Dict[str, Any] = field(default_factory=dict)
    timeline: Any = None
    estimate: Any = None

    @property
    def mfu(self) -> float:
        """Model FLOPs Utilization."""
        peak_tflops = self.config.get("peak_tflops", 0)
        pp = self.config.get("pp", 1)
        if peak_tflops == 0 or self.e2e_time == 0:
            return 0
        per_gpu_flops = self.total_flops / pp if pp > 0 else self.total_flops
        return per_gpu_flops / (peak_tflops * 1e12 * self.e2e_time)

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
            "mfu": self.mfu,
            "memory_utilization": self.memory_utilization,
            "is_feasible": self.is_feasible(),
            "memory_breakdown": (
                {
                    "weights_gb": self.memory_breakdown.weights / 1e9,
                    "activations_gb": self.memory_breakdown.activations / 1e9,
                    "gradients_gb": self.memory_breakdown.gradients / 1e9,
                    "optimizer_gb": self.memory_breakdown.optimizer_states / 1e9,
                }
                if self.memory_breakdown
                else None
            ),
            "time_breakdown": (
                {
                    "forward_ms": self.time_breakdown.forward * 1e3,
                    "backward_ms": self.time_breakdown.backward * 1e3,
                    "recompute_ms": self.time_breakdown.recompute * 1e3,
                    "communication_ms": self.time_breakdown.communication * 1e3,
                    "bubble_ms": self.time_breakdown.bubble * 1e3,
                }
                if self.time_breakdown
                else None
            ),
        }

    def __repr__(self) -> str:
        return (
            f"SimulationResult(\n"
            f"  peak_memory={self.peak_memory/1e9:.2f} GB,\n"
            f"  e2e_time={self.e2e_time*1e3:.2f} ms,\n"
            f"  mfu={self.mfu:.1%}\n"
            f")"
        )

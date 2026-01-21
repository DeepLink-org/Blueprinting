"""ScheduleIR - Scheduling representation for IR compiler.

This module defines the scheduled execution representation,
including timing and memory lifecycle information.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from collections import Counter

from sympy import Expr, Symbol


from .symmax import sym_max, _to_float


def _lazy_max(a, b):
    """Max that works with both numeric and symbolic values.
    
    Uses SymMax for correct lazy evaluation.
    """
    return sym_max(a, b)


@dataclass
class TensorLifetime:
    """Tracks the lifetime of a tensor in the schedule.
    
    Attributes:
        tensor_id: ID of the tensor
        alloc_time: Time when tensor is allocated
        free_time: Time when tensor is freed
        size: Size of tensor in bytes
    """
    tensor_id: str
    alloc_time: Union[float, Expr]
    free_time: Union[float, Expr]
    size: Union[int, Expr]


@dataclass
class ScheduledOp:
    """A scheduled operation with timing information.
    
    Attributes:
        op_id: ID of the operation (from GraphIR)
        op_type: Type of operation
        device: Device assignment
        stream: Execution stream ("compute", "comm", "h2d", etc.)
        start: Start time (symbolic or numeric)
        duration: Duration (symbolic or numeric)
        
    Memory lifecycle:
        tensors_alloc: Tensors allocated by this op
        tensors_free: Tensors freed after this op
    """
    op_id: str
    op_type: str
    device: int = 0
    stream: str = "compute"
    start: Union[float, Expr] = 0
    duration: Union[float, Expr] = 0
    
    # Memory lifecycle
    tensors_alloc: List[TensorLifetime] = field(default_factory=list)
    tensors_free: List[str] = field(default_factory=list)
    
    # Additional attributes from GraphIR
    attrs: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def end(self) -> Union[float, Expr]:
        """End time of this operation."""
        return self.start + self.duration
    
    def __repr__(self) -> str:
        return f"ScheduledOp({self.op_id}, type={self.op_type}, dev={self.device}, start={self.start}, dur={self.duration})"


@dataclass
class ScheduleIR:
    """Scheduled execution representation.
    
    Attributes:
        ops: List of scheduled operations in execution order
        num_devices: Number of devices
        num_stages: Number of pipeline stages (for PP)
        tensors: Tensor lifetime tracking
        metadata: Additional metadata
    """
    ops: List[ScheduledOp] = field(default_factory=list)
    num_devices: int = 1
    num_stages: int = 1
    tensors: Dict[str, TensorLifetime] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_op(self, op: ScheduledOp) -> "ScheduleIR":
        """Add a scheduled operation."""
        self.ops.append(op)
        return self
    
    def get_ops_on_device(self, device: int) -> List[ScheduledOp]:
        """Get all operations on a specific device."""
        return [op for op in self.ops if op.device == device]
    
    def get_ops_on_stream(self, stream: str) -> List[ScheduledOp]:
        """Get all operations on a specific stream."""
        return [op for op in self.ops if op.stream == stream]
    
    def makespan(self, device: Optional[int] = None) -> Union[float, Expr]:
        """Calculate the total execution time (makespan).
        
        Args:
            device: If specified, only consider ops on this device
            
        Returns:
            Maximum end time across all (or device-specific) operations
        """
        if not self.ops:
            return 0
        
        ops = self.ops if device is None else self.get_ops_on_device(device)
        if not ops:
            return 0
        
        # Use lazy max for symbolic expressions (avoids SymPy's expensive simplification)
        end_times = [op.end for op in ops]
        if len(end_times) == 1:
            return end_times[0]
        
        # Check if all are numeric
        all_numeric = all(isinstance(t, (int, float)) for t in end_times)
        if all_numeric:
            return max(end_times)
        
        # Use LazyMax to avoid expensive simplification
        result = end_times[0]
        for t in end_times[1:]:
            result = _lazy_max(result, t)
        return result
    
    def peak_memory(self, device: int = 0) -> Union[int, Expr]:
        """Calculate peak memory usage on a device.
        
        This scans through the schedule and tracks tensor allocations
        and frees to find the maximum memory usage point.
        
        Args:
            device: Device to analyze
            
        Returns:
            Peak memory usage in bytes
        """
        # Collect all memory events
        events = []  # (time, delta_bytes, tensor_id)
        
        for op in self.get_ops_on_device(device):
            # Allocations at op start
            for tensor in op.tensors_alloc:
                events.append((tensor.alloc_time, tensor.size, f"+{tensor.tensor_id}"))
            
            # Frees at op end (or specified free time)
            for tensor in op.tensors_alloc:
                if tensor.free_time is not None:
                    events.append((tensor.free_time, -tensor.size, f"-{tensor.tensor_id}"))
        
        # Also check standalone tensor lifetimes
        for tensor in self.tensors.values():
            events.append((tensor.alloc_time, tensor.size, f"+{tensor.tensor_id}"))
            if tensor.free_time is not None:
                events.append((tensor.free_time, -tensor.size, f"-{tensor.tensor_id}"))
        
        if not events:
            return 0
        
        # Check if everything is numeric (both time and delta)
        all_numeric = all(
            isinstance(e[0], (int, float)) and isinstance(e[1], (int, float))
            for e in events
        )
        
        if all_numeric:
            # Sort by time and scan for peak
            events.sort(key=lambda e: e[0])
            current = 0
            peak = 0
            for _, delta, _ in events:
                current += delta
                peak = max(peak, current)
            return peak
        else:
            # For symbolic: sum all allocation sizes as upper bound
            # (conservative estimate - ignores frees)
            total_alloc = 0
            for _, delta, tag in events:
                # Only add allocations (tags starting with +)
                if tag.startswith("+"):
                    if isinstance(delta, (int, float)):
                        total_alloc += delta
                    else:
                        total_alloc = total_alloc + delta
            return total_alloc
    
    def total_compute_time(self, device: int = 0) -> Union[float, Expr]:
        """Sum of all compute operation durations on a device."""
        total = 0
        for op in self.get_ops_on_device(device):
            if op.stream == "compute":
                total = total + op.duration
        return total
    
    def total_comm_time(self, device: int = 0) -> Union[float, Expr]:
        """Sum of all communication operation durations on a device."""
        total = 0
        for op in self.get_ops_on_device(device):
            if op.stream in ("comm", "nccl"):
                total = total + op.duration
        return total
    
    def __repr__(self) -> str:
        parts = [
            f"ops={len(self.ops)}",
            f"devices={self.num_devices}",
            f"stages={self.num_stages}",
            f"tensors={len(self.tensors)}",
        ]
        op_types = Counter(op.op_type for op in self.ops)
        if op_types:
            top = ", ".join(f"{k}:{v}" for k, v in op_types.most_common(5))
            parts.append(f"op_types={{ {top} }}")
        meta_keys = list(self.metadata.keys())
        if meta_keys:
            sample = ", ".join(meta_keys[:6])
            suffix = "..." if len(meta_keys) > 6 else ""
            parts.append(f"metadata_keys=[{sample}{suffix}]")
        return f"ScheduleIR({', '.join(parts)})"

    def summary(self) -> str:
        """Return a readable multi-line summary of the schedule."""
        op_types = Counter(op.op_type for op in self.ops)
        top_ops = "\n".join(
            f"  - {k}: {v}" for k, v in op_types.most_common(8)
        ) if op_types else "  (none)"
        meta_keys = list(self.metadata.keys())
        meta_line = ", ".join(meta_keys) if meta_keys else "(none)"
        return (
            "ScheduleIR Summary\n"
            f"- ops: {len(self.ops)}\n"
            f"- devices: {self.num_devices}\n"
            f"- stages: {self.num_stages}\n"
            f"- tensors: {len(self.tensors)}\n"
            f"- op_types:\n{top_ops}\n"
            f"- metadata_keys: {meta_line}"
        )

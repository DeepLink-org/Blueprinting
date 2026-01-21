"""ScheduleIR - Hierarchical schedule representation for IR compiler.

This module defines a hierarchical schedule structure:
  Stage → Device → ScheduledOp

The schedule represents the execution-side view of the computation:
- Each stage groups operations for a pipeline stage
- Each device holds ops assigned to that device
- Each op has a sequence number (event_seq) for timeline ordering

When ops are flattened and sorted by event_seq, we get TimelineIR.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Iterator
from collections import Counter

from sympy import Expr, Symbol

from .symmax import sym_max


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
    alloc_time: Union[float, Expr] = 0
    free_time: Optional[Union[float, Expr]] = None
    size: Union[int, Expr] = 0


@dataclass
class ScheduledOp:
    """A scheduled operation with timing information.
    
    Attributes:
        name: Op name (from GraphIR)
        op_path: Full path in the GraphIR hierarchy (e.g., "model.layer0.attn.q_proj")
        op_type: Type of operation
        
    Scheduling:
        device: Device assignment
        stage: Pipeline stage
        stream: Execution stream ("compute", "comm", "h2d", etc.)
        event_seq: Global event sequence number for timeline ordering
        
    Timing:
        start: Start time (symbolic or numeric)
        duration: Duration (symbolic or numeric)
        
    Memory:
        tensors_alloc: Tensors allocated by this op
        tensors_free: Tensors freed after this op
        
    Workload (from GraphIR):
        flops: Total FLOPs
        memory_bytes: Memory accessed
        comm_bytes: Communication bytes
    """
    name: str
    op_path: str = ""
    op_type: str = ""
    
    # Scheduling
    device: int = 0
    stage: int = 0
    stream: str = "compute"
    event_seq: int = 0
    
    # Timing
    start: Union[float, Expr] = 0
    duration: Union[float, Expr] = 0
    
    # Memory lifecycle
    tensors_alloc: List[TensorLifetime] = field(default_factory=list)
    tensors_free: List[str] = field(default_factory=list)
    
    # Workload
    flops: Optional[Expr] = None
    memory_bytes: Optional[Expr] = None
    comm_bytes: Optional[Expr] = None
    
    # Additional attributes
    attrs: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def end(self) -> Union[float, Expr]:
        """End time of this operation."""
        return self.start + self.duration
    
    def __repr__(self) -> str:
        parts = [f"{self.op_type}({self.name!r})"]
        parts.append(f"seq={self.event_seq}")
        parts.append(f"dev={self.device}")
        if self.stage > 0:
            parts.append(f"stage={self.stage}")
        return f"ScheduledOp({', '.join(parts)})"


@dataclass
class DeviceSchedule:
    """Schedule for a single device.
    
    Contains all operations assigned to this device, organized by stream.
    """
    device_id: int
    ops: List[ScheduledOp] = field(default_factory=list)
    
    def add_op(self, op: ScheduledOp) -> "DeviceSchedule":
        """Add an operation to this device."""
        op.device = self.device_id
        self.ops.append(op)
        return self
    
    def get_ops_on_stream(self, stream: str) -> List[ScheduledOp]:
        """Get operations on a specific stream."""
        return [op for op in self.ops if op.stream == stream]
    
    def iter_ops(self) -> Iterator[ScheduledOp]:
        """Iterate over all ops."""
        return iter(self.ops)
    
    @property
    def compute_ops(self) -> List[ScheduledOp]:
        """Get compute operations."""
        return self.get_ops_on_stream("compute")
    
    @property
    def comm_ops(self) -> List[ScheduledOp]:
        """Get communication operations."""
        return [op for op in self.ops if op.stream in ("comm", "nccl")]
    
    def makespan(self) -> Union[float, Expr]:
        """Get the total execution time on this device."""
        if not self.ops:
            return 0
        end_times = [op.end for op in self.ops]
        if len(end_times) == 1:
            return end_times[0]
        
        all_numeric = all(isinstance(t, (int, float)) for t in end_times)
        if all_numeric:
            return max(end_times)
        
        result = end_times[0]
        for t in end_times[1:]:
            result = sym_max(result, t)
        return result
    
    def total_compute_time(self) -> Union[float, Expr]:
        """Sum of compute durations."""
        total = 0
        for op in self.compute_ops:
            total = total + op.duration
        return total
    
    def total_comm_time(self) -> Union[float, Expr]:
        """Sum of communication durations."""
        total = 0
        for op in self.comm_ops:
            total = total + op.duration
        return total
    
    def __repr__(self) -> str:
        return f"DeviceSchedule(device={self.device_id}, ops={len(self.ops)})"


@dataclass
class StageSchedule:
    """Schedule for a pipeline stage.
    
    A stage contains multiple devices executing the same set of layers.
    """
    stage_id: int
    devices: Dict[int, DeviceSchedule] = field(default_factory=dict)
    layers: List[int] = field(default_factory=list)  # Layer indices in this stage
    
    def get_device(self, device_id: int) -> DeviceSchedule:
        """Get or create a device schedule."""
        if device_id not in self.devices:
            self.devices[device_id] = DeviceSchedule(device_id=device_id)
        return self.devices[device_id]
    
    def add_op(self, op: ScheduledOp, device_id: int) -> "StageSchedule":
        """Add an operation to a device in this stage."""
        op.stage = self.stage_id
        self.get_device(device_id).add_op(op)
        return self
    
    def iter_ops(self) -> Iterator[ScheduledOp]:
        """Iterate over all ops in this stage."""
        for device in self.devices.values():
            yield from device.iter_ops()
    
    def iter_devices(self) -> Iterator[DeviceSchedule]:
        """Iterate over all devices in this stage."""
        return iter(self.devices.values())
    
    def __repr__(self) -> str:
        op_count = sum(len(d.ops) for d in self.devices.values())
        return f"StageSchedule(stage={self.stage_id}, devices={len(self.devices)}, ops={op_count})"


@dataclass
class ScheduleIR:
    """Hierarchical schedule representation.
    
    Structure: Stage → Device → ScheduledOp
    
    This represents the execution-side view of computation,
    with operations organized by pipeline stage and device assignment.
    
    Attributes:
        stages: Dictionary of pipeline stages by stage_id
        num_devices: Total number of devices
        tensors: Global tensor lifetime tracking
        metadata: Schedule-level metadata
        
    Timeline conversion:
        To get TimelineIR, flatten all ops and sort by event_seq.
    """
    stages: Dict[int, StageSchedule] = field(default_factory=dict)
    num_devices: int = 1
    tensors: Dict[str, TensorLifetime] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Event sequence counter for timeline ordering
    _event_counter: int = 0
    
    # === Stage management ===
    
    def get_stage(self, stage_id: int) -> StageSchedule:
        """Get or create a stage."""
        if stage_id not in self.stages:
            self.stages[stage_id] = StageSchedule(stage_id=stage_id)
        return self.stages[stage_id]
    
    @property
    def num_stages(self) -> int:
        """Number of pipeline stages."""
        return len(self.stages) if self.stages else 1
    
    # === Op management ===
    
    def add_op(self, op: ScheduledOp, stage_id: int = 0, device_id: int = 0) -> ScheduledOp:
        """Add an operation to the schedule.
        
        Automatically assigns event_seq for timeline ordering.
        """
        op.event_seq = self._event_counter
        self._event_counter += 1
        self.get_stage(stage_id).add_op(op, device_id)
        return op
    
    def schedule_op(self, name: str, op_type: str, device: int = 0, stage: int = 0,
                    stream: str = "compute", **kwargs) -> ScheduledOp:
        """Create and add a scheduled operation."""
        op = ScheduledOp(
            name=name,
            op_type=op_type,
            device=device,
            stage=stage,
            stream=stream,
            **kwargs
        )
        return self.add_op(op, stage, device)
    
    # === Traversal ===
    
    def iter_stages(self) -> Iterator[StageSchedule]:
        """Iterate over all stages."""
        for stage_id in sorted(self.stages.keys()):
            yield self.stages[stage_id]
    
    def iter_devices(self) -> Iterator[DeviceSchedule]:
        """Iterate over all devices across all stages."""
        for stage in self.iter_stages():
            yield from stage.iter_devices()
    
    def iter_ops(self) -> Iterator[ScheduledOp]:
        """Iterate over all operations (across all stages and devices)."""
        for stage in self.iter_stages():
            yield from stage.iter_ops()
    
    def iter_ops_sorted(self) -> Iterator[ScheduledOp]:
        """Iterate over all operations sorted by event_seq (timeline order)."""
        ops = list(self.iter_ops())
        ops.sort(key=lambda op: op.event_seq)
        return iter(ops)
    
    def get_ops_on_device(self, device_id: int) -> List[ScheduledOp]:
        """Get all operations on a specific device."""
        ops = []
        for stage in self.stages.values():
            if device_id in stage.devices:
                ops.extend(stage.devices[device_id].ops)
        return ops
    
    def get_ops_on_stream(self, stream: str) -> List[ScheduledOp]:
        """Get all operations on a specific stream."""
        return [op for op in self.iter_ops() if op.stream == stream]
    
    # === Metrics ===
    
    @property
    def ops(self) -> List[ScheduledOp]:
        """Legacy: flat list of all ops."""
        return list(self.iter_ops())
    
    def makespan(self, device: Optional[int] = None) -> Union[float, Expr]:
        """Calculate total execution time.
        
        Args:
            device: If specified, only consider ops on this device
        """
        if device is not None:
            ops = self.get_ops_on_device(device)
        else:
            ops = list(self.iter_ops())
        
        if not ops:
            return 0
        
        end_times = [op.end for op in ops]
        if len(end_times) == 1:
            return end_times[0]
        
        all_numeric = all(isinstance(t, (int, float)) for t in end_times)
        if all_numeric:
            return max(end_times)
        
        result = end_times[0]
        for t in end_times[1:]:
            result = sym_max(result, t)
        return result
    
    def peak_memory(self, device: int = 0) -> Union[int, Expr]:
        """Calculate peak memory usage on a device."""
        events = []
        
        for op in self.get_ops_on_device(device):
            for tensor in op.tensors_alloc:
                events.append((tensor.alloc_time, tensor.size, f"+{tensor.tensor_id}"))
            for tensor in op.tensors_alloc:
                if tensor.free_time is not None:
                    events.append((tensor.free_time, -tensor.size, f"-{tensor.tensor_id}"))
        
        for tensor in self.tensors.values():
            events.append((tensor.alloc_time, tensor.size, f"+{tensor.tensor_id}"))
            if tensor.free_time is not None:
                events.append((tensor.free_time, -tensor.size, f"-{tensor.tensor_id}"))
        
        if not events:
            return 0
        
        all_numeric = all(
            isinstance(e[0], (int, float)) and isinstance(e[1], (int, float))
            for e in events
        )
        
        if all_numeric:
            events.sort(key=lambda e: e[0])
            current = 0
            peak = 0
            for _, delta, _ in events:
                current += delta
                peak = max(peak, current)
            return peak
        else:
            total_alloc = 0
            for _, delta, tag in events:
                if tag.startswith("+"):
                    total_alloc = total_alloc + delta if isinstance(delta, Expr) else total_alloc + delta
            return total_alloc
    
    def total_compute_time(self, device: int = 0) -> Union[float, Expr]:
        """Sum of compute durations on a device."""
        total = 0
        for op in self.get_ops_on_device(device):
            if op.stream == "compute":
                total = total + op.duration
        return total
    
    def total_comm_time(self, device: int = 0) -> Union[float, Expr]:
        """Sum of communication durations on a device."""
        total = 0
        for op in self.get_ops_on_device(device):
            if op.stream in ("comm", "nccl"):
                total = total + op.duration
        return total
    
    # === Display ===
    
    def __repr__(self) -> str:
        op_count = sum(1 for _ in self.iter_ops())
        parts = [
            f"stages={self.num_stages}",
            f"devices={self.num_devices}",
            f"ops={op_count}",
        ]
        op_types = Counter(op.op_type for op in self.iter_ops())
        if op_types:
            top = ", ".join(f"{k}:{v}" for k, v in op_types.most_common(4))
            parts.append(f"types={{ {top} }}")
        return f"ScheduleIR({', '.join(parts)})"
    
    def summary(self) -> str:
        """Return a readable multi-line summary."""
        lines = [
            f"ScheduleIR: {self.num_stages} stages, {self.num_devices} devices",
        ]
        
        # Per-stage summary
        lines.append("├─ stages:")
        for stage in self.iter_stages():
            op_count = sum(1 for _ in stage.iter_ops())
            dev_count = len(stage.devices)
            lines.append(f"│   stage{stage.stage_id}: {dev_count} devices, {op_count} ops")
        
        # Op type breakdown
        op_types = Counter(op.op_type for op in self.iter_ops())
        if op_types:
            lines.append("├─ op_types:")
            for k, v in op_types.most_common(6):
                lines.append(f"│   {k}: {v}")
        
        # Metadata
        meta = ", ".join(list(self.metadata.keys())[:6])
        lines.append(f"└─ metadata: {meta or '(none)'}")
        
        return "\n".join(lines)
    
    def tree(self, max_ops: int = 5) -> str:
        """Return a tree representation of the schedule hierarchy."""
        lines = [f"■ ScheduleIR ({self.num_stages} stages)"]
        
        for stage in self.iter_stages():
            lines.append(f"├─ □ Stage {stage.stage_id}")
            devices = list(stage.devices.values())
            
            for i, device in enumerate(devices):
                is_last_dev = (i == len(devices) - 1)
                dev_prefix = "│  └─" if is_last_dev else "│  ├─"
                lines.append(f"{dev_prefix} ◇ Device {device.device_id} ({len(device.ops)} ops)")
                
                # Show first few ops
                ops_to_show = device.ops[:max_ops]
                op_prefix = "│     " if is_last_dev else "│  │  "
                
                for j, op in enumerate(ops_to_show):
                    is_last_op = (j == len(ops_to_show) - 1 and len(device.ops) <= max_ops)
                    marker = "└─" if is_last_op else "├─"
                    lines.append(f"{op_prefix}{marker} ◆ {op.op_type}({op.name}) seq={op.event_seq}")
                
                if len(device.ops) > max_ops:
                    lines.append(f"{op_prefix}└─ ... ({len(device.ops) - max_ops} more)")
        
        return "\n".join(lines)

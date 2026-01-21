"""TimelinePass - Generate TimelineIR from ScheduleIR.

This pass converts the scheduled operations into a fine-grained
event stream that enables precise memory and timing analysis.
"""

from typing import Dict, List, Optional, Tuple, Union

from sympy import Expr, Symbol

from ..schedule import ScheduleIR, ScheduledOp, TensorLifetime
from ..timeline import (
    TimelineIR,
    TimelineEvent,
    EventType,
    StreamType,
)
from .base import Pass


class TimelinePass(Pass):
    """Pass to generate TimelineIR from ScheduleIR.
    
    This pass creates an event stream representation that captures:
    - Memory allocation and deallocation events
    - Compute operation start/end events
    - Communication operation start/end events
    - Stream synchronization points
    
    The resulting TimelineIR can be used for:
    - Precise peak memory calculation
    - Overlap analysis
    - Bubble time identification
    - Chrome trace export for visualization
    """
    
    def __init__(
        self,
        model_overlap: bool = True,
        memory_model: str = "eager",  # "eager" or "lazy"
        include_optimizer: bool = True,
    ):
        """Initialize TimelinePass.
        
        Args:
            model_overlap: Whether to model compute/communication overlap
            memory_model: Memory allocation strategy
                - "eager": Allocate at op start, free at last use
                - "lazy": Allocate on demand, free immediately after use
            include_optimizer: Whether to include optimizer state memory
        """
        self.model_overlap = model_overlap
        self.memory_model = memory_model
        self.include_optimizer = include_optimizer
    
    @property
    def name(self) -> str:
        return "TimelinePass"
    
    def run(self, schedule: ScheduleIR) -> TimelineIR:
        """Convert ScheduleIR to TimelineIR.
        
        Args:
            schedule: Scheduled operations with timing information
            
        Returns:
            TimelineIR with fine-grained event stream
        """
        timeline = TimelineIR(num_devices=schedule.num_devices)
        
        # Copy metadata
        timeline.metadata = dict(schedule.metadata)
        timeline.metadata["memory_model"] = self.memory_model
        timeline.metadata["model_overlap"] = self.model_overlap
        
        # Process each scheduled operation
        for op in schedule.iter_ops():
            self._add_op_events(timeline, op, schedule)
        
        # Add tensor lifetime events
        self._add_tensor_events(timeline, schedule)
        
        return timeline
    
    def _add_op_events(
        self,
        timeline: TimelineIR,
        op: ScheduledOp,
        schedule: ScheduleIR,
    ) -> None:
        """Add compute/communication events for an operation."""
        
        # Determine stream type based on operation type
        if op.op_type in ("AllReduce", "AllGather", "ReduceScatter", "P2P"):
            stream = StreamType.NCCL
            start_type = EventType.COMM_START
            end_type = EventType.COMM_END
        else:
            stream = StreamType.COMPUTE
            start_type = EventType.COMPUTE_START
            end_type = EventType.COMPUTE_END
        
        # Calculate end time
        end_time = op.end
        
        # Get op_id (use op_path for hierarchical, name for legacy)
        op_id = op.op_path or op.name
        
        # Add start event
        timeline.add_event(TimelineEvent(
            time=op.start,
            event_type=start_type,
            resource_id=op_id,
            device=op.device,
            stream=stream,
            op_type=op.op_type,
            metadata={
                "flops": op.attrs.get("flops_fw", 0),
                "memory": op.attrs.get("memory_fw", 0),
            }
        ))
        
        # Add end event
        timeline.add_event(TimelineEvent(
            time=end_time,
            event_type=end_type,
            resource_id=op_id,
            device=op.device,
            stream=stream,
            op_type=op.op_type,
        ))
        
        # Add memory events for tensors allocated by this op
        for tensor in op.tensors_alloc:
            # Allocation at op start
            timeline.add_event(TimelineEvent(
                time=op.start,
                event_type=EventType.ALLOC,
                resource_id=tensor.tensor_id,
                device=op.device,
                stream=StreamType.MEMORY,
                size=tensor.size,
                metadata={"producer_op": op_id}
            ))
            
            # Free at tensor's free time (if specified)
            if tensor.free_time is not None:
                timeline.add_event(TimelineEvent(
                    time=tensor.free_time,
                    event_type=EventType.FREE,
                    resource_id=tensor.tensor_id,
                    device=op.device,
                    stream=StreamType.MEMORY,
                    size=tensor.size,
                ))
    
    def _add_tensor_events(
        self,
        timeline: TimelineIR,
        schedule: ScheduleIR,
    ) -> None:
        """Add memory events from standalone tensor lifetimes."""
        
        # Track which tensors we've already added events for
        seen_tensors = set()
        for op in schedule.iter_ops():
            for tensor in op.tensors_alloc:
                seen_tensors.add(tensor.tensor_id)
        
        # Add events for tensors in schedule.tensors that weren't in ops
        for tensor_id, lifetime in schedule.tensors.items():
            if tensor_id in seen_tensors:
                continue
            
            # Determine device (default to 0 if not specified)
            device = lifetime.device if hasattr(lifetime, 'device') else 0
            
            # Allocation event
            timeline.add_event(TimelineEvent(
                time=lifetime.alloc_time,
                event_type=EventType.ALLOC,
                resource_id=tensor_id,
                device=device,
                stream=StreamType.MEMORY,
                size=lifetime.size,
            ))
            
            # Free event
            if lifetime.free_time is not None:
                timeline.add_event(TimelineEvent(
                    time=lifetime.free_time,
                    event_type=EventType.FREE,
                    resource_id=tensor_id,
                    device=device,
                    stream=StreamType.MEMORY,
                    size=lifetime.size,
                ))


class OverlapAnalysisPass(Pass):
    """Pass to analyze and model compute-communication overlap.
    
    This pass examines the TimelineIR and adjusts event timings
    to accurately model overlap between streams.
    """
    
    def __init__(
        self,
        overlap_efficiency: float = 0.9,  # Assume 90% overlap efficiency
    ):
        self.overlap_efficiency = overlap_efficiency
    
    @property
    def name(self) -> str:
        return "OverlapAnalysisPass"
    
    def run(self, timeline: TimelineIR) -> TimelineIR:
        """Analyze and adjust for compute-communication overlap.
        
        Currently, this pass analyzes the potential overlap but
        doesn't modify the timeline. Future versions could adjust
        timings based on stream dependencies.
        """
        # Analyze overlap potential
        for dev in range(timeline.num_devices):
            compute_intervals = self._get_intervals(
                timeline, dev, StreamType.COMPUTE
            )
            comm_intervals = self._get_intervals(
                timeline, dev, StreamType.NCCL
            )
            
            overlap = self._calculate_overlap(compute_intervals, comm_intervals)
            
            # Store analysis results in metadata
            if "overlap_analysis" not in timeline.metadata:
                timeline.metadata["overlap_analysis"] = {}
            
            timeline.metadata["overlap_analysis"][dev] = {
                "compute_time": sum(i[1] - i[0] for i in compute_intervals),
                "comm_time": sum(i[1] - i[0] for i in comm_intervals),
                "overlap_time": overlap,
                "efficiency": self.overlap_efficiency,
            }
        
        return timeline
    
    def _get_intervals(
        self,
        timeline: TimelineIR,
        device: int,
        stream: StreamType,
    ) -> List[Tuple[float, float]]:
        """Extract time intervals for operations on a stream."""
        events = timeline.get_events(device=device, stream=stream)
        
        intervals = []
        starts = {}
        
        for event in events:
            if not isinstance(event.time, (int, float)):
                continue
                
            if event.event_type in (EventType.COMPUTE_START, EventType.COMM_START):
                starts[event.resource_id] = event.time
            elif event.event_type in (EventType.COMPUTE_END, EventType.COMM_END):
                if event.resource_id in starts:
                    intervals.append((starts[event.resource_id], event.time))
                    del starts[event.resource_id]
        
        return sorted(intervals)
    
    def _calculate_overlap(
        self,
        intervals1: List[Tuple[float, float]],
        intervals2: List[Tuple[float, float]],
    ) -> float:
        """Calculate total overlap time between two sets of intervals."""
        overlap = 0.0
        
        for start1, end1 in intervals1:
            for start2, end2 in intervals2:
                # Check if intervals overlap
                overlap_start = max(start1, start2)
                overlap_end = min(end1, end2)
                if overlap_start < overlap_end:
                    overlap += overlap_end - overlap_start
        
        return overlap

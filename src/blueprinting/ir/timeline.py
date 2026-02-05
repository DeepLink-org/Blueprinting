"""TimelineIR - Event-based timeline representation for precise simulation.

This module provides a fine-grained event stream representation that enables:
- Precise peak memory calculation by tracking alloc/free events
- Accurate end-to-end time with compute/communication overlap
- Multi-stream simulation (compute stream, NCCL stream)
- Detailed profiling and analysis
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from sympy import Expr


class EventType(Enum):
    """Types of events in the timeline."""

    # Memory events
    ALLOC = "alloc"  # Tensor allocation
    FREE = "free"  # Tensor deallocation

    # Compute events
    COMPUTE_START = "compute_start"
    COMPUTE_END = "compute_end"

    # Communication events
    COMM_START = "comm_start"
    COMM_END = "comm_end"

    # Synchronization events
    SYNC = "sync"  # Stream synchronization
    BARRIER = "barrier"  # Multi-device barrier


class StreamType(Enum):
    """Types of execution streams."""

    COMPUTE = "compute"  # Main compute stream (CUDA default stream)
    NCCL = "nccl"  # NCCL communication stream
    MEMORY = "memory"  # Memory operations (async memcpy)
    H2D = "h2d"  # Host to device transfer
    D2H = "d2h"  # Device to host transfer


@dataclass
class TimelineEvent:
    """A single event in the timeline.

    Events are the atomic units of the timeline, representing
    instantaneous state changes (alloc, free) or the start/end
    of operations (compute, communication).

    Attributes:
        time: Event timestamp (can be symbolic)
        event_type: Type of event
        resource_id: Associated tensor_id or op_id
        device: Device index
        stream: Execution stream
        size: Bytes for memory events, 0 otherwise
        op_type: Operation type for compute/comm events
        metadata: Additional event-specific data
    """

    time: Union[float, Expr]
    event_type: EventType
    resource_id: str
    device: int = 0
    stream: StreamType = StreamType.COMPUTE
    size: Union[int, Expr] = 0
    op_type: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other: "TimelineEvent") -> bool:
        """Compare events by time for sorting."""
        if isinstance(self.time, (int, float)) and isinstance(other.time, (int, float)):
            return self.time < other.time
        # For symbolic times, use string representation for stable sorting
        return str(self.time) < str(other.time)

    def is_memory_event(self) -> bool:
        return self.event_type in (EventType.ALLOC, EventType.FREE)

    def is_compute_event(self) -> bool:
        return self.event_type in (EventType.COMPUTE_START, EventType.COMPUTE_END)

    def is_comm_event(self) -> bool:
        return self.event_type in (EventType.COMM_START, EventType.COMM_END)


@dataclass
class MemorySnapshot:
    """Memory state at a point in time.

    Used to track memory usage over time and find peaks.
    """

    time: Union[float, Expr]
    device: int
    allocated: Union[int, Expr]  # Total allocated bytes
    tensors: List[str]  # List of live tensor IDs

    def __repr__(self) -> str:
        if isinstance(self.allocated, (int, float)):
            return f"MemorySnapshot(t={self.time:.6f}, dev={self.device}, alloc={self.allocated/1e9:.2f}GB)"
        return (
            f"MemorySnapshot(t={self.time}, dev={self.device}, alloc={self.allocated})"
        )


@dataclass
class StreamState:
    """State of an execution stream at a point in time."""

    device: int
    stream: StreamType
    current_time: Union[float, Expr] = 0
    pending_ops: List[str] = field(default_factory=list)


class TimelineIR:
    """Event-based timeline representation.

    TimelineIR provides a precise simulation model by representing
    execution as a sequence of events on multiple streams. This enables:

    1. Precise memory tracking: By scanning alloc/free events
    2. Overlap modeling: By tracking multiple streams per device
    3. Bubble analysis: By identifying idle periods
    4. Critical path analysis: By tracing dependencies

    Example:
        timeline = TimelineIR(num_devices=2)

        # Add memory allocation
        timeline.add_event(TimelineEvent(
            time=0.0,
            event_type=EventType.ALLOC,
            resource_id="activation_0",
            device=0,
            size=1024 * 1024
        ))

        # Add compute operation
        timeline.add_event(TimelineEvent(
            time=0.0,
            event_type=EventType.COMPUTE_START,
            resource_id="linear_0",
            device=0,
            op_type="Linear"
        ))

        # Query metrics
        peak_mem = timeline.peak_memory(device=0)
        makespan = timeline.makespan()
    """

    def __init__(self, num_devices: int = 1):
        self.num_devices = num_devices
        self.events: List[TimelineEvent] = []
        self._sorted = True

        # Metadata
        self.metadata: Dict[str, Any] = {}

        # Stream states per device
        self.stream_states: Dict[Tuple[int, StreamType], StreamState] = {}
        for dev in range(num_devices):
            for stream in StreamType:
                self.stream_states[(dev, stream)] = StreamState(dev, stream)

    def add_event(self, event: TimelineEvent) -> None:
        """Add an event to the timeline."""
        self.events.append(event)
        self._sorted = False

    def add_events(self, events: List[TimelineEvent]) -> None:
        """Add multiple events to the timeline."""
        self.events.extend(events)
        self._sorted = False

    def _ensure_sorted(self) -> None:
        """Ensure events are sorted by time."""
        if not self._sorted:
            # Check if all times are numeric
            all_numeric = all(isinstance(e.time, (int, float)) for e in self.events)
            if all_numeric:
                self.events.sort(key=lambda e: e.time)
            else:
                # For mixed/symbolic, sort by (numeric_time, str_time) for stability
                def sort_key(e):
                    if isinstance(e.time, (int, float)):
                        return (e.time, "")
                    return (float("inf"), str(e.time))

                self.events.sort(key=sort_key)
            self._sorted = True

    def get_events(
        self,
        device: Optional[int] = None,
        stream: Optional[StreamType] = None,
        event_type: Optional[EventType] = None,
    ) -> List[TimelineEvent]:
        """Filter events by device, stream, and/or type."""
        self._ensure_sorted()
        result = self.events

        if device is not None:
            result = [e for e in result if e.device == device]
        if stream is not None:
            result = [e for e in result if e.stream == stream]
        if event_type is not None:
            result = [e for e in result if e.event_type == event_type]

        return result

    def peak_memory(self, device: int = 0) -> Union[int, Expr]:
        """Calculate peak memory usage on a device.

        Scans through alloc/free events to find the maximum
        memory usage at any point in time.

        Args:
            device: Device to analyze

        Returns:
            Peak memory in bytes (int or symbolic Expr)
        """
        self._ensure_sorted()

        memory_events = [
            e for e in self.events if e.device == device and e.is_memory_event()
        ]

        if not memory_events:
            return 0

        # Check if all sizes are numeric
        all_numeric = all(isinstance(e.size, (int, float)) for e in memory_events)

        if all_numeric:
            current = 0
            peak = 0
            for event in memory_events:
                if event.event_type == EventType.ALLOC:
                    current += event.size
                    peak = max(peak, current)
                elif event.event_type == EventType.FREE:
                    current -= event.size
            return peak
        else:
            # Symbolic: sum all allocations as upper bound
            total = 0
            for event in memory_events:
                if event.event_type == EventType.ALLOC:
                    total = total + event.size
            return total

    def memory_trace(self, device: int = 0) -> List[MemorySnapshot]:
        """Generate memory usage trace over time.

        Returns a list of snapshots showing memory usage at each
        memory event. Useful for visualization and debugging.
        """
        self._ensure_sorted()

        memory_events = [
            e for e in self.events if e.device == device and e.is_memory_event()
        ]

        snapshots = []
        current = 0
        live_tensors = []

        for event in memory_events:
            if event.event_type == EventType.ALLOC:
                current += event.size if isinstance(event.size, (int, float)) else 0
                live_tensors.append(event.resource_id)
            elif event.event_type == EventType.FREE:
                current -= event.size if isinstance(event.size, (int, float)) else 0
                if event.resource_id in live_tensors:
                    live_tensors.remove(event.resource_id)

            snapshots.append(
                MemorySnapshot(
                    time=event.time,
                    device=device,
                    allocated=current,
                    tensors=list(live_tensors),
                )
            )

        return snapshots

    def makespan(self, device: Optional[int] = None) -> Union[float, Expr]:
        """Calculate the total execution time (makespan).

        The makespan is the time from the first event to the last
        event completion across all (or specified) devices.

        Args:
            device: Specific device, or None for all devices

        Returns:
            Total execution time
        """
        self._ensure_sorted()

        events = (
            self.events
            if device is None
            else [e for e in self.events if e.device == device]
        )

        if not events:
            return 0

        # Find the latest end event
        end_events = [
            e
            for e in events
            if e.event_type in (EventType.COMPUTE_END, EventType.COMM_END)
        ]

        if not end_events:
            # No explicit end events, use last event time
            return events[-1].time if events else 0

        # Check if all times are numeric
        all_numeric = all(isinstance(e.time, (int, float)) for e in end_events)

        if all_numeric:
            return max(e.time for e in end_events)
        else:
            # Symbolic: return sum as conservative estimate
            # (In practice, would need smarter symbolic max)
            return end_events[-1].time

    def compute_time(self, device: int = 0) -> Union[float, Expr]:
        """Total time spent in compute operations."""
        compute_events = self.get_events(
            device=device,
            stream=StreamType.COMPUTE,
        )

        total = 0
        starts = {}

        for event in compute_events:
            if event.event_type == EventType.COMPUTE_START:
                starts[event.resource_id] = event.time
            elif event.event_type == EventType.COMPUTE_END and event.resource_id in starts:
                duration = event.time - starts[event.resource_id]
                total = total + duration
                del starts[event.resource_id]

        return total

    def comm_time(self, device: int = 0) -> Union[float, Expr]:
        """Total time spent in communication operations."""
        comm_events = self.get_events(device=device)

        total = 0
        starts = {}

        for event in comm_events:
            if event.event_type == EventType.COMM_START:
                starts[event.resource_id] = event.time
            elif event.event_type == EventType.COMM_END and event.resource_id in starts:
                duration = event.time - starts[event.resource_id]
                total = total + duration
                del starts[event.resource_id]

        return total

    def bubble_time(self, device: int = 0) -> Union[float, Expr]:
        """Calculate bubble (idle) time on a device.

        Bubble time is the time when neither compute nor communication
        is happening but the overall execution hasn't finished.
        """
        total_time = self.makespan(device=device)
        compute = self.compute_time(device=device)
        comm = self.comm_time(device=device)

        # Assuming no overlap for now (conservative)
        # More accurate would track actual overlap from streams
        return total_time - compute - comm

    def overlap_ratio(self, device: int = 0) -> float:
        """Calculate compute-communication overlap ratio.

        Returns the fraction of communication time that overlaps
        with compute (0 = no overlap, 1 = full overlap).
        """
        compute = self.compute_time(device=device)
        comm = self.comm_time(device=device)
        makespan = self.makespan(device=device)

        if not isinstance(compute, (int, float)):
            return 0.0  # Can't compute for symbolic
        if not isinstance(comm, (int, float)):
            return 0.0
        if not isinstance(makespan, (int, float)):
            return 0.0

        if comm == 0:
            return 1.0  # No communication to overlap

        # overlap = compute + comm - makespan
        overlap = max(0, compute + comm - makespan)
        return overlap / comm if comm > 0 else 0.0

    def to_trace_events(self) -> List[Dict[str, Any]]:
        """Export to Chrome Trace Event format for visualization.

        The output can be loaded in chrome://tracing or Perfetto
        for visual analysis.
        """
        self._ensure_sorted()
        trace_events = []

        # Track start times for duration events
        starts: Dict[str, TimelineEvent] = {}

        for event in self.events:
            if not isinstance(event.time, (int, float)):
                continue  # Skip symbolic events for trace

            if event.event_type in (EventType.COMPUTE_START, EventType.COMM_START):
                starts[event.resource_id] = event

            elif event.event_type in (EventType.COMPUTE_END, EventType.COMM_END):
                if event.resource_id in starts:
                    start_event = starts[event.resource_id]
                    duration = event.time - start_event.time

                    trace_events.append(
                        {
                            "name": f"{event.op_type or event.resource_id}",
                            "cat": event.stream.value,
                            "ph": "X",  # Complete event
                            "ts": start_event.time * 1e6,  # Convert to microseconds
                            "dur": duration * 1e6,
                            "pid": event.device,
                            "tid": hash(event.stream.value) % 100,
                            "args": {
                                "resource_id": event.resource_id,
                                **event.metadata,
                            },
                        }
                    )
                    del starts[event.resource_id]

            elif event.event_type == EventType.ALLOC:
                trace_events.append(
                    {
                        "name": f"alloc:{event.resource_id}",
                        "cat": "memory",
                        "ph": "i",  # Instant event
                        "ts": event.time * 1e6,
                        "pid": event.device,
                        "tid": 0,
                        "s": "g",  # Global scope
                        "args": {"size": event.size},
                    }
                )

            elif event.event_type == EventType.FREE:
                trace_events.append(
                    {
                        "name": f"free:{event.resource_id}",
                        "cat": "memory",
                        "ph": "i",
                        "ts": event.time * 1e6,
                        "pid": event.device,
                        "tid": 0,
                        "s": "g",
                        "args": {"size": event.size},
                    }
                )

        return trace_events

    def __repr__(self) -> str:
        return f"TimelineIR(events={len(self.events)}, devices={self.num_devices})"

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = ["TimelineIR Summary:"]
        lines.append(f"  Events: {len(self.events)}")
        lines.append(f"  Devices: {self.num_devices}")

        for dev in range(self.num_devices):
            lines.append(f"\n  Device {dev}:")
            peak_mem = self.peak_memory(dev)
            if isinstance(peak_mem, (int, float)):
                lines.append(f"    Peak Memory: {peak_mem/1e9:.2f} GB")
            else:
                lines.append(f"    Peak Memory: {peak_mem}")

            makespan = self.makespan(dev)
            if isinstance(makespan, (int, float)):
                lines.append(f"    Makespan: {makespan*1e3:.2f} ms")
            else:
                lines.append(f"    Makespan: {makespan}")

            compute = self.compute_time(dev)
            if isinstance(compute, (int, float)):
                lines.append(f"    Compute: {compute*1e3:.2f} ms")

            comm = self.comm_time(dev)
            if isinstance(comm, (int, float)):
                lines.append(f"    Comm: {comm*1e3:.2f} ms")

        return "\n".join(lines)

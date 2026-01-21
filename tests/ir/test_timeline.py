"""Tests for TimelineIR and TimelinePass."""

import pytest
from sympy import Symbol

from blueprinting.ir.timeline import (
    TimelineIR,
    TimelineEvent,
    EventType,
    StreamType,
    MemorySnapshot,
)
from blueprinting.ir.schedule import ScheduleIR, ScheduledOp, TensorLifetime
from blueprinting.ir.passes.timeline import TimelinePass, OverlapAnalysisPass


class TestTimelineEvent:
    """Tests for TimelineEvent class."""
    
    def test_event_creation(self):
        """Test basic event creation."""
        event = TimelineEvent(
            time=0.0,
            event_type=EventType.ALLOC,
            resource_id="tensor_0",
            device=0,
            size=1024
        )
        
        assert event.time == 0.0
        assert event.event_type == EventType.ALLOC
        assert event.resource_id == "tensor_0"
        assert event.size == 1024
    
    def test_event_comparison(self):
        """Test event comparison for sorting."""
        e1 = TimelineEvent(time=0.0, event_type=EventType.ALLOC, resource_id="t1")
        e2 = TimelineEvent(time=1.0, event_type=EventType.ALLOC, resource_id="t2")
        e3 = TimelineEvent(time=0.5, event_type=EventType.FREE, resource_id="t3")
        
        events = sorted([e2, e1, e3])
        assert events[0].time == 0.0
        assert events[1].time == 0.5
        assert events[2].time == 1.0
    
    def test_event_type_checks(self):
        """Test event type helper methods."""
        alloc = TimelineEvent(time=0, event_type=EventType.ALLOC, resource_id="t")
        compute = TimelineEvent(time=0, event_type=EventType.COMPUTE_START, resource_id="op")
        comm = TimelineEvent(time=0, event_type=EventType.COMM_START, resource_id="ar")
        
        assert alloc.is_memory_event()
        assert not alloc.is_compute_event()
        
        assert compute.is_compute_event()
        assert not compute.is_memory_event()
        
        assert comm.is_comm_event()
        assert not comm.is_compute_event()


class TestTimelineIR:
    """Tests for TimelineIR class."""
    
    def test_empty_timeline(self):
        """Test empty timeline."""
        timeline = TimelineIR(num_devices=1)
        
        assert len(timeline.events) == 0
        assert timeline.peak_memory(0) == 0
        assert timeline.makespan() == 0
    
    def test_add_events(self):
        """Test adding events."""
        timeline = TimelineIR()
        
        timeline.add_event(TimelineEvent(
            time=0.0,
            event_type=EventType.ALLOC,
            resource_id="t1",
            size=1000
        ))
        timeline.add_event(TimelineEvent(
            time=1.0,
            event_type=EventType.FREE,
            resource_id="t1",
            size=1000
        ))
        
        assert len(timeline.events) == 2
    
    def test_peak_memory_single_tensor(self):
        """Test peak memory with single tensor."""
        timeline = TimelineIR()
        
        timeline.add_event(TimelineEvent(
            time=0.0,
            event_type=EventType.ALLOC,
            resource_id="t1",
            size=1000
        ))
        timeline.add_event(TimelineEvent(
            time=1.0,
            event_type=EventType.FREE,
            resource_id="t1",
            size=1000
        ))
        
        assert timeline.peak_memory(0) == 1000
    
    def test_peak_memory_overlapping_tensors(self):
        """Test peak memory with overlapping allocations."""
        timeline = TimelineIR()
        
        # t1: allocated at 0, freed at 2
        # t2: allocated at 1, freed at 3
        # Peak is at t=1-2 when both are live
        
        timeline.add_event(TimelineEvent(time=0, event_type=EventType.ALLOC, resource_id="t1", size=1000))
        timeline.add_event(TimelineEvent(time=1, event_type=EventType.ALLOC, resource_id="t2", size=2000))
        timeline.add_event(TimelineEvent(time=2, event_type=EventType.FREE, resource_id="t1", size=1000))
        timeline.add_event(TimelineEvent(time=3, event_type=EventType.FREE, resource_id="t2", size=2000))
        
        assert timeline.peak_memory(0) == 3000  # 1000 + 2000
    
    def test_peak_memory_sequential_tensors(self):
        """Test peak memory with sequential allocations."""
        timeline = TimelineIR()
        
        # t1: allocated at 0, freed at 1
        # t2: allocated at 2, freed at 3
        # Never overlap
        
        timeline.add_event(TimelineEvent(time=0, event_type=EventType.ALLOC, resource_id="t1", size=1000))
        timeline.add_event(TimelineEvent(time=1, event_type=EventType.FREE, resource_id="t1", size=1000))
        timeline.add_event(TimelineEvent(time=2, event_type=EventType.ALLOC, resource_id="t2", size=2000))
        timeline.add_event(TimelineEvent(time=3, event_type=EventType.FREE, resource_id="t2", size=2000))
        
        assert timeline.peak_memory(0) == 2000  # max of individual peaks
    
    def test_makespan(self):
        """Test makespan calculation."""
        timeline = TimelineIR()
        
        timeline.add_event(TimelineEvent(time=0, event_type=EventType.COMPUTE_START, resource_id="op1"))
        timeline.add_event(TimelineEvent(time=0.5, event_type=EventType.COMPUTE_END, resource_id="op1"))
        timeline.add_event(TimelineEvent(time=0.5, event_type=EventType.COMPUTE_START, resource_id="op2"))
        timeline.add_event(TimelineEvent(time=1.0, event_type=EventType.COMPUTE_END, resource_id="op2"))
        
        assert timeline.makespan() == 1.0
    
    def test_compute_time(self):
        """Test compute time calculation."""
        timeline = TimelineIR()
        
        # Two ops: 0.5s and 0.3s
        timeline.add_event(TimelineEvent(time=0, event_type=EventType.COMPUTE_START, resource_id="op1"))
        timeline.add_event(TimelineEvent(time=0.5, event_type=EventType.COMPUTE_END, resource_id="op1"))
        timeline.add_event(TimelineEvent(time=0.5, event_type=EventType.COMPUTE_START, resource_id="op2"))
        timeline.add_event(TimelineEvent(time=0.8, event_type=EventType.COMPUTE_END, resource_id="op2"))
        
        assert abs(timeline.compute_time(0) - 0.8) < 1e-6
    
    def test_comm_time(self):
        """Test communication time calculation."""
        timeline = TimelineIR()
        
        timeline.add_event(TimelineEvent(time=0, event_type=EventType.COMM_START, resource_id="ar1", stream=StreamType.NCCL))
        timeline.add_event(TimelineEvent(time=0.1, event_type=EventType.COMM_END, resource_id="ar1", stream=StreamType.NCCL))
        
        assert abs(timeline.comm_time(0) - 0.1) < 1e-6
    
    def test_memory_trace(self):
        """Test memory trace generation."""
        timeline = TimelineIR()
        
        timeline.add_event(TimelineEvent(time=0, event_type=EventType.ALLOC, resource_id="t1", size=1000))
        timeline.add_event(TimelineEvent(time=1, event_type=EventType.ALLOC, resource_id="t2", size=2000))
        timeline.add_event(TimelineEvent(time=2, event_type=EventType.FREE, resource_id="t1", size=1000))
        
        trace = timeline.memory_trace(0)
        
        assert len(trace) == 3
        assert trace[0].allocated == 1000
        assert trace[1].allocated == 3000
        assert trace[2].allocated == 2000
    
    def test_multi_device(self):
        """Test multi-device timeline."""
        timeline = TimelineIR(num_devices=2)
        
        # Device 0
        timeline.add_event(TimelineEvent(time=0, event_type=EventType.ALLOC, resource_id="d0_t1", device=0, size=1000))
        timeline.add_event(TimelineEvent(time=1, event_type=EventType.FREE, resource_id="d0_t1", device=0, size=1000))
        
        # Device 1
        timeline.add_event(TimelineEvent(time=0, event_type=EventType.ALLOC, resource_id="d1_t1", device=1, size=2000))
        timeline.add_event(TimelineEvent(time=1, event_type=EventType.FREE, resource_id="d1_t1", device=1, size=2000))
        
        assert timeline.peak_memory(0) == 1000
        assert timeline.peak_memory(1) == 2000
    
    def test_symbolic_memory(self):
        """Test peak memory with symbolic sizes."""
        B = Symbol("B")
        timeline = TimelineIR()
        
        timeline.add_event(TimelineEvent(time=0, event_type=EventType.ALLOC, resource_id="t1", size=B * 1000))
        timeline.add_event(TimelineEvent(time=1, event_type=EventType.FREE, resource_id="t1", size=B * 1000))
        
        # Should return symbolic sum
        peak = timeline.peak_memory(0)
        assert peak == B * 1000
    
    def test_get_events_filter(self):
        """Test event filtering."""
        timeline = TimelineIR()
        
        timeline.add_event(TimelineEvent(time=0, event_type=EventType.ALLOC, resource_id="t1"))
        timeline.add_event(TimelineEvent(time=0, event_type=EventType.COMPUTE_START, resource_id="op1"))
        timeline.add_event(TimelineEvent(time=1, event_type=EventType.COMPUTE_END, resource_id="op1"))
        
        alloc_events = timeline.get_events(event_type=EventType.ALLOC)
        assert len(alloc_events) == 1
        
        compute_events = timeline.get_events(event_type=EventType.COMPUTE_START)
        assert len(compute_events) == 1


class TestTimelinePass:
    """Tests for TimelinePass."""
    
    def test_convert_schedule_to_timeline(self):
        """Test converting ScheduleIR to TimelineIR."""
        schedule = ScheduleIR()
        
        op = ScheduledOp(
            op_id="linear_0",
            op_type="Linear",
            device=0,
            start=0.0,
            duration=0.1,
        )
        op.tensors_alloc.append(TensorLifetime(
            tensor_id="act_0",
            alloc_time=0.0,
            free_time=0.1,
            size=1000
        ))
        schedule.add_op(op)
        
        timeline_pass = TimelinePass()
        timeline = timeline_pass.run(schedule)
        
        assert isinstance(timeline, TimelineIR)
        assert len(timeline.events) > 0
        
        # Should have compute start/end and alloc/free
        compute_starts = timeline.get_events(event_type=EventType.COMPUTE_START)
        assert len(compute_starts) == 1
        
        allocs = timeline.get_events(event_type=EventType.ALLOC)
        assert len(allocs) == 1
    
    def test_communication_events(self):
        """Test that communication ops create COMM events."""
        schedule = ScheduleIR()
        
        schedule.add_op(ScheduledOp(
            op_id="allreduce_0",
            op_type="AllReduce",
            device=0,
            start=0.0,
            duration=0.05,
        ))
        
        timeline_pass = TimelinePass()
        timeline = timeline_pass.run(schedule)
        
        comm_starts = timeline.get_events(event_type=EventType.COMM_START)
        assert len(comm_starts) == 1
        assert comm_starts[0].stream == StreamType.NCCL
    
    def test_multiple_ops(self):
        """Test timeline with multiple operations."""
        schedule = ScheduleIR()
        
        schedule.add_op(ScheduledOp(op_id="op1", op_type="Linear", device=0, start=0.0, duration=0.1))
        schedule.add_op(ScheduledOp(op_id="op2", op_type="Linear", device=0, start=0.1, duration=0.1))
        schedule.add_op(ScheduledOp(op_id="op3", op_type="AllReduce", device=0, start=0.2, duration=0.05))
        
        timeline_pass = TimelinePass()
        timeline = timeline_pass.run(schedule)
        
        # 3 ops -> 6 start/end events
        assert len(timeline.events) == 6


class TestOverlapAnalysisPass:
    """Tests for OverlapAnalysisPass."""
    
    def test_overlap_analysis(self):
        """Test overlap analysis."""
        timeline = TimelineIR()
        
        # Compute: 0-1
        timeline.add_event(TimelineEvent(time=0, event_type=EventType.COMPUTE_START, resource_id="op1", stream=StreamType.COMPUTE))
        timeline.add_event(TimelineEvent(time=1, event_type=EventType.COMPUTE_END, resource_id="op1", stream=StreamType.COMPUTE))
        
        # Comm: 0.5-1.0 (overlaps with compute from 0.5-1.0)
        timeline.add_event(TimelineEvent(time=0.5, event_type=EventType.COMM_START, resource_id="ar1", stream=StreamType.NCCL))
        timeline.add_event(TimelineEvent(time=1.0, event_type=EventType.COMM_END, resource_id="ar1", stream=StreamType.NCCL))
        
        overlap_pass = OverlapAnalysisPass()
        result = overlap_pass.run(timeline)
        
        assert "overlap_analysis" in result.metadata
        assert 0 in result.metadata["overlap_analysis"]


class TestTraceExport:
    """Tests for Chrome trace export."""
    
    def test_trace_events_format(self):
        """Test Chrome trace event format."""
        timeline = TimelineIR()
        
        timeline.add_event(TimelineEvent(time=0, event_type=EventType.COMPUTE_START, resource_id="op1", op_type="Linear"))
        timeline.add_event(TimelineEvent(time=0.001, event_type=EventType.COMPUTE_END, resource_id="op1", op_type="Linear"))
        timeline.add_event(TimelineEvent(time=0, event_type=EventType.ALLOC, resource_id="t1", size=1000))
        
        trace_events = timeline.to_trace_events()
        
        # Should have 1 duration event (compute) and 1 instant event (alloc)
        assert len(trace_events) == 2
        
        # Check duration event format
        duration_event = [e for e in trace_events if e["ph"] == "X"][0]
        assert "name" in duration_event
        assert "ts" in duration_event
        assert "dur" in duration_event
        
        # Check instant event format
        instant_event = [e for e in trace_events if e["ph"] == "i"][0]
        assert "name" in instant_event
        assert "alloc" in instant_event["name"]


class TestIntegration:
    """Integration tests for the timeline pipeline."""
    
    def test_full_pipeline(self):
        """Test full pipeline: Schedule -> Timeline -> Evaluate."""
        from blueprinting.ir.passes import EvaluatePass
        
        # Create schedule
        schedule = ScheduleIR()
        op = ScheduledOp(
            op_id="linear_0",
            op_type="Linear", 
            device=0,
            start=0.0,
            duration=0.001,
        )
        op.tensors_alloc.append(TensorLifetime(
            tensor_id="act_0",
            alloc_time=0.0,
            free_time=0.001,
            size=1024 * 1024  # 1MB
        ))
        schedule.add_op(op)
        
        # Convert to timeline
        timeline_pass = TimelinePass()
        timeline = timeline_pass.run(schedule)
        
        # Evaluate
        evaluate_pass = EvaluatePass(subs={})
        result = evaluate_pass.run(timeline)
        
        assert result.peak_memory == 1024 * 1024
        assert result.e2e_time > 0

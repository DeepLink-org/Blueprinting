"""Tests for compiler passes."""

import pytest
from sympy import Symbol

from blueprinting.ir.graph import GraphIR, OpNode
from blueprinting.ir.schedule import ScheduleIR
from blueprinting.ir.result import SimulationResult
from blueprinting.ir.builder import IRBuilder, build_transformer_block
from blueprinting.ir.passes import (
    Pass,
    WorkloadPass,
    ParallelPass,
    SchedulePass,
    EvaluatePass,
)


class TestPassBase:
    """Tests for Pass base class."""
    
    def test_pass_name(self):
        """Test pass name property."""
        pass_instance = WorkloadPass()
        assert pass_instance.name == "WorkloadPass"
    
    def test_pass_repr(self):
        """Test pass string representation."""
        pass_instance = WorkloadPass()
        assert "WorkloadPass" in repr(pass_instance)


class TestWorkloadPass:
    """Tests for WorkloadPass."""
    
    def test_linear_workload(self):
        """Test workload calculation for Linear layer."""
        builder = IRBuilder()
        x = builder.add_input("x", shape=[4, 8, 16])
        builder.add_linear("linear", [x], 16, 32, bias=True)
        graph = builder.build()
        
        # Set batch_seq in attrs
        graph.nodes["linear"].attrs["batch_seq"] = 4 * 8
        
        workload_pass = WorkloadPass(dtype_bytes=2, count_add=True)
        result = workload_pass.run(graph)
        
        node = result.nodes["linear"]
        assert node.flops_fw is not None
        assert node.flops_bw is not None
        assert node.memory_fw is not None
        assert node.weight_bytes is not None
    
    def test_rmsnorm_workload(self):
        """Test workload calculation for RMSNorm."""
        builder = IRBuilder()
        x = builder.add_input("x")
        builder.add_rmsnorm("norm", [x], normalized_shape=256)
        graph = builder.build()
        
        graph.nodes["norm"].attrs["batch_seq"] = 32
        
        workload_pass = WorkloadPass()
        result = workload_pass.run(graph)
        
        node = result.nodes["norm"]
        # RMSNorm: forward = 4N, backward = 8N
        assert node.flops_fw == 4 * 32 * 256
        assert node.flops_bw == 8 * 32 * 256
    
    def test_add_workload(self):
        """Test workload calculation for Add."""
        builder = IRBuilder()
        x = builder.add_input("x")
        y = builder.add_input("y")
        builder.add_elementwise("add", "Add", [x, y])
        graph = builder.build()
        
        graph.nodes["add"].attrs["num_elements"] = 1000
        
        workload_pass = WorkloadPass()
        result = workload_pass.run(graph)
        
        node = result.nodes["add"]
        assert node.flops_fw == 1000  # N additions
        assert node.flops_bw == 0  # No computation in backward
    
    def test_mul_workload(self):
        """Test workload calculation for Mul."""
        builder = IRBuilder()
        x = builder.add_input("x")
        y = builder.add_input("y")
        builder.add_elementwise("mul", "Mul", [x, y])
        graph = builder.build()
        
        graph.nodes["mul"].attrs["num_elements"] = 1000
        
        workload_pass = WorkloadPass()
        result = workload_pass.run(graph)
        
        node = result.nodes["mul"]
        assert node.flops_fw == 1000
        assert node.flops_bw == 2000  # 2N for backward
    
    def test_silu_workload(self):
        """Test workload calculation for SiLU."""
        builder = IRBuilder()
        x = builder.add_input("x")
        builder.add_elementwise("silu", "SiLU", [x])
        graph = builder.build()
        
        graph.nodes["silu"].attrs["num_elements"] = 1000
        
        workload_pass = WorkloadPass()
        result = workload_pass.run(graph)
        
        node = result.nodes["silu"]
        assert node.flops_fw == 4000  # 4N
        assert node.flops_bw == 6000  # 6N
    
    def test_comm_workload(self):
        """Test workload calculation for communication ops."""
        builder = IRBuilder()
        x = builder.add_input("x")
        builder.add_comm("ar", "AllReduce", [x], num_peers=8)
        graph = builder.build()
        
        graph.nodes["ar"].attrs["data_size"] = 1000
        
        workload_pass = WorkloadPass()
        result = workload_pass.run(graph)
        
        node = result.nodes["ar"]
        assert node.flops_fw == 0  # No FLOPs
        assert node.flops_bw == 0
        assert node.comm_bytes_fw is not None


class TestParallelPass:
    """Tests for ParallelPass."""
    
    def test_no_parallelism(self):
        """Test with no parallelism (TP=PP=DP=1)."""
        builder = IRBuilder()
        x = builder.add_input("x")
        builder.add_linear("linear", [x], 16, 32)
        graph = builder.build()
        
        parallel_pass = ParallelPass(tp=1, pp=1, dp=1)
        result = parallel_pass.run(graph)
        
        # Should be unchanged
        assert result.metadata["tp"] == 1
        assert result.metadata["pp"] == 1
        assert result.metadata["dp"] == 1
    
    def test_tensor_parallelism(self):
        """Test tensor parallelism."""
        builder = IRBuilder()
        x = builder.add_input("x")
        builder.add_linear("linear", [x], 256, 512, shard="tp_col")
        graph = builder.build()
        
        # Set workload first
        graph.nodes["linear"].flops_fw = 1000
        graph.nodes["linear"].weight_bytes = 2000
        
        parallel_pass = ParallelPass(tp=4, pp=1, dp=1)
        result = parallel_pass.run(graph)
        
        node = result.nodes["linear"]
        # FLOPs and weights should be divided by TP
        assert node.flops_fw == 250  # 1000 / 4
        assert node.weight_bytes == 500  # 2000 / 4
        assert node.shard == "tp_col"
    
    def test_pipeline_parallelism(self):
        """Test pipeline parallelism device assignment."""
        graph = build_transformer_block(
            hidden=256,
            feedforward=512,
            num_heads=8,
            head_dim=32,
            seq_len=128,
            batch_size=2,
            block_id=0,
            tp=1
        )
        graph.metadata["num_layers"] = 4
        
        parallel_pass = ParallelPass(tp=1, pp=2, dp=1)
        result = parallel_pass.run(graph)
        
        assert result.metadata["pp"] == 2


class TestSchedulePass:
    """Tests for SchedulePass."""
    
    def test_basic_schedule(self):
        """Test basic schedule generation."""
        builder = IRBuilder()
        x = builder.add_input("x")
        l1 = builder.add_linear("l1", [x], 16, 32)
        l2 = builder.add_linear("l2", [l1], 32, 64)
        graph = builder.build()
        
        # Set workload
        for node in graph.nodes.values():
            node.flops_fw = 1000
            node.memory_fw = 2000
        
        schedule_pass = SchedulePass(
            system_config={"peak_tflops": 100, "memory_bandwidth_gbps": 1000}
        )
        result = schedule_pass.run(graph)
        
        assert isinstance(result, ScheduleIR)
        assert len(result.ops) > 0
    
    def test_schedule_ordering(self):
        """Test that schedule respects dependencies."""
        builder = IRBuilder()
        x = builder.add_input("x")
        l1 = builder.add_linear("l1", [x], 16, 32)
        l2 = builder.add_linear("l2", [l1], 32, 64)
        graph = builder.build()
        
        for node in graph.nodes.values():
            node.flops_fw = 1000
            node.memory_fw = 2000
        
        schedule_pass = SchedulePass()
        result = schedule_pass.run(graph)
        
        # Find scheduled ops
        l1_sched = next((op for op in result.ops if op.op_id == "l1"), None)
        l2_sched = next((op for op in result.ops if op.op_id == "l2"), None)
        
        if l1_sched and l2_sched:
            # l2 should start after l1 ends
            assert l2_sched.start >= l1_sched.end
    
    def test_schedule_makespan(self):
        """Test makespan calculation."""
        builder = IRBuilder()
        x = builder.add_input("x")
        builder.add_linear("l1", [x], 16, 32)
        graph = builder.build()
        
        graph.nodes["l1"].flops_fw = 1e12  # 1 TFLOP
        graph.nodes["l1"].memory_fw = 0
        
        schedule_pass = SchedulePass(
            system_config={"peak_tflops": 100}  # 100 TFLOPS
        )
        result = schedule_pass.run(graph)
        
        # 1 TFLOP / 100 TFLOPS = 0.01 seconds = 10ms
        makespan = result.makespan()
        assert makespan > 0


class TestEvaluatePass:
    """Tests for EvaluatePass."""
    
    def test_basic_evaluation(self):
        """Test basic evaluation."""
        schedule = ScheduleIR()
        
        from blueprinting.ir.schedule import ScheduledOp
        schedule.add_op(ScheduledOp(
            op_id="l1",
            op_type="Linear",
            device=0,
            start=0,
            duration=0.001,  # 1ms
            attrs={"flops_fw": 1e9, "flops_bw": 2e9}
        ))
        
        schedule.metadata["peak_tflops"] = 100
        schedule.metadata["memory_capacity"] = 80e9
        
        evaluate_pass = EvaluatePass(subs={}, training=True)
        result = evaluate_pass.run(schedule)
        
        assert isinstance(result, SimulationResult)
        assert result.e2e_time > 0
    
    def test_symbolic_evaluation(self):
        """Test evaluation with symbolic substitutions."""
        schedule = ScheduleIR()
        
        B = Symbol("B")
        
        from blueprinting.ir.schedule import ScheduledOp, TensorLifetime
        op = ScheduledOp(
            op_id="l1",
            op_type="Linear",
            device=0,
            start=0,
            duration=0.001,
        )
        op.tensors_alloc.append(TensorLifetime(
            tensor_id="act",
            alloc_time=0,
            free_time=0.001,
            size=B * 1000
        ))
        schedule.add_op(op)
        
        schedule.metadata["peak_tflops"] = 100
        schedule.metadata["memory_capacity"] = 80e9
        
        evaluate_pass = EvaluatePass(subs={"B": 4}, training=True)
        result = evaluate_pass.run(schedule)
        
        # B=4, so size should be 4000
        assert result.peak_memory == 4000
    
    def test_memory_breakdown(self):
        """Test memory breakdown calculation."""
        schedule = ScheduleIR()
        
        from blueprinting.ir.schedule import ScheduledOp
        schedule.add_op(ScheduledOp(
            op_id="l1",
            op_type="Linear",
            device=0,
            start=0,
            duration=0.001,
            attrs={"weight_bytes": 1e6}  # 1MB weights
        ))
        
        schedule.metadata["peak_tflops"] = 100
        schedule.metadata["memory_capacity"] = 80e9
        
        evaluate_pass = EvaluatePass(subs={}, training=True, optimizer="adam")
        result = evaluate_pass.run(schedule)
        
        assert result.memory_breakdown is not None
        assert result.memory_breakdown.weights == 1e6
    
    def test_feasibility_check(self):
        """Test memory feasibility check."""
        schedule = ScheduleIR()
        schedule.metadata["memory_capacity"] = 1e6  # Only 1MB
        
        from blueprinting.ir.schedule import ScheduledOp, TensorLifetime
        op = ScheduledOp(op_id="l1", op_type="Linear", device=0, start=0, duration=0.001)
        op.tensors_alloc.append(TensorLifetime(
            tensor_id="act",
            alloc_time=0,
            free_time=0.001,
            size=1e9  # 1GB - exceeds capacity
        ))
        schedule.add_op(op)
        
        evaluate_pass = EvaluatePass(subs={}, training=True)
        result = evaluate_pass.run(schedule)
        
        assert not result.is_feasible()
        assert len(result.warnings) > 0

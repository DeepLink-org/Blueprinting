"""Tests for Compiler and integration tests."""

import pytest
from sympy import Symbol

from blueprinting.ir.graph import GraphIR, OpNode
from blueprinting.ir.schedule import ScheduleIR
from blueprinting.ir.result import SimulationResult, MemoryBreakdown, TimeBreakdown
from blueprinting.ir.builder import IRBuilder, build_transformer_block, build_transformer_model
from blueprinting.ir.compiler import Compiler, compile_model
from blueprinting.ir.passes import (
    WorkloadPass,
    ParallelPass,
    SchedulePass,
    EvaluatePass,
)


class TestCompiler:
    """Tests for Compiler class."""
    
    def test_empty_compiler(self):
        """Test compiler with no passes."""
        compiler = Compiler()
        assert len(compiler.passes) == 0
    
    def test_add_pass(self):
        """Test adding passes."""
        compiler = Compiler()
        compiler.add_pass(WorkloadPass())
        compiler.add_pass(ParallelPass())
        
        assert len(compiler.passes) == 2
        assert compiler.passes[0].name == "WorkloadPass"
        assert compiler.passes[1].name == "ParallelPass"
    
    def test_add_pass_chaining(self):
        """Test fluent interface for adding passes."""
        compiler = (Compiler()
            .add_pass(WorkloadPass())
            .add_pass(ParallelPass())
            .add_pass(SchedulePass())
            .add_pass(EvaluatePass()))
        
        assert len(compiler.passes) == 4
    
    def test_reset(self):
        """Test compiler reset."""
        compiler = Compiler()
        compiler.add_pass(WorkloadPass())
        compiler.reset()
        
        assert len(compiler.passes) == 0
    
    def test_repr(self):
        """Test compiler string representation."""
        compiler = (Compiler()
            .add_pass(WorkloadPass())
            .add_pass(ParallelPass()))
        
        repr_str = repr(compiler)
        assert "WorkloadPass" in repr_str
        assert "ParallelPass" in repr_str
    
    def test_default_pipeline(self):
        """Test default pipeline creation."""
        compiler = Compiler.default_pipeline(
            tp=8,
            pp=4,
            dp=2,
            subs={"B": 4},
            training=True
        )
        
        assert len(compiler.passes) == 4
        assert isinstance(compiler.passes[0], WorkloadPass)
        assert isinstance(compiler.passes[1], ParallelPass)
        assert isinstance(compiler.passes[2], SchedulePass)
        assert isinstance(compiler.passes[3], EvaluatePass)


class TestCompilerIntegration:
    """Integration tests for the full compilation pipeline."""
    
    def test_simple_linear_graph(self):
        """Test compilation of a simple linear graph."""
        # Build graph
        builder = IRBuilder()
        x = builder.add_input("x", shape=[4, 128, 256])
        builder.add_linear("linear", [x], 256, 512)
        graph = builder.build()
        
        # Set batch_seq for workload calculation
        graph.nodes["linear"].attrs["batch_seq"] = 4 * 128
        
        # Create compiler
        compiler = (Compiler()
            .add_pass(WorkloadPass())
            .add_pass(ParallelPass(tp=1, pp=1, dp=1))
            .add_pass(SchedulePass())
            .add_pass(EvaluatePass(subs={})))
        
        # Compile
        result = compiler.compile(graph)
        
        assert isinstance(result, SimulationResult)
        assert result.e2e_time >= 0
        assert result.peak_memory >= 0
    
    def test_two_layer_graph(self):
        """Test compilation of a two-layer graph."""
        builder = IRBuilder()
        x = builder.add_input("x", shape=[2, 64, 128])
        l1 = builder.add_linear("l1", [x], 128, 256)
        l2 = builder.add_linear("l2", [l1], 256, 128)
        graph = builder.build()
        
        # Set batch_seq for workload calculation
        graph.nodes["l1"].attrs["batch_seq"] = 2 * 64
        graph.nodes["l2"].attrs["batch_seq"] = 2 * 64
        
        compiler = (Compiler()
            .add_pass(WorkloadPass())
            .add_pass(ParallelPass(tp=1, pp=1, dp=1))
            .add_pass(SchedulePass())
            .add_pass(EvaluatePass(subs={})))
        
        result = compiler.compile(graph)
        
        assert isinstance(result, SimulationResult)
        assert result.e2e_time > 0
    
    @pytest.mark.slow
    def test_transformer_block_compilation(self):
        """Test compilation of a transformer block (slow)."""
        graph = build_transformer_block(
            hidden=256,
            feedforward=512,
            num_heads=8,
            head_dim=32,
            seq_len=128,
            batch_size=4,
            block_id=0,
            tp=1
        )
        
        subs = {
            "B": 4,
            "S": 128,
            "H": 256,
            "FF": 512,
            "batch_seq": 4 * 128,
            "hidden": 256,
            "feedforward": 512,
        }
        
        compiler = Compiler.default_pipeline(
            tp=1, pp=1, dp=1,
            subs=subs,
            training=True
        )
        
        result = compiler.compile(graph)
        
        assert isinstance(result, SimulationResult)
    
    @pytest.mark.slow
    def test_transformer_model_compilation(self):
        """Test compilation of a small transformer model (slow)."""
        graph = build_transformer_model(
            num_layers=1,  # Reduced from 2
            hidden=128,    # Reduced from 256
            feedforward=256,
            num_heads=4,
            head_dim=32,
            seq_len=64,    # Reduced from 128
            batch_size=2,  # Reduced from 4
            tp=1,
            pp=1
        )
        
        subs = {
            "B": 2,
            "S": 64,
            "H": 128,
            "FF": 256,
            "batch_size": 2,
            "seq_len": 64,
            "hidden": 128,
            "feedforward": 256,
            "batch_seq": 2 * 64,
            "heads": 4,
            "head_dim": 32,
        }
        
        compiler = Compiler.default_pipeline(
            tp=1, pp=1, dp=1,
            subs=subs,
            training=True
        )
        
        result = compiler.compile(graph)
        
        assert isinstance(result, SimulationResult)


class TestCompileModelFunction:
    """Tests for the compile_model convenience function."""
    
    @pytest.mark.slow
    def test_basic_compile(self):
        """Test basic model compilation (slow)."""
        graph = build_transformer_model(
            num_layers=1,  # Minimal
            hidden=128,
            feedforward=256,
            num_heads=4,
            head_dim=32,
            seq_len=64,
            batch_size=2,
            tp=1,
            pp=1
        )
        
        result = compile_model(
            graph,
            tp=1,
            pp=1,
            dp=1,
            batch_size=2,
            seq_len=64,
            hidden=128,
            feedforward=256,
            num_layers=1,
            training=True
        )
        
        assert isinstance(result, SimulationResult)


class TestSimulationResult:
    """Tests for SimulationResult class."""
    
    def test_memory_breakdown(self):
        """Test MemoryBreakdown class."""
        breakdown = MemoryBreakdown(
            weights=1e9,
            activations=2e9,
            gradients=1e9,
            optimizer_states=4e9
        )
        
        assert breakdown.total == 8e9
        assert "weights" in repr(breakdown).lower()
    
    def test_time_breakdown(self):
        """Test TimeBreakdown class."""
        breakdown = TimeBreakdown(
            forward=0.01,
            backward=0.02,
            optimizer=0.001,
            communication=0.005,
            bubble=0.002
        )
        
        assert breakdown.total == 0.038
        assert "forward" in repr(breakdown).lower()
    
    def test_simulation_result_mfu(self):
        """Test MFU calculation."""
        result = SimulationResult(
            peak_memory=10e9,
            e2e_time=0.1,
            total_flops=1e15,
            achieved_flops=1e15,
            config={"peak_tflops": 312}
        )
        
        # MFU = achieved_flops / (peak_tflops * time)
        # = 1e15 / (312e12 * 0.1) = 1e15 / 3.12e13 ≈ 32
        assert result.mfu > 0
    
    def test_simulation_result_to_dict(self):
        """Test result serialization."""
        result = SimulationResult(
            peak_memory=10e9,
            e2e_time=0.1,
            memory_breakdown=MemoryBreakdown(weights=1e9),
            time_breakdown=TimeBreakdown(forward=0.05),
            config={"memory_capacity": 80e9}
        )
        
        d = result.to_dict()
        
        assert "peak_memory_bytes" in d
        assert "peak_memory_gb" in d
        assert "e2e_time_seconds" in d
        assert "memory_breakdown" in d
        assert "time_breakdown" in d
    
    def test_feasibility(self):
        """Test feasibility check."""
        # Feasible case
        result_ok = SimulationResult(
            peak_memory=40e9,
            config={"memory_capacity": 80e9}
        )
        assert result_ok.is_feasible()
        
        # Not feasible
        result_bad = SimulationResult(
            peak_memory=100e9,
            config={"memory_capacity": 80e9}
        )
        assert not result_bad.is_feasible()


class TestScheduleIR:
    """Tests for ScheduleIR class."""
    
    def test_peak_memory_simple(self):
        """Test simple peak memory calculation."""
        from blueprinting.ir.schedule import ScheduleIR, ScheduledOp, TensorLifetime
        
        schedule = ScheduleIR()
        
        op1 = ScheduledOp(op_id="op1", op_type="Linear", device=0, start=0, duration=1)
        op1.tensors_alloc.append(TensorLifetime("t1", alloc_time=0, free_time=2, size=1000))
        
        op2 = ScheduledOp(op_id="op2", op_type="Linear", device=0, start=1, duration=1)
        op2.tensors_alloc.append(TensorLifetime("t2", alloc_time=1, free_time=3, size=2000))
        
        schedule.add_op(op1)
        schedule.add_op(op2)
        
        # Between t=1 and t=2, both t1 and t2 are alive
        peak = schedule.peak_memory(device=0)
        assert peak == 3000  # 1000 + 2000
    
    def test_makespan(self):
        """Test makespan calculation."""
        from blueprinting.ir.schedule import ScheduleIR, ScheduledOp
        
        schedule = ScheduleIR()
        schedule.add_op(ScheduledOp(op_id="op1", op_type="A", device=0, start=0, duration=1))
        schedule.add_op(ScheduledOp(op_id="op2", op_type="B", device=0, start=1, duration=2))
        schedule.add_op(ScheduledOp(op_id="op3", op_type="C", device=0, start=3, duration=1))
        
        assert schedule.makespan() == 4  # op3 ends at t=4
    
    def test_ops_by_device(self):
        """Test filtering ops by device."""
        from blueprinting.ir.schedule import ScheduleIR, ScheduledOp
        
        schedule = ScheduleIR(num_devices=2)
        schedule.add_op(ScheduledOp(op_id="op1", op_type="A", device=0, start=0, duration=1))
        schedule.add_op(ScheduledOp(op_id="op2", op_type="B", device=1, start=0, duration=1))
        schedule.add_op(ScheduledOp(op_id="op3", op_type="C", device=0, start=1, duration=1))
        
        device0_ops = schedule.get_ops_on_device(0)
        device1_ops = schedule.get_ops_on_device(1)
        
        assert len(device0_ops) == 2
        assert len(device1_ops) == 1

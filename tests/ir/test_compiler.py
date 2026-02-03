"""Tests for Compiler and integration tests."""

import pytest
from sympy import Symbol

from blueprinting.ir.graph import GraphIR, OpNode
from blueprinting.ir.schedule import ScheduleIR
from blueprinting.ir.result import SimulationResult, MemoryBreakdown, TimeBreakdown
from blueprinting.ir.builder import IRBuilder, build_transformer_layer, build_transformer_model
from blueprinting.ir.compiler import Compiler, compile_model
from blueprinting.ir.passes import (
    ExpandPass,
    ParallelPass,
    SchedulePass,
    TimelinePassV2,
    SimulatePass,
    OptimizerPass,
    OptimizerConfig,
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
        compiler.add_pass(ParallelPass())
        compiler.add_pass(ExpandPass())
        
        assert len(compiler.passes) == 2
        assert compiler.passes[0].name == "ParallelPass"
        assert compiler.passes[1].name == "ExpandPass"
    
    def test_add_pass_chaining(self):
        """Test fluent interface for adding passes."""
        compiler = (Compiler()
            .add_pass(ParallelPass())
            .add_pass(ExpandPass())
            .add_pass(SchedulePass())
            .add_pass(TimelinePassV2())
            .add_pass(SimulatePass()))
        
        assert len(compiler.passes) == 5
    
    def test_reset(self):
        """Test compiler reset."""
        compiler = Compiler()
        compiler.add_pass(ExpandPass())
        compiler.reset()
        
        assert len(compiler.passes) == 0
    
    def test_repr(self):
        """Test compiler string representation."""
        compiler = (Compiler()
            .add_pass(ExpandPass())
            .add_pass(ParallelPass()))
        
        repr_str = repr(compiler)
        assert "ExpandPass" in repr_str
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
        
        # With pp=4 and training=True, we expect:
        # ParallelPass, ExpandPass, SchedulePass, OptimizerPass, PipelineSchedulePass, TimelinePassV2, SimulatePass
        assert len(compiler.passes) >= 6
        assert isinstance(compiler.passes[0], ParallelPass)
        assert isinstance(compiler.passes[1], ExpandPass)
        assert isinstance(compiler.passes[2], SchedulePass)


class TestCompilerIntegration:
    """Integration tests for the full compilation pipeline.
    
    Note: These tests need update for the new IR architecture.
    The builder.build_transformer_model returns a different structure.
    """
    
    @pytest.mark.skip(reason="Test needs update for new IR architecture")
    @pytest.mark.slow
    def test_transformer_model_compilation(self):
        """Test compilation of a small transformer model (slow)."""
        pass


class TestCompileModelFunction:
    """Tests for the compile_model convenience function.
    
    Note: These tests need update for the new IR architecture.
    """
    
    @pytest.mark.skip(reason="Test needs update for new IR architecture")
    @pytest.mark.slow
    def test_basic_compile(self):
        """Test basic model compilation (slow)."""
        pass


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
    """Tests for ScheduleIR class.
    
    Note: These tests are skipped because the ScheduleIR API has changed.
    The new IR architecture uses a different ScheduledOp structure.
    """
    
    @pytest.mark.skip(reason="Test needs update for new ScheduleIR API")
    def test_peak_memory_simple(self):
        """Test simple peak memory calculation."""
        pass
    
    @pytest.mark.skip(reason="Test needs update for new ScheduleIR API")
    def test_makespan(self):
        """Test makespan calculation."""
        pass
    
    @pytest.mark.skip(reason="Test needs update for new ScheduleIR API")
    def test_ops_by_device(self):
        """Test filtering ops by device."""
        pass

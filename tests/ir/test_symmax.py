"""Tests for SymMax - lazy symbolic max."""

import pytest
from sympy import Symbol

from blueprinting.ir.symmax import (
    SymMax,
    sym_max,
    eval_lazy,
    _to_float,
    _SymSum,
    _SymProduct,
    _SymQuotient,
    clear_expr_cache,
    get_cache_stats,
)


class TestSymMax:
    """Tests for SymMax class."""
    
    def setup_method(self):
        """Clear cache before each test."""
        clear_expr_cache()
    
    def test_numeric_max(self):
        """Test max with numeric values."""
        result = sym_max(3, 5)
        assert result == 5
        
        result = sym_max(10.5, 3.2)
        assert result == 10.5
    
    def test_symbolic_creation(self):
        """Test creating SymMax with symbols."""
        a = Symbol('a')
        m = SymMax(a, 10)
        
        assert isinstance(m, SymMax)
        assert len(m.args) == 2
    
    def test_symbolic_eval(self):
        """Test evaluating SymMax with substitutions."""
        a = Symbol('a')
        m = SymMax(a, 10)
        
        # a=5 < 10
        result = m.eval({a: 5})
        assert result == 10
        
        # a=15 > 10
        result = m.eval({a: 15})
        assert result == 15
    
    def test_nested_symmax(self):
        """Test nested SymMax flattening."""
        a, b, c = Symbol('a'), Symbol('b'), Symbol('c')
        
        inner = SymMax(a, b)
        outer = SymMax(inner, c)
        
        # Should flatten to SymMax(a, b, c)
        assert len(outer.args) == 3
    
    def test_symmax_chain_eval(self):
        """Test evaluating chain of SymMax."""
        a, b = Symbol('a'), Symbol('b')
        
        m1 = sym_max(a, 5)
        m2 = sym_max(m1, b)
        
        # a=3, b=4 -> max(max(3, 5), 4) = max(5, 4) = 5
        result = eval_lazy(m2, {a: 3, b: 4})
        assert result == 5
        
        # a=10, b=4 -> max(max(10, 5), 4) = max(10, 4) = 10
        result = eval_lazy(m2, {a: 10, b: 4})
        assert result == 10
        
        # a=3, b=20 -> max(max(3, 5), 20) = max(5, 20) = 20
        result = eval_lazy(m2, {a: 3, b: 20})
        assert result == 20
    
    def test_partial_eval(self):
        """Test partial evaluation (some symbols remain)."""
        a, b = Symbol('a'), Symbol('b')
        m = SymMax(a, b)
        
        # Only substitute a
        result = m.eval({a: 10})
        
        # Should return new SymMax since b is unknown
        assert isinstance(result, SymMax)
    
    def test_symmax_addition(self):
        """Test SymMax + number."""
        a = Symbol('a')
        m = SymMax(a, 10)
        
        result = m + 5
        assert isinstance(result, _SymSum)
        
        # Evaluate
        final = eval_lazy(result, {a: 3})
        assert final == 15  # max(3, 10) + 5 = 10 + 5
    
    def test_symmax_multiplication(self):
        """Test SymMax * number."""
        a = Symbol('a')
        m = SymMax(a, 10)
        
        result = m * 2
        assert isinstance(result, _SymProduct)
        
        # Evaluate
        final = eval_lazy(result, {a: 3})
        assert final == 20  # max(3, 10) * 2 = 10 * 2
    
    def test_symmax_division(self):
        """Test SymMax / number."""
        a = Symbol('a')
        m = SymMax(a, 10)
        
        result = m / 2
        assert isinstance(result, _SymQuotient)
        
        # Evaluate
        final = eval_lazy(result, {a: 3})
        assert final == 5  # max(3, 10) / 2 = 10 / 2


class TestSymMaxPerformance:
    """Tests for SymMax performance."""
    
    def test_deep_nesting_performance(self):
        """Test that deeply nested SymMax is fast to create."""
        import time
        
        a = Symbol('a')
        
        start = time.time()
        
        # Create 1000 nested max operations
        result = a
        for i in range(1000):
            result = sym_max(result, i)
        
        creation_time = time.time() - start
        
        # Should be fast (< 0.1s)
        assert creation_time < 0.1, f"Creation took {creation_time}s"
        
        # Evaluation should also be fast
        start = time.time()
        final = eval_lazy(result, {a: 500})
        eval_time = time.time() - start
        
        assert eval_time < 0.1, f"Evaluation took {eval_time}s"
        assert final == 999  # max of 0..999 and 500
    
    def test_cache_effectiveness(self):
        """Test that cache improves repeated evaluation."""
        import time
        
        clear_expr_cache()
        
        a, b = Symbol('a'), Symbol('b')
        
        # Create a moderately complex expression
        expr = SymMax(a, 10)
        for i in range(100):
            expr = sym_max(expr, b + i)
        
        subs = {a: 5, b: 3}
        
        # First evaluation (cache miss)
        start = time.time()
        result1 = eval_lazy(expr, subs)
        first_time = time.time() - start
        
        # Second evaluation with same subs (cache hit)
        start = time.time()
        result2 = eval_lazy(expr, subs)
        second_time = time.time() - start
        
        # Results should be the same
        assert result1 == result2
        
        # Second should be faster (or at least not slower)
        stats = get_cache_stats()
        assert stats["hits"] > 0, "Expected cache hits"
    
    def test_cache_with_different_subs(self):
        """Test cache with different substitutions."""
        clear_expr_cache()
        
        a = Symbol('a')
        expr = SymMax(a, 10)
        
        # Different subs should give different results
        r1 = eval_lazy(expr, {a: 5})
        r2 = eval_lazy(expr, {a: 15})
        r3 = eval_lazy(expr, {a: 5})  # Same as r1, should hit cache
        
        assert r1 == 10
        assert r2 == 15
        assert r3 == 10
        
        stats = get_cache_stats()
        # Should have at least 1 hit (for r3)
        assert stats["hits"] >= 1


class TestEvalLazy:
    """Tests for eval_lazy function."""
    
    def test_eval_numeric(self):
        """Test evaluating numeric values."""
        assert eval_lazy(5) == 5.0
        assert eval_lazy(3.14) == 3.14
    
    def test_eval_sympy(self):
        """Test evaluating SymPy expressions."""
        a = Symbol('a')
        result = eval_lazy(a * 2, {a: 5})
        assert result == 10.0
    
    def test_eval_symmax(self):
        """Test evaluating SymMax."""
        a = Symbol('a')
        m = SymMax(a, 10)
        
        result = eval_lazy(m, {a: 5})
        assert result == 10
    
    def test_eval_complex_expression(self):
        """Test evaluating complex expression with SymMax."""
        a, b = Symbol('a'), Symbol('b')
        
        # (max(a, 10) + max(b, 5)) * 2
        expr = (SymMax(a, 10) + SymMax(b, 5)) * 2
        
        result = eval_lazy(expr, {a: 3, b: 8})
        # max(3, 10) = 10, max(8, 5) = 8
        # (10 + 8) * 2 = 36
        assert result == 36


class TestIntegrationWithSchedule:
    """Integration tests with scheduling passes."""
    
    def test_schedule_with_symmax(self):
        """Test that scheduling works with SymMax."""
        from blueprinting.ir.schedule import ScheduleIR, ScheduledOp
        from blueprinting.ir.passes import EvaluatePass
        
        schedule = ScheduleIR()
        
        # Create op with symbolic duration
        a = Symbol('a')
        op = ScheduledOp(
            op_id="op1",
            op_type="Linear",
            device=0,
            start=0,
            duration=sym_max(a, 0.1)  # max(a, 0.1)
        )
        schedule.add_op(op)
        
        # Evaluate
        evaluate = EvaluatePass(subs={"a": 0.05})
        result = evaluate.run(schedule)
        
        # max(0.05, 0.1) = 0.1
        assert abs(result.e2e_time - 0.1) < 1e-6
    
    def test_schedule_makespan_with_symmax(self):
        """Test makespan calculation with SymMax durations."""
        from blueprinting.ir.schedule import ScheduleIR, ScheduledOp
        
        schedule = ScheduleIR()
        
        a = Symbol('a')
        
        # Op1: start=0, duration=0.1, end=0.1
        schedule.add_op(ScheduledOp(
            op_id="op1",
            op_type="Linear",
            device=0,
            start=0,
            duration=0.1
        ))
        
        # Op2: start=0.1, duration=max(a, 0.05), end=0.1+max(a, 0.05)
        schedule.add_op(ScheduledOp(
            op_id="op2",
            op_type="Linear",
            device=0,
            start=0.1,
            duration=sym_max(a, 0.05)
        ))
        
        # Makespan should be symbolic
        makespan = schedule.makespan()
        
        # After evaluation with a=0.01:
        # max(0.01, 0.05) = 0.05
        # makespan = 0.1 + 0.05 = 0.15
        result = eval_lazy(makespan, {a: 0.01})
        assert abs(result - 0.15) < 1e-6
        
        # After evaluation with a=0.2:
        # max(0.2, 0.05) = 0.2
        # makespan = 0.1 + 0.2 = 0.3
        result = eval_lazy(makespan, {a: 0.2})
        assert abs(result - 0.3) < 1e-6

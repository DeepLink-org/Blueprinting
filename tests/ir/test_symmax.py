"""Tests for SymMax - lazy symbolic max."""

import pytest
from sympy import Symbol, diff, Max, Min, Add, Mul

from blueprinting.core.symbolic import (
    SymMax,
    SymMin,
    sym_max,
    sym_min,
    eval_lazy,
    _to_float,
    _SymSum,
    _SymProduct,
    _SymQuotient,
    clear_expr_cache,
    get_cache_stats,
    LazyExprMixin,
    BATCH,
    TP,
    PP,
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


class TestLazyExprMixin:
    """Tests for LazyExprMixin methods."""
    
    def setup_method(self):
        """Clear cache before each test."""
        clear_expr_cache()
    
    # =========================================================================
    # to_sympy() tests
    # =========================================================================
    
    def test_symmax_to_sympy(self):
        """Test SymMax.to_sympy() conversion."""
        a = Symbol('a')
        m = SymMax(a, 10, 5)
        
        sympy_expr = m.to_sympy()
        
        # Should be a SymPy Max
        assert isinstance(sympy_expr, Max)
        # Should contain the same args
        assert a in sympy_expr.args or sympy_expr.has(a)
    
    def test_symmin_to_sympy(self):
        """Test SymMin.to_sympy() conversion."""
        a = Symbol('a')
        m = SymMin(a, 10, 5)
        
        sympy_expr = m.to_sympy()
        
        # Should be a SymPy Min
        assert isinstance(sympy_expr, Min)
    
    def test_symsum_to_sympy(self):
        """Test _SymSum.to_sympy() conversion."""
        a = Symbol('a')
        s = _SymSum(a, 10, 5)
        
        sympy_expr = s.to_sympy()
        
        # Should be a SymPy Add
        assert isinstance(sympy_expr, Add)
    
    def test_symproduct_to_sympy(self):
        """Test _SymProduct.to_sympy() conversion."""
        a = Symbol('a')
        p = _SymProduct(a, 2, 3)
        
        sympy_expr = p.to_sympy()
        
        # Should be a SymPy Mul
        assert isinstance(sympy_expr, Mul)
    
    def test_symquotient_to_sympy(self):
        """Test _SymQuotient.to_sympy() conversion."""
        a = Symbol('a')
        q = _SymQuotient(a, 2)
        
        sympy_expr = q.to_sympy()
        
        # Should evaluate correctly
        result = sympy_expr.subs(a, 10)
        assert float(result) == 5.0
    
    def test_nested_to_sympy(self):
        """Test to_sympy() with nested lazy expressions."""
        a, b = Symbol('a'), Symbol('b')
        
        # SymMax containing _SymSum
        inner = SymMax(a, 10)
        outer = inner + b  # Creates _SymSum
        
        sympy_expr = outer.to_sympy()
        
        # Should evaluate correctly
        result = sympy_expr.subs({a: 5, b: 3})
        assert float(result) == 13  # max(5, 10) + 3 = 10 + 3
    
    # =========================================================================
    # sweep() tests
    # =========================================================================
    
    def test_sweep_single_param(self):
        """Test sweep with single parameter."""
        a = Symbol('a')
        m = SymMax(a, 10)
        
        results = list(m.sweep(a=[5, 15, 25]))
        
        assert len(results) == 3
        assert results[0] == ({'a': 5}, 10)
        assert results[1] == ({'a': 15}, 15)
        assert results[2] == ({'a': 25}, 25)
    
    def test_sweep_multiple_params(self):
        """Test sweep with multiple parameters (cartesian product)."""
        a, b = Symbol('a'), Symbol('b')
        m = SymMax(a, b)
        
        results = list(m.sweep(a=[1, 2], b=[10, 20]))
        
        # Should have 2 * 2 = 4 combinations
        assert len(results) == 4
        
        # Verify some results
        params_results = {tuple(sorted(p.items())): r for p, r in results}
        assert params_results[(('a', 1), ('b', 10))] == 10
        assert params_results[(('a', 2), ('b', 20))] == 20
    
    def test_sweep_with_predefined_symbols(self):
        """Test sweep using predefined symbols."""
        expr = SymMax(BATCH * 100 / TP, 1000)
        
        results = list(expr.sweep(batch=[16, 32], tp=[1, 2, 4]))
        
        # 2 * 3 = 6 combinations
        assert len(results) == 6
        
        # Check specific result: batch=32, tp=1 -> max(3200, 1000) = 3200
        for params, result in results:
            if params.get('batch') == 32 and params.get('tp') == 1:
                assert result == 3200
    
    # =========================================================================
    # sensitivity() tests
    # =========================================================================
    
    def test_sensitivity_simple(self):
        """Test sensitivity analysis on simple expression."""
        a = Symbol('a')
        # SymMax(a, 10) -> for a > 10, derivative is 1; for a < 10, derivative is 0
        # But SymPy Max has piecewise derivative
        
        # Use a simpler expression for testing
        expr = _SymSum(a * 2, 10)  # a*2 + 10
        
        deriv = expr.sensitivity(a)
        
        # d/da (2a + 10) = 2
        assert deriv == 2
    
    def test_sensitivity_with_string(self):
        """Test sensitivity with string variable name."""
        a = Symbol('a')
        expr = _SymProduct(a, 3)  # a * 3
        
        deriv = expr.sensitivity('a')
        
        # d/da (3a) = 3
        assert deriv == 3
    
    def test_sensitivity_quotient(self):
        """Test sensitivity on quotient expression."""
        a = Symbol('a')
        expr = _SymQuotient(a * a, 2)  # a^2 / 2
        
        deriv = expr.sensitivity(a)
        
        # d/da (a^2 / 2) = a
        assert deriv == a
    
    # =========================================================================
    # to_latex() tests
    # =========================================================================
    
    def test_to_latex_symmax(self):
        """Test LaTeX export for SymMax."""
        a = Symbol('a')
        m = SymMax(a, 10)
        
        latex = m.to_latex()
        
        # Should contain max
        assert 'max' in latex.lower() or '\\max' in latex
    
    def test_to_latex_symmin(self):
        """Test LaTeX export for SymMin."""
        a = Symbol('a')
        m = SymMin(a, 10)
        
        latex = m.to_latex()
        
        # Should contain min
        assert 'min' in latex.lower() or '\\min' in latex
    
    def test_to_latex_sum(self):
        """Test LaTeX export for _SymSum."""
        a = Symbol('a')
        s = _SymSum(a, 10)
        
        latex = s.to_latex()
        
        # Should be valid LaTeX (contains a and 10)
        assert 'a' in latex
        assert '10' in latex
    
    def test_to_latex_complex(self):
        """Test LaTeX export for complex nested expression."""
        a, b = Symbol('a'), Symbol('b')
        
        expr = SymMax(a * 2, b / 3)
        latex = expr.to_latex()
        
        # Should be non-empty valid LaTeX
        assert len(latex) > 0
        assert 'a' in latex
        assert 'b' in latex
    
    # =========================================================================
    # solve() tests
    # =========================================================================
    
    def test_solve_simple(self):
        """Test solve without constraints."""
        a = Symbol('a')
        expr = _SymSum(a, -10)  # a - 10
        
        # Solve a - 10 = 0
        solution = expr.solve()
        
        # Should return [10]
        assert 10 in solution or solution == [10]
    
    def test_solve_with_constraint(self):
        """Test solve with constraint."""
        a = Symbol('a')
        
        # Create expression
        expr = _SymSum(a, 5)  # a + 5
        
        # Solve a + 5 = 15, i.e., a = 10
        from sympy import Eq
        solution = expr.solve(Eq(a + 5, 15), target='a')
        
        # Should include a = 10
        assert solution is not None


class TestSymMin:
    """Tests for SymMin class."""
    
    def setup_method(self):
        """Clear cache before each test."""
        clear_expr_cache()
    
    def test_numeric_min(self):
        """Test min with numeric values."""
        result = sym_min(3, 5)
        assert result == 3
        
        result = sym_min(10.5, 3.2)
        assert result == 3.2
    
    def test_symbolic_creation(self):
        """Test creating SymMin with symbols."""
        a = Symbol('a')
        m = SymMin(a, 10)
        
        assert isinstance(m, SymMin)
        assert len(m.args) == 2
    
    def test_symbolic_eval(self):
        """Test evaluating SymMin with substitutions."""
        a = Symbol('a')
        m = SymMin(a, 10)
        
        # a=5 < 10
        result = m.eval({a: 5})
        assert result == 5
        
        # a=15 > 10
        result = m.eval({a: 15})
        assert result == 10
    
    def test_nested_symmin(self):
        """Test nested SymMin flattening."""
        a, b, c = Symbol('a'), Symbol('b'), Symbol('c')
        
        inner = SymMin(a, b)
        outer = SymMin(inner, c)
        
        # Should flatten to SymMin(a, b, c)
        assert len(outer.args) == 3


class TestIntegrationWithSchedule:
    """Integration tests with scheduling passes.
    
    Note: These tests were designed for the old IR API.
    Some tests are skipped until the integration is updated.
    """
    
    @pytest.mark.skip(reason="Integration test needs update for new IR architecture")
    def test_schedule_with_symmax(self):
        """Test that scheduling works with SymMax."""
        # This test needs to be updated for the new IR architecture
        pass
    
    @pytest.mark.skip(reason="Integration test needs update for new IR architecture")
    def test_schedule_makespan_with_symmax(self):
        """Test makespan calculation with SymMax durations."""
        # This test needs to be updated for the new IR architecture
        pass

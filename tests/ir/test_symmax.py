"""Tests for SymMax / SymMin - lazy symbolic max/min (sympy.Function).

Imports only from blueprinting.core (no blueprinting.core.symbolic, no _to_float).
"""

from sympy import Add, Symbol

from blueprinting.core import (
    BATCH,
    TP,
    SymMax,
    SymMin,
    eval_lazy,
    sym_max,
    sym_min,
)


class TestSymMax:
    """Tests for SymMax class."""

    def test_numeric_max(self):
        """Test max with numeric values."""
        assert sym_max(3, 5) == 5
        assert sym_max(10.5, 3.2) == 10.5

    def test_symbolic_creation(self):
        """Test creating SymMax with symbols."""
        a = Symbol("a")
        m = SymMax(a, 10)
        assert isinstance(m, SymMax)
        assert len(m.args) == 2

    def test_symbolic_subs(self):
        """Test evaluating SymMax with substitutions."""
        a = Symbol("a")
        m = SymMax(a, 10)
        assert float(m.subs(a, 5)) == 10
        assert float(m.subs(a, 15)) == 15

    def test_nested_symmax(self):
        """Test nested SymMax flattening."""
        a, b, c = Symbol("a"), Symbol("b"), Symbol("c")
        inner = SymMax(a, b)
        outer = SymMax(inner, c)
        assert len(outer.args) == 3

    def test_symmax_chain_eval_lazy(self):
        """Test evaluating chain of SymMax via eval_lazy."""
        a, b = Symbol("a"), Symbol("b")
        m1 = sym_max(a, 5)
        m2 = sym_max(m1, b)
        assert eval_lazy(m2, {a: 3, b: 4}) == 5
        assert eval_lazy(m2, {a: 10, b: 4}) == 10
        assert eval_lazy(m2, {a: 3, b: 20}) == 20

    def test_partial_subs(self):
        """Test partial substitution (some symbols remain)."""
        a, b = Symbol("a"), Symbol("b")
        m = SymMax(a, b)
        result = m.subs(a, 10)
        assert isinstance(result, SymMax)
        assert 10 in result.args and b in result.args

    def test_symmax_arithmetic_sympy(self):
        """Test SymMax + number returns SymPy Add."""
        a = Symbol("a")
        m = SymMax(a, 10)
        result = m + 5
        assert isinstance(result, Add)
        assert float(eval_lazy(result, {a: 3})) == 15

    def test_symmax_mul_sympy(self):
        """Test SymMax * number."""
        a = Symbol("a")
        m = SymMax(a, 10)
        result = m * 2
        assert float(eval_lazy(result, {a: 3})) == 20

    def test_symmax_div_sympy(self):
        """Test SymMax / number."""
        a = Symbol("a")
        m = SymMax(a, 10)
        result = m / 2
        assert float(eval_lazy(result, {a: 3})) == 5


class TestSymMaxPerformance:
    """Tests for SymMax performance."""

    def test_deep_nesting_performance(self):
        """Test that deeply nested SymMax is fast to create."""
        import time

        a = Symbol("a")
        start = time.time()
        result = a
        for i in range(1000):
            result = sym_max(result, i)
        creation_time = time.time() - start
        assert creation_time < 0.5, f"Creation took {creation_time}s"

        start = time.time()
        final = eval_lazy(result, {a: 500})
        eval_time = time.time() - start
        assert eval_time < 0.5, f"Evaluation took {eval_time}s"
        assert final == 999


class TestEvalLazy:
    """Tests for eval_lazy function."""

    def test_eval_numeric(self):
        """Test evaluating numeric values."""
        assert eval_lazy(5) == 5.0
        assert eval_lazy(3.14) == 3.14

    def test_eval_sympy(self):
        """Test evaluating SymPy expressions."""
        a = Symbol("a")
        assert eval_lazy(a * 2, {a: 5}) == 10.0

    def test_eval_symmax(self):
        """Test evaluating SymMax."""
        a = Symbol("a")
        m = SymMax(a, 10)
        assert eval_lazy(m, {a: 5}) == 10

    def test_eval_complex_expression(self):
        """Test evaluating complex expression with SymMax."""
        a, b = Symbol("a"), Symbol("b")
        expr = (SymMax(a, 10) + SymMax(b, 5)) * 2
        result = eval_lazy(expr, {a: 3, b: 8})
        assert result == 36


class TestSymMaxLatex:
    """Tests for SymMax/SymMin LaTeX (via SymPy)."""

    def test_symmax_latex(self):
        """Test LaTeX export for SymMax."""
        from sympy import latex

        a = Symbol("a")
        m = SymMax(a, 10)
        s = latex(m)
        assert "max" in s.lower() or "\\max" in s

    def test_symmin_latex(self):
        """Test LaTeX export for SymMin."""
        from sympy import latex

        a = Symbol("a")
        m = SymMin(a, 10)
        s = latex(m)
        assert "min" in s.lower() or "\\min" in s


class TestSymMin:
    """Tests for SymMin class."""

    def test_numeric_min(self):
        """Test min with numeric values."""
        assert sym_min(3, 5) == 3
        assert sym_min(10.5, 3.2) == 3.2

    def test_symbolic_creation(self):
        """Test creating SymMin with symbols."""
        a = Symbol("a")
        m = SymMin(a, 10)
        assert isinstance(m, SymMin)
        assert len(m.args) == 2

    def test_symbolic_subs(self):
        """Test evaluating SymMin with substitutions."""
        a = Symbol("a")
        m = SymMin(a, 10)
        assert float(m.subs(a, 5)) == 5
        assert float(m.subs(a, 15)) == 10

    def test_nested_symmin(self):
        """Test nested SymMin flattening."""
        a, b, c = Symbol("a"), Symbol("b"), Symbol("c")
        inner = SymMin(a, b)
        outer = SymMin(inner, c)
        assert len(outer.args) == 3


class TestPredefinedSymbols:
    """Tests using predefined symbols (BATCH, TP)."""

    def test_symmax_with_predefined_symbols(self):
        """Test SymMax with BATCH, TP from core."""
        expr = SymMax(BATCH * 100 / TP, 1000)
        result = eval_lazy(expr, {"batch": 32, "tp": 1})
        assert result == 3200

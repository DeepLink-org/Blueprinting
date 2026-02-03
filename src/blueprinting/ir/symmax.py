"""SymMax - Lazy symbolic max that avoids SymPy's expensive simplification.

The key insight is that SymPy's Max function triggers complex algebraic
simplification (factor_terms, gcd_terms) which is O(exponential) for
deeply nested expressions.

Solution: Create a simple container that stores the arguments without
any simplification, and only evaluates the max when concrete values
are available.

Performance optimization: Global expression cache to avoid redundant
evaluation of the same sub-expressions with the same substitutions.
"""

from typing import Any, Dict, Tuple, Union

from sympy import Expr, Symbol

# Global expression cache
# Key: (id(expr), frozenset of subs items)
# Value: evaluated result
# Note: Use id() for O(1) key creation. Cache should be cleared between compilations.
_EXPR_CACHE: Dict[Tuple[int, frozenset], Any] = {}
_CACHE_HITS = 0
_CACHE_MISSES = 0
_MAX_CACHE_SIZE = 50000


def _cache_key(expr, subs: dict) -> Tuple[int, frozenset]:
    """Create a cache key from expression and substitutions.

    Uses id(expr) for O(1) key creation. This is safe because:
    1. Within a single compilation, the same object should give the same result
    2. Cache is cleared between compilations via clear_expr_cache()
    """
    return (id(expr), frozenset(subs.items()) if subs else frozenset())


def _cache_get(expr, subs: dict):
    """Get cached result if available."""
    global _CACHE_HITS
    key = _cache_key(expr, subs)
    if key in _EXPR_CACHE:
        _CACHE_HITS += 1
        return _EXPR_CACHE[key], True
    return None, False


def _cache_put(expr, subs: dict, result):
    """Store result in cache."""
    global _CACHE_MISSES
    _CACHE_MISSES += 1

    # Simple LRU-like eviction: clear half when full
    if len(_EXPR_CACHE) >= _MAX_CACHE_SIZE:
        # Clear oldest half (dict maintains insertion order in Python 3.7+)
        keys_to_remove = list(_EXPR_CACHE.keys())[: _MAX_CACHE_SIZE // 2]
        for k in keys_to_remove:
            del _EXPR_CACHE[k]

    key = _cache_key(expr, subs)
    _EXPR_CACHE[key] = result


def clear_expr_cache():
    """Clear the expression cache."""
    global _EXPR_CACHE, _CACHE_HITS, _CACHE_MISSES
    _EXPR_CACHE.clear()
    _CACHE_HITS = 0
    _CACHE_MISSES = 0


def get_cache_stats() -> dict:
    """Get cache statistics."""
    total = _CACHE_HITS + _CACHE_MISSES
    return {
        "hits": _CACHE_HITS,
        "misses": _CACHE_MISSES,
        "hit_rate": _CACHE_HITS / total if total > 0 else 0,
        "size": len(_EXPR_CACHE),
    }


class SymMax:
    """Lazy max that defers evaluation until values are known.

    Unlike sympy.Max, this class:
    1. Never triggers automatic simplification
    2. Stores arguments as-is
    3. Evaluates to numeric max when all values are concrete
    4. Supports nested SymMax (flattens automatically)

    Example:
        >>> a = Symbol('a')
        >>> m = SymMax(a, 10)
        >>> m.eval({a: 5})
        10
        >>> m.eval({a: 15})
        15
    """

    __slots__ = ("_args",)

    def __init__(self, *args):
        """Create a lazy max expression.

        Args:
            *args: Values to take max of (numeric or symbolic)
        """
        # Flatten nested SymMax
        flat_args = []
        for arg in args:
            if isinstance(arg, SymMax):
                flat_args.extend(arg._args)
            else:
                flat_args.append(arg)
        self._args: Tuple[Any, ...] = tuple(flat_args)

    @property
    def args(self) -> Tuple[Any, ...]:
        return self._args

    def eval(self, subs: dict = None) -> Union[float, "SymMax"]:
        """Evaluate the max with given substitutions.

        Args:
            subs: Dictionary mapping symbols to values

        Returns:
            Numeric max if all args become numeric, else new SymMax
        """
        subs = subs or {}

        # Check cache first
        cached, hit = _cache_get(self, subs)
        if hit:
            return cached

        # Substitute and collect values
        evaluated = []
        for arg in self._args:
            val = self._substitute(arg, subs)
            evaluated.append(val)

        # Check if all are numeric
        numeric_vals = []
        for val in evaluated:
            num = _to_float(val)
            if num is not None:
                numeric_vals.append(num)
            else:
                # Still symbolic - return new SymMax
                result = SymMax(*evaluated)
                _cache_put(self, subs, result)
                return result

        result = max(numeric_vals)
        _cache_put(self, subs, result)
        return result

    def _substitute(self, expr: Any, subs: dict) -> Any:
        """Substitute symbols in an expression."""
        if isinstance(expr, (int, float)):
            return expr
        if isinstance(expr, (SymMax, _SymSum, _SymProduct, _SymQuotient)):
            return expr.eval(subs)
        if isinstance(expr, Expr):
            result = expr.subs(subs)
            # Try to convert to float
            num = _to_float(result)
            return num if num is not None else result
        if isinstance(expr, Symbol):
            return subs.get(expr, expr)
        return expr

    def __add__(self, other):
        """Support addition: SymMax(a, b) + c"""
        if isinstance(other, SymMax):
            # (max(a,b) + max(c,d)) - keep as symbolic sum
            return _SymSum(self, other)
        if isinstance(other, (int, float)):
            if other == 0:
                return self
            return _SymSum(self, other)
        # For sympy Expr
        return _SymSum(self, other)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            if other == 1:
                return self
            if other == 0:
                return 0
        return _SymProduct(self, other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (int, float)) and other == 1:
            return self
        return _SymQuotient(self, other)

    def __repr__(self):
        args_str = ", ".join(str(a) for a in self._args[:3])
        if len(self._args) > 3:
            args_str += f", ... ({len(self._args)} total)"
        return f"SymMax({args_str})"

    def __eq__(self, other):
        if isinstance(other, SymMax):
            return self._args == other._args
        return False

    def __hash__(self):
        return hash(("SymMax", self._args))


class _SymSum:
    """Lazy sum that defers evaluation."""

    __slots__ = ("_terms",)

    def __init__(self, *terms):
        flat = []
        for t in terms:
            if isinstance(t, _SymSum):
                flat.extend(t._terms)
            else:
                flat.append(t)
        self._terms = tuple(flat)

    def eval(self, subs: dict = None) -> Union[float, "_SymSum"]:
        subs = subs or {}

        # Check cache first
        cached, hit = _cache_get(self, subs)
        if hit:
            return cached

        total = 0
        remaining = []

        for term in self._terms:
            if isinstance(term, (SymMax, _SymSum, _SymProduct, _SymQuotient)):
                val = term.eval(subs)
            elif isinstance(term, Expr):
                val = term.subs(subs)
            else:
                val = term

            num = _to_float(val)
            if num is not None:
                total += num
            else:
                remaining.append(val)

        if not remaining:
            result = total
        elif total != 0:
            remaining.append(total)
            result = remaining[0] if len(remaining) == 1 else _SymSum(*remaining)
        elif len(remaining) == 1:
            result = remaining[0]
        else:
            result = _SymSum(*remaining)

        _cache_put(self, subs, result)
        return result

    def __add__(self, other):
        return _SymSum(self, other)

    def __radd__(self, other):
        return _SymSum(other, self)

    def __mul__(self, other):
        return _SymProduct(self, other)

    def __rmul__(self, other):
        return _SymProduct(other, self)

    def __truediv__(self, other):
        return _SymQuotient(self, other)

    def __repr__(self):
        return f"Sum({', '.join(str(t) for t in self._terms[:3])}{'...' if len(self._terms) > 3 else ''})"


class _SymProduct:
    """Lazy product that defers evaluation."""

    __slots__ = ("_factors",)

    def __init__(self, *factors):
        self._factors = factors

    def eval(self, subs: dict = None) -> Union[float, "_SymProduct"]:
        subs = subs or {}

        # Check cache first
        cached, hit = _cache_get(self, subs)
        if hit:
            return cached

        product = 1
        remaining = []

        for factor in self._factors:
            if isinstance(factor, (SymMax, _SymSum, _SymProduct, _SymQuotient)):
                val = factor.eval(subs)
            elif isinstance(factor, Expr):
                val = factor.subs(subs)
            else:
                val = factor

            num = _to_float(val)
            if num is not None:
                product *= num
            else:
                remaining.append(val)

        if not remaining:
            result = product
        elif product != 1:
            remaining.append(product)
            result = remaining[0] if len(remaining) == 1 else _SymProduct(*remaining)
        elif len(remaining) == 1:
            result = remaining[0]
        else:
            result = _SymProduct(*remaining)

        _cache_put(self, subs, result)
        return result

    def __repr__(self):
        return f"Product({self._factors})"


class _SymQuotient:
    """Lazy quotient that defers evaluation."""

    __slots__ = ("_num", "_denom")

    def __init__(self, num, denom):
        self._num = num
        self._denom = denom

    def eval(self, subs: dict = None) -> Union[float, "_SymQuotient"]:
        subs = subs or {}

        # Check cache first
        cached, hit = _cache_get(self, subs)
        if hit:
            return cached

        if isinstance(self._num, (SymMax, _SymSum, _SymProduct, _SymQuotient)):
            num_val = self._num.eval(subs)
        elif isinstance(self._num, Expr):
            num_val = self._num.subs(subs)
        else:
            num_val = self._num

        if isinstance(self._denom, (SymMax, _SymSum, _SymProduct, _SymQuotient)):
            denom_val = self._denom.eval(subs)
        elif isinstance(self._denom, Expr):
            denom_val = self._denom.subs(subs)
        else:
            denom_val = self._denom

        num_float = _to_float(num_val)
        denom_float = _to_float(denom_val)

        if num_float is not None and denom_float is not None:
            result = num_float / denom_float if denom_float != 0 else float("inf")
        else:
            result = _SymQuotient(num_val, denom_val)

        _cache_put(self, subs, result)
        return result

    def __repr__(self):
        return f"({self._num} / {self._denom})"


def _to_float(x, subs: dict = None) -> Union[float, None]:
    """Try to convert x to Python float.

    Args:
        x: Value to convert
        subs: Optional substitutions for evaluating lazy expressions

    Returns None if x is symbolic/cannot be converted.
    """
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, (SymMax, _SymSum, _SymProduct, _SymQuotient)):
        if subs:
            # Try evaluating with subs
            result = x.eval(subs)
            if isinstance(result, (int, float)):
                return float(result)
        return None  # Still lazy
    if hasattr(x, "is_number") and x.is_number:
        try:
            return float(x)
        except (TypeError, ValueError):
            return None
    return None


def sym_max(a, b) -> Union[float, SymMax]:
    """Create a lazy max expression.

    Fast path for numeric values, lazy SymMax for symbolic.

    Args:
        a: First value
        b: Second value

    Returns:
        max(a, b) if both numeric, SymMax(a, b) otherwise
    """
    a_num = _to_float(a)
    b_num = _to_float(b)

    if a_num is not None and b_num is not None:
        return max(a_num, b_num)

    return SymMax(a, b)


def eval_lazy(expr, subs: dict = None) -> Union[float, Any]:
    """Evaluate a lazy expression (SymMax, _SymSum, etc).

    Args:
        expr: Expression to evaluate (can be numeric, SymMax, SymPy Expr)
        subs: Symbol substitutions (keys can be Symbol or str)

    Returns:
        Numeric result if fully evaluated, else remaining symbolic expr
    """
    subs = subs or {}

    # Normalize subs: convert string keys to Symbol
    normalized_subs = {}
    for k, v in subs.items():
        if isinstance(k, str):
            normalized_subs[Symbol(k)] = v
        else:
            normalized_subs[k] = v

    if isinstance(expr, (int, float)):
        return float(expr)

    if isinstance(expr, (SymMax, _SymSum, _SymProduct, _SymQuotient)):
        return expr.eval(normalized_subs)

    if isinstance(expr, Expr):
        result = expr.subs(normalized_subs)
        num = _to_float(result)
        return num if num is not None else result

    return expr

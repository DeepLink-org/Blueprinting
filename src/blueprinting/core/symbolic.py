"""符号表达式扩展

提供扩展的符号表达式能力，作为模拟器的计算基础。

核心设计理念:
- 惰性求值: 延迟计算直到所有符号具体化
- 避免 SymPy 开销: SymMax/SymMin 等避免触发昂贵的代数简化
- 统一接口: eval_lazy() 统一处理各种表达式类型

包含:
- 预定义符号: BATCH, SEQ, HIDDEN, TP, PP, DP 等
- SymPick: 符号条件表达式 (if-then-else)
- SymMax/SymMin: 惰性最大/最小值
- 工具函数: sym_max, sym_min, eval_lazy, is_symbolic, free_symbols
"""
from typing import Any, Dict, Set, Tuple, Union

from sympy import Expr, S, Symbol, sympify


# ==============================================================================
# 预定义符号常量
# ==============================================================================

# 模型参数符号
BATCH = Symbol("batch", positive=True, integer=True)
SEQ = Symbol("seq", positive=True, integer=True)
HIDDEN = Symbol("hidden", positive=True, integer=True)
FEEDFORWARD = Symbol("feedforward", positive=True, integer=True)
NUM_LAYERS = Symbol("num_layers", positive=True, integer=True)
ATTN_HEADS = Symbol("attn_heads", positive=True, integer=True)
HEAD_DIM = Symbol("head_dim", positive=True, integer=True)
VOCAB_SIZE = Symbol("vocab_size", positive=True, integer=True)

# 并行策略符号
TP = Symbol("tp", positive=True, integer=True)  # Tensor Parallel
PP = Symbol("pp", positive=True, integer=True)  # Pipeline Parallel
DP = Symbol("dp", positive=True, integer=True)  # Data Parallel
CP = Symbol("cp", positive=True, integer=True)  # Context Parallel
EP = Symbol("ep", positive=True, integer=True)  # Expert Parallel

# 执行参数符号
MICRO_BATCH = Symbol("micro_batch", positive=True, integer=True)
NUM_MICRO_BATCHES = Symbol("num_micro_batches", positive=True, integer=True)

# 硬件参数符号
PEAK_FLOPS = Symbol("peak_flops", positive=True)
MEM_BANDWIDTH = Symbol("mem_bandwidth", positive=True)
NET_BANDWIDTH = Symbol("net_bandwidth", positive=True)

# 符号分组，方便批量操作
MODEL_SYMBOLS = {BATCH, SEQ, HIDDEN, FEEDFORWARD, NUM_LAYERS, ATTN_HEADS, HEAD_DIM, VOCAB_SIZE}
PARALLEL_SYMBOLS = {TP, PP, DP, CP, EP}
EXEC_SYMBOLS = {MICRO_BATCH, NUM_MICRO_BATCHES}
HARDWARE_SYMBOLS = {PEAK_FLOPS, MEM_BANDWIDTH, NET_BANDWIDTH}
ALL_SYMBOLS = MODEL_SYMBOLS | PARALLEL_SYMBOLS | EXEC_SYMBOLS | HARDWARE_SYMBOLS


def get_symbol(name: str) -> Symbol:
    """根据名称获取预定义符号，不存在则创建新符号."""
    symbol_map = {s.name: s for s in ALL_SYMBOLS}
    return symbol_map.get(name, Symbol(name))


class SymPick(Expr):
    """符号条件表达式
    
    SymPick(condition, true_expr, false_expr)
    
    - 条件为符号时保持惰性求值
    - subs() 时自动选择分支
    - 可参与其他符号运算
    
    Examples
    --------
    >>> from sympy import Symbol
    >>> tp = Symbol('tp')
    >>> expr = SymPick(tp > 1, tp * 100, 0)
    >>> expr.subs(tp, 4)
    400
    >>> expr.subs(tp, 1)
    0
    
    嵌套使用:
    >>> sp = Symbol('sp')
    >>> nested = SymPick(tp > 1, SymPick(sp, tp * 2, tp), 0)
    >>> nested.subs([(tp, 4), (sp, True)])
    8
    """
    
    @property
    def cond(self):
        """条件表达式"""
        return self.args[0]
    
    @property
    def true_expr(self):
        """条件为真时的表达式"""
        return self.args[1]
    
    @property
    def false_expr(self):
        """条件为假时的表达式"""
        return self.args[2]
    
    def __new__(cls, cond, true_expr, false_expr):
        # 将参数转换为 sympy 对象
        cond = sympify(cond)
        true_expr = sympify(true_expr)
        false_expr = sympify(false_expr)
        
        # 条件可确定时直接返回结果
        if cond == S.true or cond is True:
            return true_expr
        if cond == S.false or cond is False:
            return false_expr
        
        # 尝试直接求值布尔条件
        try:
            if bool(cond):
                return true_expr
            return false_expr
        except TypeError:
            # 条件含符号，无法求值，保持 SymPick 形式
            pass
        
        return super().__new__(cls, cond, true_expr, false_expr)
    
    def _eval_subs(self, old, new):
        """替换后重新求值条件"""
        cond = self.cond.subs(old, new)
        true_expr = self.true_expr.subs(old, new)
        false_expr = self.false_expr.subs(old, new)
        
        # 尝试求值布尔条件
        try:
            if bool(cond):
                return true_expr
            return false_expr
        except TypeError:
            # 条件仍含符号，保持 SymPick 形式
            return SymPick(cond, true_expr, false_expr)
    
    def _sympystr(self, printer):
        """字符串表示"""
        return f"SymPick({printer.doprint(self.cond)}, {printer.doprint(self.true_expr)}, {printer.doprint(self.false_expr)})"
    
    def _latex(self, printer):
        """LaTeX 表示"""
        cond = printer.doprint(self.cond)
        t = printer.doprint(self.true_expr)
        f = printer.doprint(self.false_expr)
        return rf"\begin{{cases}} {t} & \text{{if }} {cond} \\ {f} & \text{{otherwise}} \end{{cases}}"


# ==============================================================================
# SymMax - 惰性符号最大值
# ==============================================================================

# Global expression cache
# Key: (id(expr), frozenset of subs items)
# Value: evaluated result
# Note: Use id() for O(1) key creation. Cache should be cleared between compilations.
_EXPR_CACHE: Dict[Tuple[int, frozenset], Any] = {}
_CACHE_HITS = 0
_CACHE_MISSES = 0
_MAX_CACHE_SIZE = 50000


def _cache_key(expr, subs: dict) -> Tuple[int, frozenset]:
    """Create a cache key from expression and substitutions."""
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

    if len(_EXPR_CACHE) >= _MAX_CACHE_SIZE:
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


# ==============================================================================
# LazyExprMixin - 惰性表达式分析方法
# ==============================================================================


class LazyExprMixin:
    """惰性表达式分析方法 Mixin.
    
    为惰性表达式类 (SymMax, SymMin, SymSum, SymProduct, SymQuotient) 
    提供参数空间探索、敏感度分析、约束求解等高级功能。
    
    Example:
        >>> from blueprinting.core import SymMax, TP, BATCH
        >>> 
        >>> comm_time = BATCH * 1000 / TP
        >>> compute_time = BATCH * 100
        >>> total = SymMax(comm_time, compute_time)
        >>> 
        >>> # 参数空间扫描
        >>> for params, result in total.sweep(tp=[1,2,4], batch=[16,32]):
        ...     print(f"{params} -> {result}")
        >>> 
        >>> # 敏感度分析
        >>> d_tp = total.sensitivity('tp')
        >>> 
        >>> # 导出 LaTeX
        >>> print(total.to_latex())
    """
    
    def sweep(self, **param_ranges):
        """参数空间扫描.
        
        对给定的参数范围进行笛卡尔积扫描，返回每组参数对应的求值结果。
        
        Args:
            **param_ranges: 参数范围，如 tp=[1,2,4,8], batch=[16,32,64]
            
        Yields:
            (params_dict, result) 元组
            
        Example:
            >>> expr = SymMax(BATCH / TP, 100)
            >>> list(expr.sweep(tp=[1, 2], batch=[16, 32]))
            [({'tp': 1, 'batch': 16}, 100), ({'tp': 1, 'batch': 32}, 100), 
             ({'tp': 2, 'batch': 16}, 100), ({'tp': 2, 'batch': 32}, 100)]
        """
        from itertools import product
        
        keys = list(param_ranges.keys())
        values_list = list(param_ranges.values())
        
        # 将字符串 key 转换为 Symbol
        symbol_keys = [get_symbol(k) if isinstance(k, str) else k for k in keys]
        
        for values in product(*values_list):
            # 原始参数字典（用于返回，保持用户传入的 key 类型）
            params = dict(zip(keys, values))
            # Symbol 参数字典（用于 eval）
            symbol_params = dict(zip(symbol_keys, values))
            result = self.eval(symbol_params)
            yield params, result
    
    def sensitivity(self, wrt: Union[str, Symbol]) -> Expr:
        """敏感度分析 - 对指定符号求偏导.
        
        将惰性表达式转换为 SymPy 表达式后求偏导数，
        用于分析参数变化对结果的边际影响。
        
        Args:
            wrt: 求导变量名 (str) 或 Symbol
            
        Returns:
            偏导数表达式
            
        Example:
            >>> expr = SymMax(BATCH * 100 / TP, BATCH * 50)
            >>> d_tp = expr.sensitivity('tp')
            >>> # 可进一步求值: d_tp.subs({'tp': 4, 'batch': 32})
        """
        from sympy import diff
        
        sym = get_symbol(wrt) if isinstance(wrt, str) else wrt
        sympy_expr = self.to_sympy()
        return diff(sympy_expr, sym)
    
    def solve(self, *constraints, target: str = None, minimize: bool = True):
        """约束求解.
        
        在给定约束条件下求解符号，可选择最小化/最大化目标。
        
        Args:
            *constraints: 约束条件表达式（如 memory < 80e9）
            target: 目标变量名（可选）
            minimize: True 最小化目标，False 最大化
            
        Returns:
            满足约束的解 (dict) 或解集 (list)
            
        Note:
            对于复杂约束，可能返回符号解或无解。
            建议结合 sweep() 进行数值探索。
        """
        from sympy import solve as sympy_solve
        
        sympy_expr = self.to_sympy()
        
        if not constraints:
            # 无约束，求解表达式 = 0
            return sympy_solve(sympy_expr)
        
        # 有约束，联合求解
        all_constraints = list(constraints)
        if target:
            target_sym = get_symbol(target) if isinstance(target, str) else target
            return sympy_solve(all_constraints, target_sym)
        
        return sympy_solve(all_constraints)
    
    def to_latex(self) -> str:
        """导出 LaTeX 公式.
        
        将惰性表达式转换为 LaTeX 格式的数学公式。
        
        Returns:
            LaTeX 字符串
            
        Example:
            >>> expr = SymMax(BATCH * 100, TP * 50)
            >>> print(expr.to_latex())
            \\max\\left(100 batch, 50 tp\\right)
        """
        from sympy import latex
        
        sympy_expr = self.to_sympy()
        return latex(sympy_expr)
    
    def to_sympy(self) -> Expr:
        """转换为纯 SymPy 表达式.
        
        子类必须实现此方法，将惰性表达式转换为对应的 SymPy 表达式。
        
        Returns:
            SymPy Expr 对象
        """
        raise NotImplementedError("Subclass must implement to_sympy()")
    
    # =========================================================================
    # 通用辅助方法（子类可直接使用）
    # =========================================================================
    
    @staticmethod
    def _convert_to_sympy(arg):
        """将参数转换为 SymPy 表达式（用于 to_sympy）."""
        if hasattr(arg, 'to_sympy'):
            return arg.to_sympy()
        return sympify(arg)
    
    def _substitute_arg(self, expr: Any, subs: dict) -> Any:
        """对单个参数执行替换."""
        if isinstance(expr, (int, float)):
            return expr
        # 惰性表达式类型都有 eval 和 args 属性（使用公有 args 而非私有 _args）
        if hasattr(expr, 'eval') and hasattr(expr, 'args'):
            return expr.eval(subs)
        if isinstance(expr, Expr):
            result = expr.subs(subs)
            num = _to_float(result)
            return num if num is not None else result
        if isinstance(expr, Symbol):
            return subs.get(expr, expr)
        return expr
    
    # =========================================================================
    # 通用算术运算（子类可覆盖）
    # =========================================================================
    
    def __add__(self, other):
        if isinstance(other, (int, float)) and other == 0:
            return self
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


class SymMax(LazyExprMixin):
    """惰性最大值，延迟求值直到所有值已知.

    与 sympy.Max 不同，此类:
    1. 不触发自动简化
    2. 原样存储参数
    3. 所有值为具体数值时才求值
    4. 支持嵌套（自动展平）

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
        """Evaluate the max with given substitutions."""
        subs = subs or {}

        cached, hit = _cache_get(self, subs)
        if hit:
            return cached

        evaluated = [self._substitute_arg(arg, subs) for arg in self._args]

        numeric_vals = []
        for val in evaluated:
            num = _to_float(val)
            if num is not None:
                numeric_vals.append(num)
            else:
                result = SymMax(*evaluated)
                _cache_put(self, subs, result)
                return result

        result = max(numeric_vals)
        _cache_put(self, subs, result)
        return result

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

    def to_sympy(self) -> Expr:
        """转换为 SymPy Max 表达式."""
        from sympy import Max
        return Max(*[self._convert_to_sympy(a) for a in self._args])


class SymMin(LazyExprMixin):
    """惰性最小值，延迟求值直到所有值已知.

    与 SymMax 对称，用于计算最小值。

    Example:
        >>> a = Symbol('a')
        >>> m = SymMin(a, 10)
        >>> m.eval({a: 5})
        5
        >>> m.eval({a: 15})
        10
    """

    __slots__ = ("_args",)

    def __init__(self, *args):
        flat_args = []
        for arg in args:
            if isinstance(arg, SymMin):
                flat_args.extend(arg._args)
            else:
                flat_args.append(arg)
        self._args: Tuple[Any, ...] = tuple(flat_args)

    @property
    def args(self) -> Tuple[Any, ...]:
        return self._args

    def eval(self, subs: dict = None) -> Union[float, "SymMin"]:
        """Evaluate the min with given substitutions."""
        subs = subs or {}

        cached, hit = _cache_get(self, subs)
        if hit:
            return cached

        evaluated = [self._substitute_arg(arg, subs) for arg in self._args]

        numeric_vals = []
        for val in evaluated:
            num = _to_float(val)
            if num is not None:
                numeric_vals.append(num)
            else:
                result = SymMin(*evaluated)
                _cache_put(self, subs, result)
                return result

        result = min(numeric_vals)
        _cache_put(self, subs, result)
        return result

    def __repr__(self):
        args_str = ", ".join(str(a) for a in self._args[:3])
        if len(self._args) > 3:
            args_str += f", ... ({len(self._args)} total)"
        return f"SymMin({args_str})"

    def __eq__(self, other):
        if isinstance(other, SymMin):
            return self._args == other._args
        return False

    def __hash__(self):
        return hash(("SymMin", self._args))

    def to_sympy(self) -> Expr:
        """转换为 SymPy Min 表达式."""
        from sympy import Min
        return Min(*[self._convert_to_sympy(a) for a in self._args])


class _SymSum(LazyExprMixin):
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

    @property
    def args(self) -> Tuple[Any, ...]:
        return self._terms

    def eval(self, subs: dict = None) -> Union[float, "_SymSum"]:
        subs = subs or {}

        cached, hit = _cache_get(self, subs)
        if hit:
            return cached

        total = 0
        remaining = []

        for term in self._terms:
            val = self._substitute_arg(term, subs)
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

    def __repr__(self):
        terms_str = ', '.join(str(t) for t in self._terms[:3])
        return f"Sum({terms_str}{'...' if len(self._terms) > 3 else ''})"

    def to_sympy(self) -> Expr:
        """转换为 SymPy Add 表达式."""
        from sympy import Add
        return Add(*[self._convert_to_sympy(t) for t in self._terms])


class _SymProduct(LazyExprMixin):
    """Lazy product that defers evaluation."""

    __slots__ = ("_factors",)

    def __init__(self, *factors):
        self._factors = factors

    @property
    def args(self) -> Tuple[Any, ...]:
        return self._factors

    def eval(self, subs: dict = None) -> Union[float, "_SymProduct"]:
        subs = subs or {}

        cached, hit = _cache_get(self, subs)
        if hit:
            return cached

        product = 1
        remaining = []

        for factor in self._factors:
            val = self._substitute_arg(factor, subs)
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

    def to_sympy(self) -> Expr:
        """转换为 SymPy Mul 表达式."""
        from sympy import Mul
        return Mul(*[self._convert_to_sympy(f) for f in self._factors])


class _SymQuotient(LazyExprMixin):
    """Lazy quotient that defers evaluation."""

    __slots__ = ("_num", "_denom")

    def __init__(self, num, denom):
        self._num = num
        self._denom = denom

    @property
    def args(self) -> Tuple[Any, Any]:
        return (self._num, self._denom)

    def eval(self, subs: dict = None) -> Union[float, "_SymQuotient"]:
        subs = subs or {}

        cached, hit = _cache_get(self, subs)
        if hit:
            return cached

        num_val = self._substitute_arg(self._num, subs)
        denom_val = self._substitute_arg(self._denom, subs)

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

    def to_sympy(self) -> Expr:
        """转换为 SymPy 除法表达式."""
        num = self._convert_to_sympy(self._num)
        denom = self._convert_to_sympy(self._denom)
        return num / denom


# 惰性表达式类型元组（用于类型检查）
_LAZY_TYPES = (SymMax, SymMin, _SymSum, _SymProduct, _SymQuotient)


def _to_float(x, subs: dict = None) -> Union[float, None]:
    """Try to convert x to Python float."""
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, _LAZY_TYPES):
        if subs:
            result = x.eval(subs)
            if isinstance(result, (int, float)):
                return float(result)
        return None
    if hasattr(x, "is_number") and x.is_number:
        try:
            return float(x)
        except (TypeError, ValueError):
            return None
    return None


def sym_max(a, b) -> Union[float, SymMax]:
    """Create a lazy max expression.
    
    Fast path for numeric values, lazy SymMax for symbolic.
    """
    a_num = _to_float(a)
    b_num = _to_float(b)

    if a_num is not None and b_num is not None:
        return max(a_num, b_num)

    return SymMax(a, b)


def sym_min(a, b) -> Union[float, SymMin]:
    """Create a lazy min expression.
    
    Fast path for numeric values, lazy SymMin for symbolic.
    """
    a_num = _to_float(a)
    b_num = _to_float(b)

    if a_num is not None and b_num is not None:
        return min(a_num, b_num)

    return SymMin(a, b)


def eval_lazy(expr, subs: dict = None) -> Union[float, Any]:
    """Evaluate a lazy expression (SymMax, SymMin, _SymSum, etc).
    
    Args:
        expr: Expression to evaluate (numeric, SymMax, SymMin, SymPy Expr)
        subs: Symbol substitutions (keys can be Symbol or str)
        
    Returns:
        Numeric result if fully evaluated, else remaining symbolic expr
    """
    subs = subs or {}

    # Normalize subs: convert string keys to Symbol
    normalized_subs = {}
    for k, v in subs.items():
        if isinstance(k, str):
            normalized_subs[get_symbol(k)] = v
        else:
            normalized_subs[k] = v

    if isinstance(expr, (int, float)):
        return float(expr)

    if isinstance(expr, _LAZY_TYPES):
        return expr.eval(normalized_subs)

    if isinstance(expr, Expr):
        result = expr.subs(normalized_subs)
        num = _to_float(result)
        return num if num is not None else result

    return expr


# ==============================================================================
# 符号分析工具函数
# ==============================================================================

def is_symbolic(expr) -> bool:
    """检查表达式是否包含符号（未完全具体化）.
    
    Args:
        expr: 任意表达式
        
    Returns:
        True if expr contains symbols, False if fully concrete
        
    Examples:
        >>> is_symbolic(42)
        False
        >>> is_symbolic(HIDDEN * 2)
        True
        >>> is_symbolic(SymMax(BATCH, 16))
        True
    """
    if isinstance(expr, (int, float)):
        return False
    if isinstance(expr, _LAZY_TYPES):
        return True  # 惰性类型始终视为符号
    if isinstance(expr, Expr):
        return len(expr.free_symbols) > 0
    if isinstance(expr, SymPick):
        return True  # SymPick 始终视为符号
    return False


def free_symbols(expr) -> Set[Symbol]:
    """获取表达式中的自由符号.
    
    Args:
        expr: 任意表达式
        
    Returns:
        Set of free symbols in the expression
        
    Examples:
        >>> free_symbols(HIDDEN * BATCH)
        {hidden, batch}
        >>> free_symbols(SymMax(TP, 1))
        {tp}
    """
    if isinstance(expr, (int, float)):
        return set()
    
    # 惰性表达式类型统一使用 args 属性
    if isinstance(expr, _LAZY_TYPES):
        result = set()
        for arg in expr.args:
            result |= free_symbols(arg)
        return result
    
    if isinstance(expr, (Expr, SymPick)):
        return expr.free_symbols
    
    return set()


def is_concrete(expr) -> bool:
    """检查表达式是否完全具体化（无符号）.
    
    is_concrete(x) 等价于 not is_symbolic(x)
    """
    return not is_symbolic(expr)


def substitute(expr, subs: dict):
    """对表达式进行符号替换.
    
    统一接口，处理各种表达式类型。
    
    Args:
        expr: 任意表达式
        subs: 替换字典 {Symbol: value} 或 {str: value}
        
    Returns:
        替换后的表达式
    """
    return eval_lazy(expr, subs)


# ==============================================================================
# 公共 API 导出
# ==============================================================================

__all__ = [
    # 预定义符号
    "BATCH", "SEQ", "HIDDEN", "FEEDFORWARD", "NUM_LAYERS", 
    "ATTN_HEADS", "HEAD_DIM", "VOCAB_SIZE",
    "TP", "PP", "DP", "CP", "EP",
    "MICRO_BATCH", "NUM_MICRO_BATCHES",
    "PEAK_FLOPS", "MEM_BANDWIDTH", "NET_BANDWIDTH",
    # 符号分组
    "MODEL_SYMBOLS", "PARALLEL_SYMBOLS", "EXEC_SYMBOLS", 
    "HARDWARE_SYMBOLS", "ALL_SYMBOLS",
    # 符号工具
    "get_symbol",
    # 表达式类型
    "SymPick", "SymMax", "SymMin", "LazyExprMixin",
    # 缓存管理
    "ExprCache", "clear_expr_cache", "get_cache_stats",
    # 工具函数
    "sym_max", "sym_min", "eval_lazy",
    "is_symbolic", "is_concrete", "free_symbols", "substitute",
]

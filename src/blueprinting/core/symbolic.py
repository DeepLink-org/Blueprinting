"""符号表达式扩展

基于 sympy.Function 的惰性求值符号表达式：
- SymPick: 条件表达式 (if-then-else)
- SymMax / SymMin: 惰性 max / min
- eval_lazy: 统一替换求值
"""

from typing import Any, Dict

from sympy import Expr, Function, S, Symbol, sympify

from .symbols import ALL_SYMBOLS


def get_symbol(name: str) -> Symbol:
    """根据名称获取预定义符号，不存在则创建新符号."""
    symbol_map = {s.name: s for s in ALL_SYMBOLS}
    return symbol_map.get(name, Symbol(name))


class SymPick(Function):
    """符号条件表达式 (if-then-else).

    条件可判定时自动折叠，否则保持惰性。subs() 后自动重新触发 eval。

    >>> tp = Symbol('tp')
    >>> SymPick(tp > 1, tp * 100, 0).subs(tp, 4)
    400
    """

    nargs = (3,)

    @classmethod
    def eval(cls, cond, true_expr, false_expr):
        if cond is S.true or cond == S.true:
            return true_expr
        if cond is S.false or cond == S.false:
            return false_expr
        try:
            return true_expr if bool(cond) else false_expr
        except TypeError:
            return None

    def _latex(self, printer):
        c = printer.doprint(self.args[0])
        t = printer.doprint(self.args[1])
        f = printer.doprint(self.args[2])
        return rf"\begin{{cases}} {t} & \text{{if }} {c} \\ {f} & \text{{otherwise}} \end{{cases}}"


class SymMax(Function):
    """惰性最大值，全部具体化时折叠，否则保持未求值。

    与 sympy.Max 的区别：不触发代数简化。支持嵌套自动展平。

    >>> a = Symbol('a')
    >>> SymMax(a, 10).subs(a, 15)
    15
    """

    @classmethod
    def eval(cls, *args):
        flat, changed = [], False
        for a in args:
            if isinstance(a, SymMax):
                flat.extend(a.args)
                changed = True
            else:
                flat.append(a)
        if changed:
            return cls(*flat)
        if all(a.is_number for a in args):
            return sympify(max(float(a) for a in args))
        return None

    def _latex(self, printer):
        return rf"\max\left({', '.join(printer.doprint(a) for a in self.args)}\right)"


class SymMin(Function):
    """惰性最小值，与 SymMax 对称。

    >>> a = Symbol('a')
    >>> SymMin(a, 10).subs(a, 5)
    5
    """

    @classmethod
    def eval(cls, *args):
        flat, changed = [], False
        for a in args:
            if isinstance(a, SymMin):
                flat.extend(a.args)
                changed = True
            else:
                flat.append(a)
        if changed:
            return cls(*flat)
        if all(a.is_number for a in args):
            return sympify(min(float(a) for a in args))
        return None

    def _latex(self, printer):
        return rf"\min\left({', '.join(printer.doprint(a) for a in self.args)}\right)"


def sym_max(a, b):
    """创建惰性 max，纯数值时直接返回 float。"""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return max(float(a), float(b))
    return SymMax(sympify(a), sympify(b))


def sym_min(a, b):
    """创建惰性 min，纯数值时直接返回 float。"""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return min(float(a), float(b))
    return SymMin(sympify(a), sympify(b))


def eval_lazy(expr, subs: dict = None):
    """对表达式执行符号替换并尝试数值化.

    Args:
        expr: 任意表达式
        subs: 替换字典，key 可以是 Symbol 或 str
    """
    if isinstance(expr, (int, float)):
        return expr

    subs = subs or {}
    normalized: Dict[Symbol, Any] = {
        (get_symbol(k) if isinstance(k, str) else k): v for k, v in subs.items()
    }

    if isinstance(expr, Expr):
        result = expr.subs(normalized)
        if result.is_number:
            try:
                return float(result)
            except (TypeError, ValueError):
                return result
        return result

    return expr

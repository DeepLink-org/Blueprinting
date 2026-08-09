"""Minimal immutable expression AST for exact workload quantities.

This AST deliberately models exact quantities, not target performance.  It is
closed, serializable, and safe to partially bind without evaluating arbitrary
Python or SymPy input.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from numbers import Real
from typing import Any, Union

from blueprinting.schema.codec import enum_type, record_type

from .axes import BindingAxis
from .errors import BindingError

Number = int | float
Scalar = Union[int, float, "Symbol", "ScalarExpr"]


@enum_type("compiler.expr_op")
class ExprOp(Enum):
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"
    CEIL_DIV = "ceil_div"
    MAX = "max"
    MIN = "min"


class _ExpressionOperators:
    def __add__(self, other: Scalar) -> Scalar:
        return expression(ExprOp.ADD, self, other)

    def __radd__(self, other: Scalar) -> Scalar:
        return expression(ExprOp.ADD, other, self)

    def __sub__(self, other: Scalar) -> Scalar:
        return expression(ExprOp.SUB, self, other)

    def __rsub__(self, other: Scalar) -> Scalar:
        return expression(ExprOp.SUB, other, self)

    def __mul__(self, other: Scalar) -> Scalar:
        return expression(ExprOp.MUL, self, other)

    def __rmul__(self, other: Scalar) -> Scalar:
        return expression(ExprOp.MUL, other, self)

    def __truediv__(self, other: Scalar) -> Scalar:
        return expression(ExprOp.DIV, self, other)

    def __rtruediv__(self, other: Scalar) -> Scalar:
        return expression(ExprOp.DIV, other, self)


@record_type("compiler.symbol")
@dataclass(frozen=True)
class Symbol(_ExpressionOperators):
    name: str
    axis: BindingAxis
    integer: bool = True
    positive: bool = False

    def __post_init__(self) -> None:
        if not self.name or not self.name.replace("_", "a").isalnum():
            raise ValueError(f"invalid symbol name: {self.name!r}")
        if not isinstance(self.axis, BindingAxis):
            raise TypeError("symbol axis must be a BindingAxis")


@record_type("compiler.scalar_expr")
@dataclass(frozen=True)
class ScalarExpr(_ExpressionOperators):
    op: ExprOp
    args: tuple[Scalar, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.op, ExprOp):
            raise TypeError("scalar expression operation must be ExprOp")
        args = tuple(_coerce(item) for item in self.args)
        object.__setattr__(self, "args", args)
        if self.op in (ExprOp.SUB, ExprOp.DIV, ExprOp.CEIL_DIV) and len(args) != 2:
            raise ValueError(f"{self.op.value} expects exactly two arguments")
        if self.op in (ExprOp.ADD, ExprOp.MUL, ExprOp.MAX, ExprOp.MIN) and len(args) < 2:
            raise ValueError(f"{self.op.value} expects at least two arguments")

    @property
    def free_symbols(self) -> tuple[Symbol, ...]:
        found = set()
        for item in self.args:
            found.update(free_symbols(item))
        return tuple(sorted(found, key=lambda symbol: (symbol.axis.value, symbol.name)))

    def subs(self, bindings: Mapping[Any, Number]) -> Scalar:
        return substitute(self, bindings)

    def evaluate(self, bindings: Mapping[Any, Number]) -> Number:
        result = substitute(self, bindings)
        if isinstance(result, (Symbol, ScalarExpr)):
            missing = ", ".join(f"{item.axis.value}.{item.name}" for item in free_symbols(result))
            raise BindingError(f"expression still has unbound symbols: {missing}")
        return result


def _coerce(value: Scalar) -> Scalar:
    if isinstance(value, bool) or not isinstance(value, (Real, Symbol, ScalarExpr)):
        raise TypeError(f"unsupported scalar expression value: {value!r}")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("scalar expressions do not permit NaN or infinity")
    return value


def expression(op: ExprOp, *args: Scalar) -> Scalar:
    """Create a lightly folded expression while preserving operand order."""

    values = tuple(_coerce(item) for item in args)
    if all(isinstance(item, Real) and not isinstance(item, bool) for item in values):
        return _evaluate_numeric(op, values)  # type: ignore[arg-type]
    if op is ExprOp.ADD:
        values = tuple(item for item in values if item != 0)
        if len(values) == 1:
            return values[0]
    elif op is ExprOp.MUL:
        if any(item == 0 for item in values):
            return 0
        values = tuple(item for item in values if item != 1)
        if len(values) == 1:
            return values[0]
    elif op is ExprOp.SUB and values[1] == 0:
        return values[0]
    return ScalarExpr(op=op, args=values)


def ceil_div(left: Scalar, right: Scalar) -> Scalar:
    return expression(ExprOp.CEIL_DIV, left, right)


def maximum(*values: Scalar) -> Scalar:
    return expression(ExprOp.MAX, *values)


def minimum(*values: Scalar) -> Scalar:
    return expression(ExprOp.MIN, *values)


def free_symbols(value: Scalar) -> tuple[Symbol, ...]:
    if isinstance(value, Symbol):
        return (value,)
    if isinstance(value, ScalarExpr):
        return value.free_symbols
    return ()


def substitute(value: Scalar, bindings: Mapping[Any, Number]) -> Scalar:
    if isinstance(value, Symbol):
        if value in bindings:
            return _validate_bound_value(value, bindings[value])
        qualified = f"{value.axis.value}.{value.name}"
        if qualified in bindings:
            return _validate_bound_value(value, bindings[qualified])
        if value.name in bindings:
            return _validate_bound_value(value, bindings[value.name])
        return value
    if isinstance(value, ScalarExpr):
        return expression(value.op, *(substitute(item, bindings) for item in value.args))
    return value


def _validate_bound_value(symbol: Symbol, value: Number) -> Number:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise BindingError(f"binding for {symbol.name!r} must be numeric")
    if symbol.integer and int(value) != value:
        raise BindingError(f"binding for integer symbol {symbol.name!r} must be integral")
    if symbol.positive and value <= 0:
        raise BindingError(f"binding for positive symbol {symbol.name!r} must be greater than zero")
    return int(value) if symbol.integer else value


def _evaluate_numeric(op: ExprOp, values: tuple[Number, ...]) -> Number:
    if op is ExprOp.ADD:
        return sum(values)
    if op is ExprOp.SUB:
        return values[0] - values[1]
    if op is ExprOp.MUL:
        result: Number = 1
        for item in values:
            result *= item
        return result
    if op is ExprOp.DIV:
        if values[1] == 0:
            raise ZeroDivisionError("division by zero in scalar expression")
        return values[0] / values[1]
    if op is ExprOp.CEIL_DIV:
        if values[1] == 0:
            raise ZeroDivisionError("division by zero in scalar expression")
        return math.ceil(values[0] / values[1])
    if op is ExprOp.MAX:
        return max(values)
    if op is ExprOp.MIN:
        return min(values)
    raise AssertionError(f"unhandled expression operation: {op}")

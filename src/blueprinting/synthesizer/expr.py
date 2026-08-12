"""Immutable algebraic expressions for exact workload quantities.

Each operation is a distinct constructor, so invalid arity is not representable.
The small ``ExprOp`` enum remains only as a convenient smart-constructor input;
it is not part of the canonical expression representation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from enum import Enum
from numbers import Real
from typing import Annotated, Any, ClassVar, TypeAlias, cast

from typing_extensions import assert_never

from blueprinting.schema.authoring import (
    FiniteFloat,
    SymbolName,
    ValueConstraint,
    VariantSpec,
    adt,
    is_adt_variant,
    record,
    seal_adt,
    variant,
)

from .axes import BindingAxis
from .errors import BindingError

Number = int | float


class ExprOp(Enum):
    """Surface syntax accepted by :func:`expression`; not a wire discriminator."""

    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"
    CEIL_DIV = "ceil_div"
    MAX = "max"
    MIN = "min"


class _ExpressionOperators:
    def __add__(self, other: Scalar) -> Scalar:
        return expression(ExprOp.ADD, cast(Scalar, self), other)

    def __radd__(self, other: Scalar) -> Scalar:
        return expression(ExprOp.ADD, other, cast(Scalar, self))

    def __sub__(self, other: Scalar) -> Scalar:
        return expression(ExprOp.SUB, cast(Scalar, self), other)

    def __rsub__(self, other: Scalar) -> Scalar:
        return expression(ExprOp.SUB, other, cast(Scalar, self))

    def __mul__(self, other: Scalar) -> Scalar:
        return expression(ExprOp.MUL, cast(Scalar, self), other)

    def __rmul__(self, other: Scalar) -> Scalar:
        return expression(ExprOp.MUL, other, cast(Scalar, self))

    def __truediv__(self, other: Scalar) -> Scalar:
        return expression(ExprOp.DIV, cast(Scalar, self), other)

    def __rtruediv__(self, other: Scalar) -> Scalar:
        return expression(ExprOp.DIV, other, cast(Scalar, self))


@record("blueprinting.expression.symbol")
class Symbol(_ExpressionOperators):
    name: SymbolName
    axis: BindingAxis
    integer: bool = True
    positive: bool = False


@adt(wire="blueprinting.expression.scalar")
class ScalarExpr(_ExpressionOperators):
    """Closed family of exact scalar expression constructors."""

    __variant_spec__: ClassVar[VariantSpec]

    @property
    def free_symbols(self) -> tuple[Symbol, ...]:
        found: set[Symbol] = set()
        for item in operands(cast(ScalarExprVariant, self)):
            found.update(free_symbols(item))
        return tuple(sorted(found, key=lambda symbol: (symbol.axis.value, symbol.name)))

    def subs(self, bindings: Mapping[Any, Number]) -> Scalar:
        return substitute(cast(ScalarExprVariant, self), bindings)

    def evaluate(self, bindings: Mapping[Any, Number]) -> Number:
        result = substitute(cast(ScalarExprVariant, self), bindings)
        if isinstance(result, Symbol) or is_adt_variant(result, ScalarExpr):
            missing = ", ".join(f"{item.axis.value}.{item.name}" for item in free_symbols(cast(Scalar, result)))
            raise BindingError(f"expression still has unbound symbols: {missing}")
        return cast(Number, result)


@variant("add")
class Add(ScalarExpr):
    terms: ScalarOperands


@variant("subtract")
class Subtract(ScalarExpr):
    left: Scalar
    right: Scalar


@variant("multiply")
class Multiply(ScalarExpr):
    factors: ScalarOperands


@variant("divide")
class Divide(ScalarExpr):
    numerator: Scalar
    denominator: Scalar


@variant("ceil-divide")
class CeilDivide(ScalarExpr):
    numerator: Scalar
    denominator: Scalar


@variant("maximum")
class Maximum(ScalarExpr):
    values: ScalarOperands


@variant("minimum")
class Minimum(ScalarExpr):
    values: ScalarOperands


ScalarExprVariant: TypeAlias = Add | Subtract | Multiply | Divide | CeilDivide | Maximum | Minimum
seal_adt(ScalarExpr, ScalarExprVariant)
Scalar: TypeAlias = int | FiniteFloat | Symbol | ScalarExprVariant
ScalarOperands: TypeAlias = Annotated[
    tuple[Scalar, ...],
    ValueConstraint.AT_LEAST_TWO_ITEMS,
]


def _coerce(value: Scalar) -> Scalar:
    if isinstance(value, bool) or not (isinstance(value, (Real, Symbol)) or is_adt_variant(value, ScalarExpr)):
        raise TypeError(f"unsupported scalar expression value: {value!r}")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("scalar expressions do not permit NaN or infinity")
    return cast(Scalar, value)


def operands(value: ScalarExprVariant) -> tuple[Scalar, ...]:
    """Return constructor operands for generic visitors without exposing arity tags."""

    match value:
        case Add(terms=terms):
            return terms
        case Subtract(left=left, right=right):
            return (left, right)
        case Multiply(factors=factors):
            return factors
        case Divide(numerator=left, denominator=right) | CeilDivide(numerator=left, denominator=right):
            return (left, right)
        case Maximum(values=values) | Minimum(values=values):
            return values
    assert_never(value)


def expression(op: ExprOp, *args: Scalar) -> Scalar:
    """Create a lightly folded algebraic expression while preserving operand order."""

    values = tuple(_coerce(item) for item in args)
    if all(isinstance(item, Real) and not isinstance(item, bool) for item in values):
        return _evaluate_numeric(op, cast(tuple[Number, ...], values))
    match op:
        case ExprOp.ADD:
            values = tuple(item for item in values if item != 0)
            if len(values) == 1:
                return values[0]
            return Add(values)
        case ExprOp.SUB:
            if len(values) != 2:
                raise ValueError("subtract expects exactly two arguments")
            if values[1] == 0:
                return values[0]
            return Subtract(values[0], values[1])
        case ExprOp.MUL:
            if any(item == 0 for item in values):
                return 0
            values = tuple(item for item in values if item != 1)
            if len(values) == 1:
                return values[0]
            return Multiply(values)
        case ExprOp.DIV:
            if len(values) != 2:
                raise ValueError("divide expects exactly two arguments")
            return Divide(values[0], values[1])
        case ExprOp.CEIL_DIV:
            if len(values) != 2:
                raise ValueError("ceil-divide expects exactly two arguments")
            return CeilDivide(values[0], values[1])
        case ExprOp.MAX:
            return Maximum(values)
        case ExprOp.MIN:
            return Minimum(values)
    assert_never(op)


def ceil_div(left: Scalar, right: Scalar) -> Scalar:
    return expression(ExprOp.CEIL_DIV, left, right)


def maximum(*values: Scalar) -> Scalar:
    return expression(ExprOp.MAX, *values)


def minimum(*values: Scalar) -> Scalar:
    return expression(ExprOp.MIN, *values)


def free_symbols(value: Scalar) -> tuple[Symbol, ...]:
    if isinstance(value, Symbol):
        return (value,)
    if is_adt_variant(value, ScalarExpr):
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
    match value:
        case Add(terms=terms):
            return expression(ExprOp.ADD, *(substitute(item, bindings) for item in terms))
        case Subtract(left=left, right=right):
            return expression(ExprOp.SUB, substitute(left, bindings), substitute(right, bindings))
        case Multiply(factors=factors):
            return expression(ExprOp.MUL, *(substitute(item, bindings) for item in factors))
        case Divide(numerator=left, denominator=right):
            return expression(ExprOp.DIV, substitute(left, bindings), substitute(right, bindings))
        case CeilDivide(numerator=left, denominator=right):
            return expression(ExprOp.CEIL_DIV, substitute(left, bindings), substitute(right, bindings))
        case Maximum(values=values):
            return expression(ExprOp.MAX, *(substitute(item, bindings) for item in values))
        case Minimum(values=values):
            return expression(ExprOp.MIN, *(substitute(item, bindings) for item in values))
        case int() | float():
            return value
    assert_never(value)


def _validate_bound_value(symbol: Symbol, value: Number) -> Number:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise BindingError(f"binding for {symbol.name!r} must be numeric")
    if symbol.integer and int(value) != value:
        raise BindingError(f"binding for integer symbol {symbol.name!r} must be integral")
    if symbol.positive and value <= 0:
        raise BindingError(f"binding for positive symbol {symbol.name!r} must be greater than zero")
    return int(value) if symbol.integer else value


def _evaluate_numeric(op: ExprOp, values: tuple[Number, ...]) -> Number:
    match op:
        case ExprOp.ADD:
            if len(values) < 2:
                raise ValueError("add expects at least two arguments")
            return sum(values)
        case ExprOp.SUB:
            if len(values) != 2:
                raise ValueError("subtract expects exactly two arguments")
            return values[0] - values[1]
        case ExprOp.MUL:
            if len(values) < 2:
                raise ValueError("multiply expects at least two arguments")
            result: Number = 1
            for item in values:
                result *= item
            return result
        case ExprOp.DIV:
            if len(values) != 2:
                raise ValueError("divide expects exactly two arguments")
            if values[1] == 0:
                raise ZeroDivisionError("division by zero in scalar expression")
            return values[0] / values[1]
        case ExprOp.CEIL_DIV:
            if len(values) != 2:
                raise ValueError("ceil-divide expects exactly two arguments")
            if values[1] == 0:
                raise ZeroDivisionError("division by zero in scalar expression")
            return math.ceil(values[0] / values[1])
        case ExprOp.MAX:
            if len(values) < 2:
                raise ValueError("maximum expects at least two arguments")
            return max(values)
        case ExprOp.MIN:
            if len(values) < 2:
                raise ValueError("minimum expects at least two arguments")
            return min(values)
    assert_never(op)


__all__ = [
    "Add",
    "CeilDivide",
    "Divide",
    "ExprOp",
    "Maximum",
    "Minimum",
    "Multiply",
    "Scalar",
    "ScalarExpr",
    "ScalarExprVariant",
    "Subtract",
    "Symbol",
    "ceil_div",
    "expression",
    "free_symbols",
    "maximum",
    "minimum",
    "operands",
    "substitute",
]

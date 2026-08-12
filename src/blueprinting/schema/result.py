"""Small algebraic result type with ordered diagnostic accumulation."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Generic, TypeAlias, TypeVar, cast

from typing_extensions import Never

from .diagnostics import EMPTY_DIAGNOSTICS, DiagnosticError, DiagnosticSet

T = TypeVar("T")
U = TypeVar("U")
E = TypeVar("E")
F = TypeVar("F")
T_co = TypeVar("T_co", covariant=True)
E_co = TypeVar("E_co", covariant=True)


class Result(Generic[T_co, E_co]):
    """Success or expected failure without hidden exception control flow."""

    @property
    def is_ok(self) -> bool:
        return isinstance(self, Ok)

    @property
    def is_err(self) -> bool:
        return isinstance(self, Err)

    def map(self: Result[T, E], function: Callable[[T], U]) -> Result[U, E]:
        if isinstance(self, Ok):
            success = cast(Ok[T], self)
            return Ok(function(success.value), success.diagnostics)
        return cast(Err[E], self)

    def map_error(self: Result[T, E], function: Callable[[E], F]) -> Result[T, F]:
        if isinstance(self, Err):
            failure = cast(Err[E], self)
            return Err(function(failure.error))
        return cast(Ok[T], self)

    def and_then(self: Result[T, E], function: Callable[[T], Result[U, E]]) -> Result[U, E]:
        if isinstance(self, Err):
            return cast(Err[E], self)
        success = cast(Ok[T], self)
        following = function(success.value)
        if isinstance(following, Ok):
            next_success = cast(Ok[U], following)
            return Ok(next_success.value, success.diagnostics.extend(next_success.diagnostics))
        failure = cast(Err[E], following)
        if isinstance(failure.error, DiagnosticSet) and success.diagnostics.diagnostics:
            return Err(cast(E, success.diagnostics.extend(failure.error)))
        return failure

    def or_raise(
        self: Result[T, E],
        factory: Callable[[E], BaseException] | None = None,
    ) -> T:
        """Unwrap only at an explicit exception boundary."""

        if isinstance(self, Ok):
            return cast(Ok[T], self).value
        failure = cast(Err[E], self)
        if factory is not None:
            raise factory(failure.error)
        if isinstance(failure.error, DiagnosticSet):
            raise DiagnosticError(failure.error)
        raise RuntimeError(f"unhandled Result error: {failure.error!r}")


@dataclass(frozen=True)
class Ok(Result[T_co, Never], Generic[T_co]):
    value: T_co
    diagnostics: DiagnosticSet = field(default_factory=DiagnosticSet)

    def __post_init__(self) -> None:
        if not isinstance(self.diagnostics, DiagnosticSet):
            raise TypeError("Ok diagnostics must be a DiagnosticSet")
        if self.diagnostics.errors:
            raise ValueError("Ok diagnostics may contain warnings but not errors")


@dataclass(frozen=True)
class Err(Result[Never, E_co], Generic[E_co]):
    error: E_co

    def __post_init__(self) -> None:
        if isinstance(self.error, DiagnosticSet) and not self.error.errors:
            raise ValueError("Err diagnostics must contain at least one error")


Checked: TypeAlias = Result[T, DiagnosticSet]


def collect_results(results: Iterable[Checked[T]]) -> Checked[tuple[T, ...]]:
    """Accumulate independent Checked values without losing diagnostic order."""

    values: list[T] = []
    warnings = EMPTY_DIAGNOSTICS
    failures = EMPTY_DIAGNOSTICS
    for result in results:
        if isinstance(result, Ok):
            success = cast(Ok[T], result)
            values.append(success.value)
            warnings = warnings.extend(success.diagnostics)
        else:
            failures = failures.extend(cast(Err[DiagnosticSet], result).error)
    if failures.errors:
        return Err(warnings.extend(failures))
    return Ok(tuple(values), warnings)


def checked(value: T, diagnostics: DiagnosticSet = EMPTY_DIAGNOSTICS) -> Checked[T]:
    """Create Ok or Err according to the supplied immutable diagnostics."""

    return Err(diagnostics) if diagnostics.errors else Ok(value, diagnostics)


__all__ = ["Checked", "Err", "Ok", "Result", "checked", "collect_results"]

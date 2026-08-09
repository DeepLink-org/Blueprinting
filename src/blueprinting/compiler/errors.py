"""Error and diagnostic types for the canonical compiler.

The compiler distinguishes user-facing contract failures from implementation
errors.  Verification collects diagnostics first and raises them as one error,
which is substantially more useful than failing on the first malformed edge.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum


class CompilerError(Exception):
    """Base class for canonical compiler failures."""


class SerializationError(CompilerError):
    """Raised when canonical serialization or deserialization fails."""


class InvalidIdError(CompilerError, ValueError):
    """Raised when a stable compiler identifier is malformed."""


class BindingError(CompilerError, ValueError):
    """Raised when a typed binding is invalid or incomplete."""


class MissingBindingError(BindingError):
    """Raised when a lowering pass is missing a required binding axis."""


class MissingAnalysisError(CompilerError, LookupError):
    """Raised when a pass requires an analysis absent from the current snapshot."""


class PassContractError(CompilerError):
    """Raised when a pass violates its declared contract."""


class PassExecutionError(CompilerError):
    """Wraps an unexpected exception raised by a compiler pass."""

    def __init__(self, pass_name: str, cause: BaseException):
        self.pass_name = pass_name
        self.cause = cause
        super().__init__(f"pass {pass_name!r} failed: {cause}")


class Severity(Enum):
    """Diagnostic severity."""

    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class Diagnostic:
    """One stable, machine-readable verification diagnostic."""

    code: str
    message: str
    path: tuple[str, ...] = ()
    severity: Severity = Severity.ERROR
    hint: str | None = None

    def render(self) -> str:
        location = ".".join(self.path) if self.path else "<root>"
        suffix = f" Hint: {self.hint}" if self.hint else ""
        return f"[{self.code}] {location}: {self.message}{suffix}"


@dataclass(frozen=True)
class VerificationReport:
    """Immutable result of verifying one IR snapshot."""

    diagnostics: tuple[Diagnostic, ...] = ()

    @property
    def errors(self) -> tuple[Diagnostic, ...]:
        return tuple(item for item in self.diagnostics if item.severity is Severity.ERROR)

    @property
    def warnings(self) -> tuple[Diagnostic, ...]:
        return tuple(item for item in self.diagnostics if item.severity is Severity.WARNING)

    @property
    def ok(self) -> bool:
        return not self.errors

    def extend(self, other: VerificationReport) -> VerificationReport:
        return VerificationReport(self.diagnostics + other.diagnostics)

    def require_ok(self, subject: str = "IR") -> None:
        if not self.ok:
            raise IRVerificationError(subject, self.errors)


class IRVerificationError(CompilerError, ValueError):
    """Raised when a canonical IR snapshot violates its contract."""

    def __init__(self, subject: str, diagnostics: Iterable[Diagnostic]):
        self.subject = subject
        self.diagnostics = tuple(diagnostics)
        rendered = "\n".join(f"  - {item.render()}" for item in self.diagnostics)
        super().__init__(f"{subject} verification failed:\n{rendered}")


class DiagnosticBag:
    """Mutable diagnostic accumulator scoped to one verifier invocation."""

    __slots__ = ("_items",)

    def __init__(self) -> None:
        self._items = []

    def error(
        self,
        code: str,
        message: str,
        *path: str,
        hint: str | None = None,
    ) -> None:
        self._items.append(Diagnostic(code=code, message=message, path=tuple(path), hint=hint))

    def warning(
        self,
        code: str,
        message: str,
        *path: str,
        hint: str | None = None,
    ) -> None:
        self._items.append(
            Diagnostic(
                code=code,
                message=message,
                path=tuple(path),
                severity=Severity.WARNING,
                hint=hint,
            )
        )

    def report(self) -> VerificationReport:
        return VerificationReport(tuple(self._items))

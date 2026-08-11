"""Domain-free diagnostics shared by schema, derivation, and applications."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import Enum


class Severity(Enum):
    """Stable diagnostic severity used across Blueprinting contracts."""

    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True, order=True)
class Diagnostic:
    """One stable, machine-readable contract diagnostic."""

    code: str
    message: str
    path: tuple[str, ...] = ()
    severity: Severity = Severity.ERROR
    hint: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not self.code:
            raise ValueError("diagnostic code must be a non-empty string")
        if not isinstance(self.message, str) or not self.message:
            raise ValueError("diagnostic message must be a non-empty string")
        if any(not isinstance(item, str) or not item for item in self.path):
            raise ValueError("diagnostic paths must contain non-empty strings")
        if not isinstance(self.severity, Severity):
            raise TypeError("diagnostic severity must be Severity")
        if self.hint is not None and (not isinstance(self.hint, str) or not self.hint):
            raise ValueError("diagnostic hint must be non-empty when present")

    def prefixed(self, *path: str) -> Diagnostic:
        """Return the same diagnostic below an additional stable path."""

        return Diagnostic(self.code, self.message, tuple(path) + self.path, self.severity, self.hint)

    def render(self) -> str:
        location = ".".join(self.path) if self.path else "<root>"
        suffix = f" Hint: {self.hint}" if self.hint else ""
        return f"[{self.code}] {location}: {self.message}{suffix}"


@dataclass(frozen=True)
class DiagnosticSet:
    """Immutable, ordered diagnostics produced by one contract boundary."""

    diagnostics: tuple[Diagnostic, ...] = ()

    def __post_init__(self) -> None:
        values = tuple(self.diagnostics)
        if any(not isinstance(item, Diagnostic) for item in values):
            raise TypeError("diagnostic sets may contain only Diagnostic values")
        object.__setattr__(self, "diagnostics", values)

    def __iter__(self) -> Iterator[Diagnostic]:
        return iter(self.diagnostics)

    def __len__(self) -> int:
        return len(self.diagnostics)

    @property
    def errors(self) -> tuple[Diagnostic, ...]:
        return tuple(item for item in self.diagnostics if item.severity is Severity.ERROR)

    @property
    def warnings(self) -> tuple[Diagnostic, ...]:
        return tuple(item for item in self.diagnostics if item.severity is Severity.WARNING)

    @property
    def ok(self) -> bool:
        return not self.errors

    def extend(self, other: DiagnosticSet) -> DiagnosticSet:
        if not isinstance(other, DiagnosticSet):
            raise TypeError("diagnostics may only be extended with another DiagnosticSet")
        return DiagnosticSet(self.diagnostics + other.diagnostics)

    def prefixed(self, *path: str) -> DiagnosticSet:
        return DiagnosticSet(tuple(item.prefixed(*path) for item in self.diagnostics))

    @classmethod
    def of(cls, *diagnostics: Diagnostic) -> DiagnosticSet:
        return cls(tuple(diagnostics))


EMPTY_DIAGNOSTICS = DiagnosticSet()


class DiagnosticError(ValueError):
    """Explicit exception adapter for Checked values at application boundaries."""

    def __init__(self, diagnostics: DiagnosticSet, subject: str = "Blueprinting contract") -> None:
        if not isinstance(diagnostics, DiagnosticSet):
            raise TypeError("DiagnosticError requires a DiagnosticSet")
        self.subject = subject
        self.diagnostics = diagnostics
        rendered = "\n".join(f"  - {item.render()}" for item in diagnostics.errors or diagnostics.diagnostics)
        super().__init__(f"{subject} failed:\n{rendered}")


class DiagnosticBag:
    """Local mutable builder that publishes only immutable diagnostics."""

    __slots__ = ("_items",)

    def __init__(self) -> None:
        self._items: list[Diagnostic] = []

    def error(self, code: str, message: str, *path: str, hint: str | None = None) -> None:
        self._items.append(Diagnostic(code, message, tuple(path), Severity.ERROR, hint))

    def warning(self, code: str, message: str, *path: str, hint: str | None = None) -> None:
        self._items.append(Diagnostic(code, message, tuple(path), Severity.WARNING, hint))

    def add(self, diagnostic: Diagnostic) -> None:
        if not isinstance(diagnostic, Diagnostic):
            raise TypeError("diagnostic bag entries must be Diagnostic values")
        self._items.append(diagnostic)

    def extend(self, diagnostics: Iterable[Diagnostic]) -> None:
        for diagnostic in diagnostics:
            self.add(diagnostic)

    def report(self) -> DiagnosticSet:
        return DiagnosticSet(tuple(self._items))


__all__ = [
    "Diagnostic",
    "DiagnosticBag",
    "DiagnosticError",
    "DiagnosticSet",
    "EMPTY_DIAGNOSTICS",
    "Severity",
]

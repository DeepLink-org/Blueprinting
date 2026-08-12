"""Error and diagnostic types for formal synthesis.

The synthesizer distinguishes user-facing contract failures from implementation
errors.  Verification collects diagnostics first and raises them as one error,
which is substantially more useful than failing on the first malformed edge.
"""

from __future__ import annotations

from collections.abc import Iterable

from blueprinting.schema.diagnostics import (
    Diagnostic as Diagnostic,
)
from blueprinting.schema.diagnostics import (
    DiagnosticBag as _DiagnosticBag,
)
from blueprinting.schema.diagnostics import (
    DiagnosticSet,
)
from blueprinting.schema.diagnostics import (
    Severity as Severity,
)


class SynthesisError(Exception):
    """Base class for formal-synthesis failures."""


class InvalidIdError(SynthesisError, ValueError):
    """Raised when a stable synthesis identifier is malformed."""


class BindingError(SynthesisError, ValueError):
    """Raised when a typed binding is invalid or incomplete."""


class MissingBindingError(BindingError):
    """Raised when a lowering pass is missing a required binding axis."""


class MissingAnalysisError(SynthesisError, LookupError):
    """Raised when a pass requires an analysis absent from the current snapshot."""


class PassContractError(SynthesisError):
    """Raised when a pass violates its declared contract."""


class PassExecutionError(SynthesisError):
    """Wraps an unexpected exception raised by a derivation pass."""

    def __init__(self, pass_name: str, cause: BaseException):
        self.pass_name = pass_name
        self.cause = cause
        super().__init__(f"pass {pass_name!r} failed: {cause}")


class VerificationReport(DiagnosticSet):
    """Backward-compatible name for the domain-free immutable diagnostics."""

    def require_ok(self, subject: str = "IR") -> None:
        if not self.ok:
            raise IRVerificationError(subject, self.errors)


class IRVerificationError(SynthesisError, ValueError):
    """Raised when a canonical IR snapshot violates its contract."""

    def __init__(self, subject: str, diagnostics: Iterable[Diagnostic]):
        self.subject = subject
        self.diagnostics = tuple(diagnostics)
        rendered = "\n".join(f"  - {item.render()}" for item in self.diagnostics)
        super().__init__(f"{subject} verification failed:\n{rendered}")


class DiagnosticBag(_DiagnosticBag):
    """Compatibility builder that publishes VerificationReport."""

    def report(self) -> VerificationReport:
        return VerificationReport(super().report().diagnostics)

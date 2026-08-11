"""Canonical immutable-value and serialization primitives.

This package is deliberately domain-free. Workload, system, synthesis, and
analysis contracts may depend on it; it must not import any of those packages.
"""

from blueprinting.schema.codec import (
    canonical_decode,
    canonical_dump_raw,
    canonical_dumps,
    canonical_loads,
    canonical_parse,
    content_digest,
    raw_content_digest,
)
from blueprinting.schema.frozen import EMPTY_MAP, FrozenDict, freeze, thaw

from .contracts import TypeUniverse, compile_type_universe
from .diagnostics import (
    EMPTY_DIAGNOSTICS,
    Diagnostic,
    DiagnosticBag,
    DiagnosticError,
    DiagnosticSet,
    Severity,
)
from .errors import SchemaError, SerializationError
from .result import Checked, Err, Ok, Result, checked, collect_results

__all__ = [
    "EMPTY_MAP",
    "FrozenDict",
    "Checked",
    "Diagnostic",
    "DiagnosticBag",
    "DiagnosticError",
    "DiagnosticSet",
    "EMPTY_DIAGNOSTICS",
    "Err",
    "Ok",
    "Result",
    "SchemaError",
    "SerializationError",
    "Severity",
    "TypeUniverse",
    "canonical_dumps",
    "canonical_decode",
    "canonical_dump_raw",
    "canonical_loads",
    "canonical_parse",
    "content_digest",
    "checked",
    "collect_results",
    "compile_type_universe",
    "freeze",
    "raw_content_digest",
    "thaw",
]

"""Canonical immutable-value and serialization primitives.

This package is deliberately domain-free. Workload, system, synthesis, and
analysis contracts may depend on it; it must not import any of those packages.
"""

from blueprinting.schema.codec import canonical_dumps, canonical_loads, content_digest, enum_type, record_type
from blueprinting.schema.frozen import EMPTY_MAP, FrozenDict, freeze, thaw

from .errors import SchemaError, SerializationError

__all__ = [
    "EMPTY_MAP",
    "FrozenDict",
    "SchemaError",
    "SerializationError",
    "canonical_dumps",
    "canonical_loads",
    "content_digest",
    "enum_type",
    "freeze",
    "record_type",
    "thaw",
]

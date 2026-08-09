"""Failures raised by canonical schema primitives."""


class SchemaError(Exception):
    """Base class for immutable-schema and serialization failures."""


class SerializationError(SchemaError):
    """Raised when canonical serialization or deserialization fails."""

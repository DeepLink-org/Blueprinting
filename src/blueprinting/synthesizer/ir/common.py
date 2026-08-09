"""Shared contracts for the canonical formal representations.

The classes in this module intentionally keep serialization, schema identity,
diagnostics, and content addressing out of individual dialect implementations.
An IR object is an immutable value; ``to_json`` wraps that value in a checked
snapshot envelope instead of asking every dialect to reinvent persistence.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, ClassVar, TypeVar

from blueprinting.schema.codec import canonical_dumps, canonical_loads, content_digest, enum_type, record_type
from blueprinting.schema.errors import SerializationError
from blueprinting.schema.frozen import FrozenDict, freeze

from ..errors import DiagnosticBag, VerificationReport
from ..expr import Scalar, ScalarExpr, Symbol
from ..ids import StableId

_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")
_DIGEST_RE = re.compile(r"^[0-9a-f]{40}$")
IR = TypeVar("IR", bound="CanonicalIRMixin")
Entity = TypeVar("Entity")


def is_content_digest(value: str) -> bool:
    return isinstance(value, str) and _DIGEST_RE.fullmatch(value) is not None


def frozen_map(value: Any) -> FrozenDict:
    """Copy an extension mapping into the synthesizer's immutable value domain."""

    result = freeze(value)
    if not isinstance(result, FrozenDict):
        raise TypeError("expected a mapping")
    return result


def require_instance(value: Any, expected: type[Any], field_name: str) -> None:
    if not isinstance(value, expected):
        raise TypeError(f"{field_name} must be {expected.__name__}")


def typed_tuple(value: Iterable[Any], expected: type[Any], field_name: str) -> tuple[Any, ...]:
    result = tuple(value)
    if any(not isinstance(item, expected) for item in result):
        raise TypeError(f"{field_name} must contain only {expected.__name__} values")
    return result


@record_type("compiler.schema_version")
@dataclass(frozen=True, order=True)
class SchemaVersion:
    """Semantic version of one serialized IR schema."""

    major: int
    minor: int = 0
    patch: int = 0

    def __post_init__(self) -> None:
        for name in ("major", "minor", "patch"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"schema {name} must be a non-negative integer")

    @classmethod
    def parse(cls, value: str) -> SchemaVersion:
        parts = value.split(".")
        if len(parts) != 3 or any(not part.isdigit() for part in parts):
            raise ValueError(f"invalid schema version: {value!r}")
        return cls(*(int(part) for part in parts))

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


@record_type("compiler.ir_header")
@dataclass(frozen=True)
class IRHeader:
    """Version and provenance header embedded in every canonical IR."""

    schema_name: str
    schema_version: SchemaVersion
    producer_version: str = "0.1.0"
    feature_set: frozenset[str] = frozenset()
    parent_digests: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.feature_set, (str, bytes)):
            raise TypeError("feature_set must be an iterable of feature names")
        object.__setattr__(self, "feature_set", frozenset(self.feature_set))
        object.__setattr__(self, "parent_digests", tuple(self.parent_digests))
        require_instance(self.schema_version, SchemaVersion, "schema_version")
        if not isinstance(self.schema_name, str):
            raise TypeError("schema_name must be a string")
        if _NAME_RE.fullmatch(self.schema_name) is None:
            raise ValueError(f"invalid schema name: {self.schema_name!r}")
        if not isinstance(self.producer_version, str) or not self.producer_version:
            raise ValueError("producer_version must not be empty")
        if any(not isinstance(feature, str) or _NAME_RE.fullmatch(feature) is None for feature in self.feature_set):
            raise ValueError("IR feature names must be stable identifiers")
        if any(not is_content_digest(item) for item in self.parent_digests):
            raise ValueError("parent_digests must contain canonical 160-bit hex digests")

    def with_parents(self, *digests: str) -> IRHeader:
        return replace(self, parent_digests=tuple(digests))


def make_header(
    schema_name: str,
    schema_version: SchemaVersion,
    *,
    parent_digests: Iterable[str] = (),
    features: Iterable[str] = (),
    producer_version: str = "0.1.0",
) -> IRHeader:
    return IRHeader(
        schema_name=schema_name,
        schema_version=schema_version,
        producer_version=producer_version,
        feature_set=frozenset(features),
        parent_digests=tuple(parent_digests),
    )


@record_type("compiler.ir_snapshot")
@dataclass(frozen=True)
class IRSnapshot:
    """Self-checking persistence envelope for a canonical IR value."""

    schema_name: str
    schema_version: SchemaVersion
    producer_version: str
    feature_set: frozenset[str]
    content_digest: str
    payload: Any

    def __post_init__(self) -> None:
        if isinstance(self.feature_set, (str, bytes)):
            raise TypeError("snapshot feature_set must be an iterable of feature names")
        object.__setattr__(self, "feature_set", frozenset(self.feature_set))
        require_instance(self.schema_version, SchemaVersion, "snapshot schema_version")
        if not isinstance(self.schema_name, str) or not isinstance(self.producer_version, str):
            raise TypeError("snapshot schema and producer names must be strings")
        if not is_content_digest(self.content_digest):
            raise ValueError("snapshot content_digest must be a canonical 160-bit hex digest")


@enum_type("compiler.effect_kind")
class EffectKind(Enum):
    READ = "read"
    WRITE = "write"
    STATE = "state"
    RANDOM = "random"
    IO = "io"


@record_type("compiler.effect")
@dataclass(frozen=True)
class Effect:
    kind: EffectKind
    resource: str

    def __post_init__(self) -> None:
        require_instance(self.kind, EffectKind, "effect kind")
        if not self.resource:
            raise ValueError("effect resource must not be empty")


@record_type("compiler.operation_name")
@dataclass(frozen=True, order=True)
class OperationName:
    """Structured operation identity; dialect is never inferred from a string."""

    dialect: str
    name: str

    def __post_init__(self) -> None:
        if not isinstance(self.dialect, str) or not isinstance(self.name, str):
            raise TypeError("operation dialect and name must be strings")
        if _NAME_RE.fullmatch(self.dialect) is None or _NAME_RE.fullmatch(self.name) is None:
            raise ValueError(f"invalid operation name: {self.dialect}.{self.name}")

    @classmethod
    def parse(cls, value: str) -> OperationName:
        dialect, separator, name = value.partition(".")
        if not separator:
            raise ValueError("operation names must use '<dialect>.<name>' form")
        return cls(dialect=dialect, name=name)

    def __str__(self) -> str:
        return f"{self.dialect}.{self.name}"


@record_type("compiler.tensor_type")
@dataclass(frozen=True)
class TensorType:
    """Target-neutral logical tensor type."""

    shape: tuple[Scalar, ...]
    dtype: str
    layout: tuple[int, ...] | None = None
    attributes: FrozenDict = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(self.shape))
        if self.layout is not None:
            object.__setattr__(self, "layout", tuple(self.layout))
        object.__setattr__(self, "attributes", frozen_map(self.attributes))
        if not isinstance(self.dtype, str):
            raise TypeError("tensor dtype must be a string")
        if _NAME_RE.fullmatch(self.dtype) is None:
            raise ValueError(f"invalid dtype name: {self.dtype!r}")
        for dimension in self.shape:
            if isinstance(dimension, bool) or not isinstance(dimension, (int, float, Symbol, ScalarExpr)):
                raise TypeError(f"invalid tensor dimension: {dimension!r}")
            if isinstance(dimension, float) and not math.isfinite(dimension):
                raise ValueError("concrete tensor dimensions must be finite")
            if isinstance(dimension, (int, float)) and dimension <= 0:
                raise ValueError("concrete tensor dimensions must be greater than zero")
        if self.layout is not None and tuple(sorted(self.layout)) != tuple(range(len(self.shape))):
            raise ValueError("tensor layout must be a permutation of shape dimensions")

    @property
    def rank(self) -> int:
        return len(self.shape)


class CanonicalIRMixin:
    """Behavior shared by immutable canonical IR roots."""

    SCHEMA_NAME: ClassVar[str]
    SCHEMA_VERSION: ClassVar[SchemaVersion]
    header: IRHeader

    @property
    def digest(self) -> str:
        return content_digest(self, f"ir:{self.SCHEMA_NAME}")

    def to_json(self) -> str:
        """Serialize through a self-describing, digest-checked envelope."""

        self.require_valid()
        snapshot = IRSnapshot(
            schema_name=self.header.schema_name,
            schema_version=self.header.schema_version,
            producer_version=self.header.producer_version,
            feature_set=self.header.feature_set,
            content_digest=self.digest,
            payload=self,
        )
        return canonical_dumps(snapshot)

    @classmethod
    def from_json(cls: type[IR], payload: str) -> IR:
        decoded = canonical_loads(payload)
        if not isinstance(decoded, IRSnapshot):
            raise SerializationError("canonical IR JSON must contain an IR snapshot envelope")
        if decoded.schema_name != cls.SCHEMA_NAME or decoded.schema_version != cls.SCHEMA_VERSION:
            raise SerializationError(
                f"snapshot schema {decoded.schema_name}@{decoded.schema_version} is not "
                f"{cls.SCHEMA_NAME}@{cls.SCHEMA_VERSION}"
            )
        if type(decoded.payload) is not cls:
            raise SerializationError(f"snapshot payload is {type(decoded.payload).__name__}, expected {cls.__name__}")
        payload_header = decoded.payload.header
        envelope_metadata = (
            decoded.schema_name,
            decoded.schema_version,
            decoded.producer_version,
            decoded.feature_set,
        )
        payload_metadata = (
            payload_header.schema_name,
            payload_header.schema_version,
            payload_header.producer_version,
            payload_header.feature_set,
        )
        if envelope_metadata != payload_metadata:
            raise SerializationError("snapshot envelope metadata does not match its payload header")
        expected = content_digest(decoded.payload, f"ir:{cls.SCHEMA_NAME}")
        if decoded.content_digest != expected:
            raise SerializationError("canonical IR snapshot digest mismatch")
        decoded.payload.require_valid()
        return decoded.payload

    def verify(self) -> VerificationReport:
        raise NotImplementedError

    def require_valid(self) -> None:
        self.verify().require_ok(type(self).__name__)

    def _verify_common(self, bag: DiagnosticBag) -> None:
        if self.header.schema_name != self.SCHEMA_NAME:
            bag.error(
                "schema.name",
                f"expected {self.SCHEMA_NAME!r}, got {self.header.schema_name!r}",
                "header",
                "schema_name",
            )
        if self.header.schema_version != self.SCHEMA_VERSION:
            bag.error(
                "schema.version",
                f"expected {self.SCHEMA_VERSION}, got {self.header.schema_version}",
                "header",
                "schema_version",
            )
        try:
            canonical_dumps(self)
        except SerializationError as error:
            bag.error("serialization.unsupported", str(error), "<snapshot>")


def verify_unique_ids(
    bag: DiagnosticBag,
    entities: Sequence[Entity],
    id_of: Callable[[Entity], StableId],
    path: str,
) -> None:
    seen = set()
    for index, entity in enumerate(entities):
        identifier = id_of(entity)
        if identifier in seen:
            bag.error("id.duplicate", f"duplicate identifier {identifier}", path, str(index), "id")
        seen.add(identifier)


def verify_ordered_dag(
    bag: DiagnosticBag,
    entities: Sequence[Entity],
    id_of: Callable[[Entity], StableId],
    dependencies_of: Callable[[Entity], Iterable[StableId]],
    path: str,
) -> None:
    """Verify references and require canonical topological sequence order."""

    identifiers = {id_of(entity) for entity in entities}
    seen = set()
    for index, entity in enumerate(entities):
        identifier = id_of(entity)
        dependencies = tuple(dependencies_of(entity))
        if len(set(dependencies)) != len(dependencies):
            bag.error("dag.duplicate_dependency", "dependencies must be unique", path, str(index), "dependencies")
        if identifier in dependencies:
            bag.error("dag.self_dependency", f"{identifier} depends on itself", path, str(index), "dependencies")
        for dependency in dependencies:
            if dependency not in identifiers:
                bag.error(
                    "reference.unknown",
                    f"dependency {dependency} is not declared",
                    path,
                    str(index),
                    "dependencies",
                )
            elif dependency not in seen:
                bag.error(
                    "dag.not_topological",
                    f"dependency {dependency} must precede {identifier}",
                    path,
                    str(index),
                    "dependencies",
                    hint="store canonical DAG nodes in topological order",
                )
        seen.add(identifier)


def verify_known_references(
    bag: DiagnosticBag,
    references: Iterable[StableId],
    known: Iterable[StableId],
    *path: str,
) -> None:
    known_set = set(known)
    for reference in references:
        if reference not in known_set:
            bag.error("reference.unknown", f"unknown reference {reference}", *path)


def verify_nonnegative_scalar(bag: DiagnosticBag, value: Scalar, *path: str) -> None:
    if isinstance(value, bool):
        bag.error("scalar.invalid", "boolean is not a scalar quantity", *path)
    elif not isinstance(value, (int, float, Symbol, ScalarExpr)):
        bag.error("scalar.invalid", f"unsupported scalar quantity {value!r}", *path)
    elif isinstance(value, float) and not math.isfinite(value):
        bag.error("scalar.nonfinite", "quantity must be finite", *path)
    elif isinstance(value, (int, float)) and value < 0:
        bag.error("scalar.negative", "quantity must not be negative", *path)


def reject_reserved_attributes(
    bag: DiagnosticBag,
    attributes: FrozenDict,
    reserved: frozenset[str],
    *path: str,
) -> None:
    """Reject semantic fields smuggled through extension dictionaries."""

    def walk(value: Any, location: tuple[str, ...]) -> None:
        if isinstance(value, FrozenDict):
            for key, item in value.items():
                if key.lower() in reserved:
                    bag.error(
                        "attribute.reserved",
                        f"{key!r} is a semantic field and is illegal in this IR layer",
                        *(location + (key,)),
                    )
                walk(item, location + (key,))
        elif isinstance(value, (tuple, frozenset)):
            values = value if isinstance(value, tuple) else tuple(sorted(value, key=repr))
            for index, item in enumerate(values):
                walk(item, location + (str(index),))

    walk(attributes, tuple(path))

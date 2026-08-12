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
from dataclasses import field, replace
from enum import Enum
from typing import Any, ClassVar, TypeAlias, TypeVar

from blueprinting.schema.authoring import (
    ContentDigest,
    NonEmptyText,
    NonNegativeInt,
    PositiveFiniteFloat,
    PositiveInt,
    StableName,
    enum,
    is_adt_variant,
    record,
)
from blueprinting.schema.codec import canonical_dumps, canonical_loads, content_digest
from blueprinting.schema.diagnostics import Diagnostic, DiagnosticSet
from blueprinting.schema.errors import SerializationError
from blueprinting.schema.frozen import FrozenDict
from blueprinting.schema.result import Checked, Err, Ok, checked

from ..errors import DiagnosticBag, IRVerificationError, VerificationReport
from ..expr import Scalar, ScalarExpr, ScalarExprVariant, Symbol
from ..ids import StableId

_DIGEST_RE = re.compile(r"^[0-9a-f]{40}$")
_KNOWN_TARGET_DIALECTS = frozenset({"cuda", "lpu", "nccl", "rccl", "rocm"})
TYPED_SEMANTICS_FEATURE = "typed-semantics"
REQUIRED_IR_FEATURES = frozenset({TYPED_SEMANTICS_FEATURE})
IR = TypeVar("IR", bound="CanonicalIRMixin")
Entity = TypeVar("Entity")
TensorDimension: TypeAlias = PositiveInt | PositiveFiniteFloat | Symbol | ScalarExprVariant


def is_content_digest(value: str) -> bool:
    return isinstance(value, str) and _DIGEST_RE.fullmatch(value) is not None


@record("blueprinting.ir.schema-version", order=True)
class SchemaVersion:
    """Semantic version of one serialized IR schema."""

    major: NonNegativeInt
    minor: NonNegativeInt = 0
    patch: NonNegativeInt = 0

    @classmethod
    def parse(cls, value: str) -> SchemaVersion:
        parts = value.split(".")
        if len(parts) != 3 or any(not part.isdigit() for part in parts):
            raise ValueError(f"invalid schema version: {value!r}")
        return cls(*(int(part) for part in parts))

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


@record("blueprinting.ir.header")
class IRHeader:
    """Version and provenance header embedded in every canonical IR."""

    schema_name: StableName
    schema_version: SchemaVersion
    producer_version: NonEmptyText = "0.0.0"
    feature_set: frozenset[StableName] = frozenset()
    parent_digests: tuple[ContentDigest, ...] = ()

    def with_parents(self, *digests: str) -> IRHeader:
        return replace(self, parent_digests=tuple(digests))


def make_header(
    schema_name: str,
    schema_version: SchemaVersion,
    *,
    parent_digests: Iterable[str] = (),
    features: Iterable[str] = (),
    producer_version: str = "0.0.0",
) -> IRHeader:
    return IRHeader(
        schema_name=schema_name,
        schema_version=schema_version,
        producer_version=producer_version,
        feature_set=REQUIRED_IR_FEATURES | frozenset(features),
        parent_digests=tuple(parent_digests),
    )


@record("blueprinting.ir.snapshot")
class IRSnapshot:
    """Self-checking persistence envelope for a canonical IR value."""

    schema_name: StableName
    schema_version: SchemaVersion
    producer_version: NonEmptyText
    feature_set: frozenset[StableName]
    content_digest: ContentDigest
    payload: Any


@enum("blueprinting.ir.effect-kind")
class EffectKind(Enum):
    READ = "read"
    WRITE = "write"
    STATE = "state"
    RANDOM = "random"
    IO = "io"


@record("blueprinting.ir.effect")
class Effect:
    kind: EffectKind
    resource: NonEmptyText


@record("blueprinting.ir.operation-name", order=True)
class OperationName:
    """Structured operation identity; dialect is never inferred from a string."""

    dialect: StableName
    name: StableName

    @classmethod
    def parse(cls, value: str) -> OperationName:
        dialect, separator, name = value.partition(".")
        if not separator:
            raise ValueError("operation names must use '<dialect>.<name>' form")
        return cls(dialect=dialect, name=name)

    def __str__(self) -> str:
        return f"{self.dialect}.{self.name}"


def is_known_target_dialect(operation: OperationName) -> bool:
    """Recognize built-in target dialects forbidden before target binding."""

    return operation.dialect.lower() in _KNOWN_TARGET_DIALECTS


@record("blueprinting.ir.tensor-type")
class TensorType:
    """Target-neutral logical tensor type."""

    shape: tuple[TensorDimension, ...]
    dtype: StableName
    layout: tuple[int, ...] | None = None
    attributes: FrozenDict = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
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
    def _decode_json(cls: type[IR], payload: str) -> IR:
        decoded = canonical_loads(payload)
        if not isinstance(decoded, IRSnapshot):
            raise SerializationError("canonical IR JSON must contain an IR snapshot envelope")
        if decoded.schema_name != cls.SCHEMA_NAME or decoded.schema_version != cls.SCHEMA_VERSION:
            raise SerializationError(
                f"snapshot schema {decoded.schema_name}@{decoded.schema_version} is not "
                f"{cls.SCHEMA_NAME}@{cls.SCHEMA_VERSION}"
            )
        missing_features = REQUIRED_IR_FEATURES - decoded.feature_set
        if missing_features:
            rendered = ", ".join(sorted(missing_features))
            raise SerializationError(f"snapshot is missing required feature epoch(s): {rendered}")
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

    @classmethod
    def from_json(cls: type[IR], payload: str) -> Checked[IR]:
        """Decode an exact-schema snapshot without exception control flow."""

        try:
            return Ok(cls._decode_json(payload))
        except IRVerificationError as error:
            return Err(DiagnosticSet(tuple(error.diagnostics)))
        except SerializationError as error:
            return Err(
                DiagnosticSet.of(
                    Diagnostic("serialization.snapshot", str(error), ("snapshot",)),
                )
            )

    @classmethod
    def require_from_json(cls: type[IR], payload: str) -> IR:
        """Explicit exception adapter for trusted internal/replay boundaries."""

        return cls.from_json(payload).or_raise(
            lambda diagnostics: SerializationError("; ".join(item.render() for item in diagnostics.errors))
        )

    @classmethod
    def load_migrated(cls: type[IR], payload: str, *, registry: Any = None) -> Checked[IR]:
        """Load through an explicit registered migration path."""

        if registry is None:
            from ..schema_migration import DEFAULT_SCHEMA_MIGRATIONS

            registry = DEFAULT_SCHEMA_MIGRATIONS
        try:
            result = registry.migrate_json(
                payload,
                schema_name=cls.SCHEMA_NAME,
                target_version=cls.SCHEMA_VERSION,
            )
        except SerializationError as error:
            return Err(DiagnosticSet.of(Diagnostic("serialization.migration", str(error), ("snapshot",))))
        return cls.from_json(result.payload)

    def diagnostics(self) -> VerificationReport:
        raise NotImplementedError

    def verify(self: IR) -> Checked[IR]:
        """Return the immutable snapshot or all expected verifier failures."""

        return checked(self, self.diagnostics())

    def require_valid(self) -> None:
        self.diagnostics().require_ok(type(self).__name__)

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
        missing_features = REQUIRED_IR_FEATURES - self.header.feature_set
        if missing_features:
            bag.error(
                "schema.feature_epoch",
                f"missing required feature epoch(s): {', '.join(sorted(missing_features))}",
                "header",
                "feature_set",
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
    elif not (isinstance(value, (int, float, Symbol)) or is_adt_variant(value, ScalarExpr)):
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

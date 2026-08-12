"""Stable, typed identifiers and lowering lineage."""

from __future__ import annotations

import hashlib
import re
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Annotated, Any, ClassVar, TypeAlias, TypeVar

from blueprinting.schema.authoring import NonEmptyText, ValueConstraint, enum, record
from blueprinting.schema.codec import canonical_dumps

from .errors import InvalidIdError

_ID_RE = re.compile(r"^[0-9a-f]{32}$")
IdType = TypeVar("IdType", bound="StableId")


@dataclass(frozen=True, order=True)
class StableId:
    """Base implementation for a typed 128-bit synthesis identifier."""

    value: str
    PREFIX: ClassVar[str] = "id"

    def __post_init__(self) -> None:
        if not isinstance(self.value, str) or _ID_RE.fullmatch(self.value) is None:
            raise InvalidIdError(f"invalid {type(self).__name__} value: {self.value!r}")

    @classmethod
    def new(cls: type[IdType]) -> IdType:
        return cls(uuid.uuid4().hex)

    @classmethod
    def derive(cls: type[IdType], *parts: Any) -> IdType:
        hasher = hashlib.blake2b(digest_size=16)
        hasher.update(cls.PREFIX.encode("ascii"))
        hasher.update(b"\x00")
        hasher.update(canonical_dumps(tuple(parts)).encode("utf-8"))
        return cls(hasher.hexdigest())

    @classmethod
    def parse(cls: type[IdType], value: str) -> IdType:
        prefix = f"{cls.PREFIX}:"
        if not value.startswith(prefix):
            raise InvalidIdError(f"expected {cls.PREFIX!r} identifier, got {value!r}")
        return cls(value[len(prefix) :])

    def __str__(self) -> str:
        return f"{self.PREFIX}:{self.value}"


@record("blueprinting.ir.id.node", order=True)
class NodeId(StableId):
    PREFIX: ClassVar[str] = "node"


@record("blueprinting.ir.id.value", order=True)
class ValueId(StableId):
    PREFIX: ClassVar[str] = "value"


@record("blueprinting.ir.id.buffer", order=True)
class BufferId(StableId):
    PREFIX: ClassVar[str] = "buffer"


@record("blueprinting.ir.id.command", order=True)
class CommandId(StableId):
    PREFIX: ClassVar[str] = "command"


@record("blueprinting.ir.id.instruction", order=True)
class InstructionId(StableId):
    PREFIX: ClassVar[str] = "instruction"


@record("blueprinting.ir.id.device", order=True)
class DeviceId(StableId):
    PREFIX: ClassVar[str] = "device"


@record("blueprinting.ir.id.queue", order=True)
class QueueId(StableId):
    PREFIX: ClassVar[str] = "queue"


@record("blueprinting.ir.id.memory-region", order=True)
class MemoryRegionId(StableId):
    PREFIX: ClassVar[str] = "memory-region"


@record("blueprinting.ir.id.token", order=True)
class TokenId(StableId):
    PREFIX: ClassVar[str] = "token"


@enum("blueprinting.ir.lineage-kind")
class LineageKind(Enum):
    ROOT = "root"
    PRESERVED = "preserved"
    CLONED = "cloned"
    DECOMPOSED = "decomposed"
    FUSED = "fused"
    LOWERED = "lowered"
    GENERATED = "generated"


LineageSources: TypeAlias = Annotated[
    tuple[StableId, ...],
    ValueConstraint.UNIQUE_ITEMS,
]


@record("blueprinting.ir.lineage")
class Lineage:
    """Typed provenance from source entities to one lowering product."""

    kind: LineageKind
    transform: NonEmptyText
    sources: LineageSources = ()

    def __post_init__(self) -> None:
        if self.kind is LineageKind.ROOT and self.sources:
            raise ValueError("root lineage cannot have source IDs")

    @classmethod
    def root(cls, transform: str = "import") -> Lineage:
        return cls(kind=LineageKind.ROOT, transform=transform)

    @classmethod
    def lowered(cls, transform: str, sources: Iterable[StableId]) -> Lineage:
        return cls(kind=LineageKind.LOWERED, transform=transform, sources=tuple(sources))

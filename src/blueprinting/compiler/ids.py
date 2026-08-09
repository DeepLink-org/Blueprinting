"""Stable, typed identifiers and lowering lineage."""

from __future__ import annotations

import hashlib
import re
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, TypeVar

from .codec import canonical_dumps, enum_type, record_type
from .errors import InvalidIdError

_ID_RE = re.compile(r"^[0-9a-f]{32}$")
IdType = TypeVar("IdType", bound="StableId")


@dataclass(frozen=True, order=True)
class StableId:
    """Base implementation for a typed 128-bit compiler identifier."""

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


@record_type("compiler.id.node")
@dataclass(frozen=True, order=True)
class NodeId(StableId):
    PREFIX: ClassVar[str] = "node"


@record_type("compiler.id.value")
@dataclass(frozen=True, order=True)
class ValueId(StableId):
    PREFIX: ClassVar[str] = "value"


@record_type("compiler.id.buffer")
@dataclass(frozen=True, order=True)
class BufferId(StableId):
    PREFIX: ClassVar[str] = "buffer"


@record_type("compiler.id.command")
@dataclass(frozen=True, order=True)
class CommandId(StableId):
    PREFIX: ClassVar[str] = "command"


@record_type("compiler.id.instruction")
@dataclass(frozen=True, order=True)
class InstructionId(StableId):
    PREFIX: ClassVar[str] = "instruction"


@record_type("compiler.id.device")
@dataclass(frozen=True, order=True)
class DeviceId(StableId):
    PREFIX: ClassVar[str] = "device"


@record_type("compiler.id.queue")
@dataclass(frozen=True, order=True)
class QueueId(StableId):
    PREFIX: ClassVar[str] = "queue"


@record_type("compiler.id.memory_region")
@dataclass(frozen=True, order=True)
class MemoryRegionId(StableId):
    PREFIX: ClassVar[str] = "memory-region"


@record_type("compiler.id.token")
@dataclass(frozen=True, order=True)
class TokenId(StableId):
    PREFIX: ClassVar[str] = "token"


@enum_type("compiler.lineage_kind")
class LineageKind(Enum):
    ROOT = "root"
    PRESERVED = "preserved"
    CLONED = "cloned"
    DECOMPOSED = "decomposed"
    FUSED = "fused"
    LOWERED = "lowered"
    GENERATED = "generated"


@record_type("compiler.lineage")
@dataclass(frozen=True)
class Lineage:
    """Typed provenance from source entities to one lowering product."""

    kind: LineageKind
    transform: str
    sources: tuple[StableId, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "sources", tuple(self.sources))
        if not isinstance(self.kind, LineageKind):
            raise TypeError("lineage kind must be LineageKind")
        if any(not isinstance(source, StableId) for source in self.sources):
            raise TypeError("lineage sources must be stable compiler IDs")
        if not self.transform:
            raise ValueError("lineage transform must not be empty")
        if len(set(self.sources)) != len(self.sources):
            raise ValueError("lineage sources must be unique")
        if self.kind is LineageKind.ROOT and self.sources:
            raise ValueError("root lineage cannot have source IDs")

    @classmethod
    def root(cls, transform: str = "import") -> Lineage:
        return cls(kind=LineageKind.ROOT, transform=transform)

    @classmethod
    def lowered(cls, transform: str, sources: Iterable[StableId]) -> Lineage:
        return cls(kind=LineageKind.LOWERED, transform=transform, sources=tuple(sources))

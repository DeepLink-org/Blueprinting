"""Closed-world canonical codec used by formal synthesis snapshots.

The decoder only constructs explicitly registered record and enum types.  It
never imports a class named by an input payload, which keeps IR loading
deterministic and avoids the usual arbitrary-import trap in generic codecs.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any, TypeVar

from .errors import SerializationError

T = TypeVar("T")

_RECORD_TYPES: dict[str, type[Any]] = {}
_RECORD_TAGS: dict[type[Any], str] = {}
_RECORD_FIELD_ALIASES: dict[str, dict[str, str]] = {}
_ENUM_TYPES: dict[str, type[Enum]] = {}
_ENUM_TAGS: dict[type[Enum], str] = {}


def record_type(
    tag: str,
    *,
    field_aliases: Mapping[str, str] | None = None,
) -> Callable[[type[T]], type[T]]:
    """Register a frozen dataclass and optional legacy field aliases."""

    if not isinstance(tag, str) or not tag:
        raise TypeError("canonical record tag must be a non-empty string")
    aliases = dict(field_aliases or {})
    if any(
        not isinstance(legacy, str) or not legacy or not isinstance(current, str) or not current or legacy == current
        for legacy, current in aliases.items()
    ):
        raise TypeError("canonical field aliases must map distinct non-empty strings")
    if len(set(aliases.values())) != len(aliases):
        raise TypeError("canonical field aliases must have unique destinations")

    def decorate(cls: type[T]) -> type[T]:
        if not is_dataclass(cls):
            raise TypeError(f"canonical record {cls.__name__} must be a dataclass")
        parameters = getattr(cls, "__dataclass_params__", None)
        if parameters is None or not parameters.frozen:
            raise TypeError(f"canonical record {cls.__name__} must be frozen")
        init_fields = {item.name for item in fields(cls) if item.init}
        unknown_destinations = set(aliases.values()) - init_fields
        if unknown_destinations:
            rendered = ", ".join(sorted(unknown_destinations))
            raise TypeError(f"canonical field aliases target unknown fields: {rendered}")
        conflicting_sources = set(aliases) & init_fields
        if conflicting_sources:
            rendered = ", ".join(sorted(conflicting_sources))
            raise TypeError(f"canonical field aliases shadow current fields: {rendered}")
        previous = _RECORD_TYPES.get(tag)
        if previous is not None and previous is not cls:
            raise RuntimeError(f"canonical record tag {tag!r} is already registered")
        previous_aliases = _RECORD_FIELD_ALIASES.get(tag)
        if previous_aliases is not None and previous_aliases != aliases:
            raise RuntimeError(f"canonical record tag {tag!r} has conflicting field aliases")
        _RECORD_TYPES[tag] = cls
        _RECORD_TAGS[cls] = tag
        _RECORD_FIELD_ALIASES[tag] = aliases
        return cls

    return decorate


def enum_type(tag: str) -> Callable[[type[T]], type[T]]:
    """Register an enum for canonical round-trip serialization."""

    if not isinstance(tag, str) or not tag:
        raise TypeError("canonical enum tag must be a non-empty string")

    def decorate(cls: type[T]) -> type[T]:
        if not issubclass(cls, Enum):
            raise TypeError(f"canonical enum {cls.__name__} must derive from Enum")
        previous = _ENUM_TYPES.get(tag)
        if previous is not None and previous is not cls:
            raise RuntimeError(f"canonical enum tag {tag!r} is already registered")
        _ENUM_TYPES[tag] = cls  # type: ignore[assignment]
        _ENUM_TAGS[cls] = tag  # type: ignore[index]
        return cls

    return decorate


def _encode(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            raise SerializationError("canonical IR does not permit NaN or infinity")
        return value

    if isinstance(value, bytes):
        return {"$bytes": base64.b64encode(value).decode("ascii")}

    enum_tag = _ENUM_TAGS.get(type(value))
    if enum_tag is not None:
        return {"$enum": enum_tag, "value": _encode(value.value)}

    record_tag = _RECORD_TAGS.get(type(value))
    if record_tag is not None:
        payload = {}
        for item in fields(value):
            if item.init:
                payload[item.name] = _encode(getattr(value, item.name))
        return {"$type": record_tag, "fields": payload}

    if isinstance(value, tuple):
        return {"$tuple": [_encode(item) for item in value]}

    if isinstance(value, frozenset):
        encoded = [_encode(item) for item in value]
        encoded.sort(key=_encoded_sort_key)
        return {"$frozenset": encoded}

    if isinstance(value, Mapping):
        items = []
        keys = tuple(value)
        if any(not isinstance(key, str) for key in keys):
            raise SerializationError("canonical mapping keys must be strings")
        for key in sorted(keys):
            items.append([key, _encode(value[key])])
        return {"$map": items}

    raise SerializationError(f"unsupported canonical value: {type(value).__module__}.{type(value).__qualname__}")


def _encoded_sort_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _decode(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SerializationError("canonical IR does not permit NaN or infinity")
        return value

    if isinstance(value, list):
        raise SerializationError("canonical arrays must use an explicit tuple or set tag")

    if not isinstance(value, dict):
        raise SerializationError(f"invalid canonical payload node: {type(value).__name__}")

    if set(value) == {"$bytes"}:
        try:
            return base64.b64decode(value["$bytes"], validate=True)
        except (TypeError, ValueError, binascii.Error) as error:
            raise SerializationError("invalid canonical bytes payload") from error

    if set(value) == {"$tuple"}:
        items = value["$tuple"]
        if not isinstance(items, list):
            raise SerializationError("canonical tuple payload must be an array")
        return tuple(_decode(item) for item in items)

    if set(value) == {"$frozenset"}:
        items = value["$frozenset"]
        if not isinstance(items, list):
            raise SerializationError("canonical frozenset payload must be an array")
        return frozenset(_decode(item) for item in items)

    if set(value) == {"$map"}:
        items = value["$map"]
        if not isinstance(items, list):
            raise SerializationError("canonical mapping payload must be an array")
        result = {}
        for pair in items:
            if not isinstance(pair, list) or len(pair) != 2 or not isinstance(pair[0], str):
                raise SerializationError("invalid canonical mapping entry")
            if pair[0] in result:
                raise SerializationError(f"duplicate canonical mapping key: {pair[0]!r}")
            result[pair[0]] = _decode(pair[1])
        return result

    if set(value) == {"$enum", "value"}:
        tag = value["$enum"]
        if not isinstance(tag, str):
            raise SerializationError("canonical enum tag must be a string")
        enum_cls = _ENUM_TYPES.get(tag)
        if enum_cls is None:
            raise SerializationError(f"unknown canonical enum tag: {tag!r}")
        try:
            return enum_cls(_decode(value["value"]))
        except (TypeError, ValueError) as error:
            raise SerializationError(f"invalid value for canonical enum {tag!r}") from error

    if set(value) == {"$type", "fields"}:
        tag = value["$type"]
        if not isinstance(tag, str):
            raise SerializationError("canonical record tag must be a string")
        record_cls = _RECORD_TYPES.get(tag)
        if record_cls is None:
            raise SerializationError(f"unknown canonical record tag: {tag!r}")
        payload = value["fields"]
        if not isinstance(payload, dict):
            raise SerializationError(f"fields for canonical record {tag!r} must be an object")
        try:
            aliases = _RECORD_FIELD_ALIASES.get(tag, {})
            decoded = {}
            for name, item in payload.items():
                current_name = aliases.get(name, name)
                if current_name in decoded:
                    raise SerializationError(
                        f"canonical record {tag!r} supplies both a current field and its legacy alias: {current_name!r}"
                    )
                decoded[current_name] = _decode(item)
            return record_cls(**decoded)
        except SerializationError:
            raise
        except (TypeError, ValueError) as error:
            raise SerializationError(f"invalid canonical record {tag!r}: {error}") from error

    raise SerializationError("invalid or ambiguous canonical tagged object")


def canonical_dumps(value: Any) -> str:
    """Serialize a registered value to deterministic UTF-8 JSON text."""

    return json.dumps(_encode(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_loads(payload: str) -> Any:
    """Deserialize canonical JSON using the closed-world type registry."""

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise SerializationError(f"duplicate canonical JSON key: {key!r}")
            result[key] = value
        return result

    def reject_nonfinite_constant(value: str) -> Any:
        raise SerializationError(f"canonical JSON does not permit {value}")

    try:
        raw = json.loads(
            payload,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite_constant,
        )
    except SerializationError:
        raise
    except (TypeError, ValueError) as error:
        raise SerializationError("invalid canonical JSON") from error
    return _decode(raw)


def content_digest(value: Any, domain: str = "blueprinting") -> str:
    """Return a domain-separated BLAKE2 digest of a canonical value."""

    hasher = hashlib.blake2b(digest_size=20)
    hasher.update(domain.encode("utf-8"))
    hasher.update(b"\x00")
    hasher.update(canonical_dumps(value).encode("utf-8"))
    return hasher.hexdigest()

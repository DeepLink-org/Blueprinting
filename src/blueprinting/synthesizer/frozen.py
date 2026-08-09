"""Small immutable containers used at IR boundaries."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import Any


def freeze(value: Any) -> Any:
    """Recursively freeze JSON-like extension data.

    Registered immutable synthesis records pass through unchanged.  Mutable
    mappings and sequences are copied so callers cannot mutate an IR snapshot
    through an alias retained outside the synthesizer.
    """

    if isinstance(value, FrozenDict):
        return value
    if isinstance(value, Mapping):
        return FrozenDict(value)
    if isinstance(value, tuple):
        return tuple(freeze(item) for item in value)
    if isinstance(value, list):
        return tuple(freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(freeze(item) for item in value)
    return value


def thaw(value: Any) -> Any:
    """Return a mutable JSON-like copy suitable for user-facing APIs."""

    if isinstance(value, FrozenDict):
        return {key: thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw(item) for item in value]
    if isinstance(value, frozenset):
        return [thaw(item) for item in sorted(value, key=repr)]
    return value


class FrozenDict(Mapping):
    """A compact, hashable mapping with recursively frozen values."""

    __slots__ = ("_hash", "_items")

    def __init__(
        self,
        source: Mapping[str, Any] | None = None,
        *,
        items: Iterable[tuple[str, Any]] | None = None,
    ) -> None:
        if source is not None and items is not None:
            raise TypeError("provide either source or items, not both")
        raw_items = source.items() if source is not None else (items or ())
        copied: dict[str, Any] = {}
        for key, value in raw_items:
            if not isinstance(key, str):
                raise TypeError("FrozenDict keys must be strings")
            if key in copied:
                raise ValueError(f"duplicate FrozenDict key: {key!r}")
            copied[key] = freeze(value)
        self._items = tuple(sorted(copied.items(), key=lambda pair: pair[0]))
        self._hash = None

    def __getitem__(self, key: str) -> Any:
        for item_key, value in self._items:
            if item_key == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash(self._items)
        return self._hash

    def __repr__(self) -> str:
        body = ", ".join(f"{key!r}: {value!r}" for key, value in self._items)
        return f"FrozenDict({{{body}}})"

    def __reduce__(self):
        """Use the public constructor for process and UI cache round-trips."""

        return FrozenDict, (dict(self._items),)

    def evolve(self, **changes: Any) -> FrozenDict:
        updated = dict(self._items)
        updated.update(changes)
        return FrozenDict(updated)

    def to_dict(self) -> dict[str, Any]:
        return {key: thaw(value) for key, value in self._items}


EMPTY_MAP = FrozenDict()

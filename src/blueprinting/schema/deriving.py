"""Small, type-checker-visible deriving helpers for canonical algebraic data."""

from __future__ import annotations

import math
import re
import types
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Annotated, Any, ClassVar, Literal, TypeGuard, TypeVar, Union, get_args, get_origin, get_type_hints

from typing_extensions import dataclass_transform

from .codec import enum_type, record_type
from .frozen import FrozenDict

T = TypeVar("T")

_WIRE_RE = re.compile(r"[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*")
_LOCAL_TAG_RE = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*")


@enum_type("blueprinting.schema.value-constraint")
class ValueConstraint(Enum):
    """Small closed vocabulary of reusable structural refinements."""

    NON_EMPTY = "non_empty"
    NON_BLANK = "non_blank"
    AT_LEAST_TWO_ITEMS = "at_least_two_items"
    UNIQUE_ITEMS = "unique_items"
    NON_EMPTY_ITEMS = "non_empty_items"
    NON_NEGATIVE = "non_negative"
    NON_NEGATIVE_ITEMS = "non_negative_items"
    POSITIVE = "positive"
    AT_LEAST_TWO = "at_least_two"
    AT_MOST_ONE = "at_most_one"
    FINITE = "finite"
    CANONICAL_LOWER_TEXT = "canonical_lower_text"
    STABLE_NAME = "stable_name"
    CONTENT_DIGEST = "content_digest"
    SYMBOL_NAME = "symbol_name"


@dataclass(frozen=True)
class ADTSpec:
    """One closed canonical sum-type family."""

    wire: str
    root: type[Any]

    @property
    def prefix(self) -> str:
        return self.wire

    def tag(self, local_tag: str) -> str:
        return f"{self.prefix}.{local_tag}"


@dataclass(frozen=True)
class VariantSpec:
    """Manifest entry for one explicitly named ADT constructor."""

    family: ADTSpec
    local_tag: str
    wire_tag: str
    constructor: type[Any]


_ADT_SPECS: dict[type[Any], ADTSpec] = {}
_VARIANT_SPECS: dict[type[Any], VariantSpec] = {}
_ADT_CLOSURES: dict[type[Any], tuple[type[Any], ...]] = {}
_ANNOTATION_CACHE: dict[type[Any], dict[str, Any]] = {}


def _structural_annotations(cls: type[Any]) -> dict[str, Any]:
    """Resolve one record's annotations once, after its decorator has returned."""

    annotations = _ANNOTATION_CACHE.get(cls)
    if annotations is None:
        annotations = get_type_hints(cls, include_extras=True)
        _ANNOTATION_CACHE[cls] = annotations
    return annotations


def _matches(value: Any, annotation: Any) -> bool:
    if annotation is Any:
        return True
    if annotation in _VARIANT_SPECS:
        return type(value) is annotation
    origin = get_origin(annotation)
    if origin is ClassVar:
        return True
    if origin is Annotated:
        base, *metadata = get_args(annotation)
        return _matches(value, base) and all(
            not isinstance(constraint, ValueConstraint) or _matches_constraint(value, constraint)
            for constraint in metadata
        )
    if origin is Literal:
        return any(type(value) is type(expected) and value == expected for expected in get_args(annotation))
    if origin in {types.UnionType, Union}:
        return any(_matches(value, item) for item in get_args(annotation))
    if origin is tuple:
        arguments = get_args(annotation)
        if not isinstance(value, tuple):
            return False
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            return all(_matches(item, arguments[0]) for item in value)
        return len(value) == len(arguments) and all(
            _matches(item, expected) for item, expected in zip(value, arguments)
        )
    if origin is frozenset:
        arguments = get_args(annotation)
        return isinstance(value, frozenset) and (not arguments or all(_matches(item, arguments[0]) for item in value))
    if origin is FrozenDict:
        arguments = get_args(annotation)
        return isinstance(value, FrozenDict) and (
            not arguments or all(_matches(item, arguments[0]) for item in value.values())
        )
    if origin in {dict, Mapping}:
        arguments = get_args(annotation)
        if not isinstance(value, Mapping):
            return False
        if len(arguments) != 2:
            return True
        key_type, value_type = arguments
        return all(_matches(key, key_type) and _matches(item, value_type) for key, item in value.items())
    if origin in {list, Sequence}:
        arguments = get_args(annotation)
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            return False
        return not arguments or all(_matches(item, arguments[0]) for item in value)
    if origin is not None:
        try:
            return isinstance(value, origin)
        except TypeError:
            return True
    if annotation is int and isinstance(value, bool):
        return False
    try:
        return isinstance(value, annotation)
    except TypeError:
        return True


def _matches_constraint(value: Any, constraint: ValueConstraint) -> bool:
    try:
        if constraint is ValueConstraint.NON_EMPTY:
            return len(value) > 0
        if constraint is ValueConstraint.NON_BLANK:
            return isinstance(value, str) and bool(value.strip())
        if constraint is ValueConstraint.AT_LEAST_TWO_ITEMS:
            return len(value) >= 2
        if constraint is ValueConstraint.UNIQUE_ITEMS:
            return len(set(value)) == len(value)
        if constraint is ValueConstraint.NON_EMPTY_ITEMS:
            return all(bool(item) for item in value)
        if constraint is ValueConstraint.NON_NEGATIVE:
            return bool(value >= 0)
        if constraint is ValueConstraint.NON_NEGATIVE_ITEMS:
            return all(item >= 0 for item in value)
        if constraint is ValueConstraint.POSITIVE:
            return bool(value > 0)
        if constraint is ValueConstraint.AT_LEAST_TWO:
            return bool(value >= 2)
        if constraint is ValueConstraint.AT_MOST_ONE:
            return bool(value <= 1)
        if constraint is ValueConstraint.FINITE:
            return not isinstance(value, float) or math.isfinite(value)
        if constraint is ValueConstraint.CANONICAL_LOWER_TEXT:
            return isinstance(value, str) and value == value.strip().lower()
        if constraint is ValueConstraint.STABLE_NAME:
            return isinstance(value, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]*", value) is not None
        if constraint is ValueConstraint.CONTENT_DIGEST:
            return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value) is not None
        if constraint is ValueConstraint.SYMBOL_NAME:
            return isinstance(value, str) and bool(value) and value.replace("_", "a").isalnum()
    except (TypeError, ValueError):
        return False
    return False


def _install_structural_post_init(cls: type[T]) -> None:
    local_invariant = cls.__dict__.get("__post_init__")
    invariants: list[Callable[[Any], None]] = []
    for base in cls.__bases__:
        inherited = getattr(base, "__record_invariants__", None)
        if inherited is not None:
            invariants.extend(inherited)
            continue
        inherited_post_init = getattr(base, "__post_init__", None)
        if inherited_post_init is not None:
            invariants.append(inherited_post_init)
    if local_invariant is not None:
        invariants.append(local_invariant)

    unique_invariants = tuple(dict.fromkeys(invariants))
    setattr(cls, "__record_invariants__", unique_invariants)  # noqa: B010

    def structural_post_init(self: Any) -> None:
        annotations = _structural_annotations(type(self))
        for name, annotation in annotations.items():
            if get_origin(annotation) is ClassVar:
                continue
            value = getattr(self, name)
            if not _matches(value, annotation):
                raise TypeError(f"{type(self).__name__}.{name} must match {annotation!r}, got {type(value).__name__}")
        for invariant in unique_invariants:
            invariant(self)

    setattr(cls, "__post_init__", structural_post_init)  # noqa: B010


@dataclass_transform(frozen_default=True)
def record(
    tag: str,
    *,
    order: bool = False,
) -> Callable[[type[T]], type[T]]:
    """Derive an immutable canonical record from an annotated class declaration."""

    if not isinstance(tag, str) or _WIRE_RE.fullmatch(tag) is None:
        raise TypeError("canonical record tag must be a dotted lowercase wire identity")

    def decorate(cls: type[T]) -> type[T]:
        if "__dataclass_fields__" in cls.__dict__:
            raise TypeError("@record derives its own frozen dataclass; do not combine it with @dataclass")
        _install_structural_post_init(cls)
        cls = dataclass(frozen=True, slots=True, order=order)(cls)
        return record_type(tag)(cls)

    return decorate


def enum(tag: str) -> Callable[[type[T]], type[T]]:
    """Register one closed enumeration through the public authoring surface."""

    if not isinstance(tag, str) or _WIRE_RE.fullmatch(tag) is None:
        raise TypeError("canonical enum tag must be a dotted lowercase wire identity")
    return enum_type(tag)


@dataclass_transform(frozen_default=True)
def adt(*, wire: str) -> Callable[[type[T]], type[T]]:
    """Declare the shared semantic wire namespace for a closed sum type."""

    if not isinstance(wire, str) or _WIRE_RE.fullmatch(wire) is None:
        raise TypeError("ADT wire namespace must be a dotted lowercase identity")

    def decorate(cls: type[T]) -> type[T]:
        if "__dataclass_fields__" in cls.__dict__:
            raise TypeError("@adt derives its own frozen dataclass; do not combine it with @dataclass")
        cls = dataclass(frozen=True, slots=True)(cls)
        if cls in _ADT_SPECS:
            raise RuntimeError(f"ADT family {cls.__name__} is already registered")
        if any(item.wire == wire for item in _ADT_SPECS.values()):
            raise RuntimeError(f"ADT wire namespace {wire!r} is already registered")
        spec = ADTSpec(wire, cls)
        _ADT_SPECS[cls] = spec
        setattr(cls, "__adt_spec__", spec)  # noqa: B010

        root = cls

        def adt_new(constructor: type[Any], *_args: Any, **_kwargs: Any) -> Any:
            if constructor is root:
                raise TypeError(f"ADT family {root.__name__} is abstract; instantiate one of its sealed variants")
            return object.__new__(constructor)

        setattr(cls, "__new__", staticmethod(adt_new))  # noqa: B010
        return cls

    return decorate


def _family_of(cls: type[Any]) -> ADTSpec:
    families = tuple(_ADT_SPECS[base] for base in cls.__bases__ if base in _ADT_SPECS)
    if len(families) != 1:
        raise TypeError("canonical variant must directly inherit from exactly one declared ADT family")
    return families[0]


@dataclass_transform(frozen_default=True)
def variant(local_tag: str) -> Callable[[type[T]], type[T]]:
    """Derive and register one explicitly named constructor of an ADT family."""

    if not isinstance(local_tag, str) or _LOCAL_TAG_RE.fullmatch(local_tag) is None:
        raise TypeError("variant tag must be a short lowercase kebab-case identity")

    def decorate(cls: type[T]) -> type[T]:
        family = _family_of(cls)
        if family.root in _ADT_CLOSURES:
            raise RuntimeError(f"ADT family {family.root.__name__} is sealed and cannot accept late variants")
        if any(item.family == family and item.local_tag == local_tag for item in _VARIANT_SPECS.values()):
            raise RuntimeError(f"ADT family {family.root.__name__} already declares variant {local_tag!r}")
        constructor = record(family.tag(local_tag))(cls)
        spec = VariantSpec(family, local_tag, family.tag(local_tag), constructor)
        _VARIANT_SPECS[constructor] = spec
        setattr(constructor, "__variant_spec__", spec)  # noqa: B010
        return constructor

    return decorate


def adt_manifest(family: type[Any]) -> tuple[VariantSpec, ...]:
    """Return a deterministic documentation/schema manifest for one ADT family."""

    spec = _ADT_SPECS.get(family)
    if spec is None:
        raise TypeError(f"{family.__name__} is not a declared ADT family")
    return tuple(
        sorted((item for item in _VARIANT_SPECS.values() if item.family == spec), key=lambda item: item.local_tag)
    )


def _union_members(variants: Any) -> tuple[type[Any], ...]:
    origin = get_origin(variants)
    members = get_args(variants) if origin in {types.UnionType, Union} else (variants,)
    if not members or any(not isinstance(item, type) for item in members):
        raise TypeError("an ADT closure must contain concrete constructor types")
    return tuple(members)


def seal_adt(family: type[Any], variants: Any) -> Any:
    """Declare the explicit runtime closure corresponding to a static Union alias."""

    if family not in _ADT_SPECS:
        raise TypeError(f"{getattr(family, '__name__', family)!r} is not a declared ADT family")
    members = _union_members(variants)
    if any(_VARIANT_SPECS.get(item) is None for item in members):
        raise TypeError("an ADT closure may contain only registered variants")
    if any(_VARIANT_SPECS[item].family.root is not family for item in members):
        raise TypeError("an ADT closure cannot mix constructors from different families")
    if len(set(members)) != len(members):
        raise ValueError("an ADT closure cannot repeat a constructor")
    registered = tuple(item.constructor for item in adt_manifest(family))
    if set(members) != set(registered):
        missing = tuple(item.__name__ for item in registered if item not in members)
        unknown = tuple(item.__name__ for item in members if item not in registered)
        detail = []
        if missing:
            detail.append(f"missing {', '.join(missing)}")
        if unknown:
            detail.append(f"unknown {', '.join(unknown)}")
        raise ValueError(f"ADT closure must contain exactly its registered variants: {'; '.join(detail)}")
    previous = _ADT_CLOSURES.get(family)
    if previous is not None and previous != registered:
        raise RuntimeError(f"ADT family {family.__name__} is already sealed with another closure")
    _ADT_CLOSURES[family] = registered
    return variants


def adt_closure(family: type[Any]) -> tuple[type[Any], ...] | None:
    """Return the declared constructor closure, if the family has been sealed."""

    return _ADT_CLOSURES.get(family)


def is_adt_variant(value: object, family: type[T]) -> TypeGuard[T]:
    """Return whether ``value`` is an exact constructor in a sealed family."""

    if family not in _ADT_SPECS:
        raise TypeError(f"{getattr(family, '__name__', family)!r} is not a declared ADT family")
    closure = _ADT_CLOSURES.get(family)
    if closure is None:
        raise TypeError(f"ADT family {family.__name__} is not sealed")
    return type(value) in closure


def require_adt_variant(value: object, family: type[Any], subject: str = "value") -> None:
    """Reject family roots, subclasses, and constructors outside the exact closure."""

    if not is_adt_variant(value, family):
        actual = f"{type(value).__module__}.{type(value).__qualname__}"
        raise TypeError(f"{subject} must be an exact sealed variant of {family.__name__}, got {actual}")


def adt_family_manifest() -> tuple[ADTSpec, ...]:
    return tuple(sorted(_ADT_SPECS.values(), key=lambda item: item.wire))


__all__ = [
    "ADTSpec",
    "VariantSpec",
    "adt",
    "adt_closure",
    "adt_family_manifest",
    "adt_manifest",
    "is_adt_variant",
    "record",
    "require_adt_variant",
    "seal_adt",
    "variant",
]

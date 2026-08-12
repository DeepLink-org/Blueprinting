"""Runtime compilation of canonical records and algebraic type declarations."""

from __future__ import annotations

import types
from dataclasses import MISSING, Field, dataclass, fields, is_dataclass
from enum import Enum
from typing import Annotated, Any, ClassVar, Literal, TypeVar, Union, get_args, get_origin, get_type_hints

from .codec import canonical_dumps, canonical_type_manifest, content_digest
from .deriving import (
    ADTSpec,
    adt_closure,
    adt_family_manifest,
    adt_manifest,
)
from .diagnostics import Diagnostic, DiagnosticBag, DiagnosticSet
from .result import Checked, checked


@dataclass(frozen=True, order=True)
class CanonicalFieldContract:
    name: str
    annotation: str
    init: bool
    keyword_only: bool
    default_kind: str
    default_identity: str | None
    default_value: str | None


@dataclass(frozen=True, order=True)
class CanonicalEnumMemberContract:
    name: str
    canonical_name: str
    value: str


@dataclass(frozen=True, order=True)
class CanonicalTypeContract:
    kind: str
    wire: str
    python_type: str
    fields: tuple[CanonicalFieldContract, ...] = ()
    enum_members: tuple[CanonicalEnumMemberContract, ...] = ()


@dataclass(frozen=True, order=True)
class AlgebraicFamilyContract:
    wire: str
    python_type: str
    variants: tuple[str, ...]
    sealed_variants: tuple[str, ...]


@dataclass(frozen=True)
class TypeUniverse:
    """Immutable runtime view of the currently loaded typed algebraic declarations."""

    canonical_types: tuple[CanonicalTypeContract, ...]
    algebraic_families: tuple[AlgebraicFamilyContract, ...]

    @property
    def digest(self) -> str:
        return content_digest(
            (
                tuple(
                    (
                        item.kind,
                        item.wire,
                        item.python_type,
                        tuple(
                            (
                                field.name,
                                field.annotation,
                                field.init,
                                field.keyword_only,
                                field.default_kind,
                                field.default_identity,
                                field.default_value,
                            )
                            for field in item.fields
                        ),
                        tuple((member.name, member.canonical_name, member.value) for member in item.enum_members),
                    )
                    for item in self.canonical_types
                ),
                tuple(
                    (item.wire, item.python_type, item.variants, item.sealed_variants)
                    for item in self.algebraic_families
                ),
            ),
            "type-universe",
        )


def _type_name(value: type[Any]) -> str:
    return f"{value.__module__}.{value.__qualname__}"


def _callable_name(value: Any) -> str:
    module = getattr(value, "__module__", type(value).__module__)
    qualname = getattr(value, "__qualname__", type(value).__qualname__)
    return f"{module}.{qualname}"


def _annotation_shape(annotation: Any) -> str:
    """Return a cross-version structural identity for one resolved annotation."""

    if annotation is Any:
        return "typing.Any"
    if annotation is None or annotation is type(None):
        return "builtins.None"
    if annotation is Ellipsis:
        return "builtins.Ellipsis"
    if isinstance(annotation, TypeVar):
        constraints = tuple(_annotation_shape(item) for item in annotation.__constraints__)
        bound = _annotation_shape(annotation.__bound__) if annotation.__bound__ is not None else ""
        return f"typevar[{annotation.__name__};{','.join(constraints)};{bound}]"
    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if origin in {types.UnionType, Union}:
        return f"union[{','.join(sorted(_annotation_shape(item) for item in arguments))}]"
    if origin is Annotated:
        base, *metadata = arguments
        rendered_metadata = ",".join(canonical_dumps(item) for item in metadata)
        return f"annotated[{_annotation_shape(base)};{rendered_metadata}]"
    if origin is Literal:
        return f"literal[{','.join(canonical_dumps(item) for item in arguments)}]"
    if origin is ClassVar:
        return f"classvar[{_annotation_shape(arguments[0])}]"
    if origin is not None:
        origin_name = _type_name(origin) if isinstance(origin, type) else str(origin)
        return f"{origin_name}[{','.join(_annotation_shape(item) for item in arguments)}]"
    if isinstance(annotation, type):
        return _type_name(annotation)
    return str(annotation)


def _adt_roots_in(annotation: Any) -> tuple[type[Any], ...]:
    roots = {item.root for item in adt_family_manifest()}
    found: set[type[Any]] = set()

    def visit(item: Any) -> None:
        if isinstance(item, type) and item in roots:
            found.add(item)
            return
        origin = get_origin(item)
        if origin is Annotated:
            arguments = get_args(item)
            if arguments:
                visit(arguments[0])
            return
        for argument in get_args(item):
            visit(argument)

    visit(annotation)
    return tuple(sorted(found, key=_type_name))


def _field_default(field: Field[Any], bag: DiagnosticBag, wire: str) -> tuple[str, str | None, str | None]:
    if field.default is not MISSING:
        try:
            return "value", None, canonical_dumps(field.default)
        except Exception as error:
            bag.error(
                "contract.schema.invalid_default",
                f"field {field.name!r} has a non-canonical default: {error}",
                "type",
                wire,
                field.name,
            )
            return "value", None, f"<invalid:{type(field.default).__module__}.{type(field.default).__qualname__}>"
    if field.default_factory is not MISSING:
        factory = field.default_factory
        identity = _callable_name(factory)
        try:
            first = canonical_dumps(factory())
            second = canonical_dumps(factory())
            if first != second:
                raise ValueError("successive calls produced different canonical defaults")
            return "factory", identity, first
        except Exception as error:
            bag.error(
                "contract.schema.invalid_default_factory",
                f"field {field.name!r} default factory {identity} is not deterministic/canonical: {error}",
                "type",
                wire,
                field.name,
            )
            return "factory", identity, "<invalid>"
    return "required", None, None


def _canonical_type_contract(
    kind: str,
    wire: str,
    python_type: type[Any],
    bag: DiagnosticBag,
) -> CanonicalTypeContract:
    if kind == "record":
        if not is_dataclass(python_type):
            bag.error("contract.schema.not_dataclass", "registered record is not a dataclass", "type", wire)
            return CanonicalTypeContract(kind, wire, _type_name(python_type))
        annotations = get_type_hints(python_type, include_extras=True)
        field_contracts = []
        for field in fields(python_type):
            if not field.init:
                continue
            annotation = annotations.get(field.name, Any)
            roots = _adt_roots_in(annotation)
            if roots:
                bag.error(
                    "contract.schema.adt_root_field",
                    f"field {field.name!r} refers to abstract ADT root(s): "
                    f"{', '.join(_type_name(item) for item in roots)}; use the exact sealed union alias",
                    "type",
                    wire,
                    field.name,
                )
            default_kind, default_identity, default_value = _field_default(field, bag, wire)
            field_contracts.append(
                CanonicalFieldContract(
                    field.name,
                    _annotation_shape(annotation),
                    field.init,
                    field.kw_only is True,
                    default_kind,
                    default_identity,
                    default_value,
                )
            )
        return CanonicalTypeContract(kind, wire, _type_name(python_type), tuple(field_contracts))
    if kind == "enum":
        if not issubclass(python_type, Enum):
            bag.error("contract.schema.not_enum", "registered enum does not derive from Enum", "type", wire)
            return CanonicalTypeContract(kind, wire, _type_name(python_type))
        members = tuple(
            CanonicalEnumMemberContract(name, member.name, canonical_dumps(member.value))
            for name, member in python_type.__members__.items()
        )
        return CanonicalTypeContract(kind, wire, _type_name(python_type), enum_members=members)
    bag.error("contract.schema.unknown_kind", f"unknown canonical registration kind {kind!r}", "type", wire)
    return CanonicalTypeContract(kind, wire, _type_name(python_type))


def _family_contract(spec: ADTSpec, bag: DiagnosticBag) -> AlgebraicFamilyContract:
    manifest = adt_manifest(spec.root)
    variants = tuple(item.constructor for item in manifest)
    closure = adt_closure(spec.root)
    path = ("adt", spec.wire)
    if closure is None:
        bag.error(
            "contract.adt.unsealed",
            f"ADT family {_type_name(spec.root)} has no explicit Union closure",
            *path,
            hint="declare VariantAlias = A | B and call seal_adt(Family, VariantAlias)",
        )
        closure = ()
    else:
        missing = tuple(item for item in variants if item not in closure)
        unknown = tuple(item for item in closure if item not in variants)
        if missing:
            bag.error(
                "contract.adt.missing_variant",
                f"sealed closure omits: {', '.join(_type_name(item) for item in missing)}",
                *path,
            )
        if unknown:
            bag.error(
                "contract.adt.unknown_variant",
                f"sealed closure contains unregistered types: {', '.join(_type_name(item) for item in unknown)}",
                *path,
            )
    return AlgebraicFamilyContract(
        spec.wire,
        _type_name(spec.root),
        tuple(f"{item.local_tag}:{item.wire_tag}:{_type_name(item.constructor)}" for item in manifest),
        tuple(_type_name(item) for item in closure),
    )


def compile_type_universe(*, additional: DiagnosticSet = DiagnosticSet()) -> Checked[TypeUniverse]:
    """Compile loaded runtime declarations without importing any static checker."""

    bag = DiagnosticBag()
    bag.extend(additional.diagnostics)
    canonical_types = tuple(
        _canonical_type_contract(kind, wire, python_type, bag) for kind, wire, python_type in canonical_type_manifest()
    )
    families = tuple(_family_contract(item, bag) for item in adt_family_manifest())
    universe = TypeUniverse(canonical_types, families)
    return checked(universe, bag.report())


def contract_error(code: str, message: str, *path: str) -> Diagnostic:
    """Small helper for domain-specific contract compiler contributors."""

    return Diagnostic(code, message, tuple(path))


__all__ = [
    "AlgebraicFamilyContract",
    "CanonicalEnumMemberContract",
    "CanonicalFieldContract",
    "CanonicalTypeContract",
    "TypeUniverse",
    "compile_type_universe",
    "contract_error",
]

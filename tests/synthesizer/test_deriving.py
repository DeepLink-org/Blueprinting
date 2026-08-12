from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass

import pytest

from blueprinting.schema import canonical_dumps, canonical_loads
from blueprinting.schema.authoring import (
    adt,
    adt_manifest,
    is_adt_variant,
    record,
    seal_adt,
    variant,
)


@adt(wire="tests.expression")
class _Expression:
    pass


@variant("literal")
class _Literal(_Expression):
    value: int


@variant("pair")
class _Pair(_Expression):
    left: _Literal
    right: _Literal


_ExpressionVariant = _Literal | _Pair
seal_adt(_Expression, _ExpressionVariant)


def test_variant_derives_frozen_slotted_canonical_records_and_short_wire_tags() -> None:
    value = _Pair(_Literal(1), _Literal(2))

    assert canonical_loads(canonical_dumps(value)) == value
    assert not hasattr(value, "__dict__")
    with pytest.raises(FrozenInstanceError):
        value.left = _Literal(3)  # type: ignore[misc]
    assert [(item.local_tag, item.wire_tag) for item in adt_manifest(_Expression)] == [
        ("literal", "tests.expression.literal"),
        ("pair", "tests.expression.pair"),
    ]


def test_authoring_decorators_own_dataclass_derivation() -> None:
    @dataclass(frozen=True)
    class _LegacyRecord:
        value: int

    with pytest.raises(TypeError, match="@record derives its own frozen dataclass"):
        record("tests.legacy-record")(_LegacyRecord)

    @dataclass(frozen=True)
    class _LegacyFamily:
        pass

    with pytest.raises(TypeError, match="@adt derives its own frozen dataclass"):
        adt(wire="tests.legacy-family")(_LegacyFamily)


def test_variant_structural_validation_is_derived_from_annotations() -> None:
    with pytest.raises(TypeError, match="_Literal.value"):
        _Literal("not-an-int")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="_Pair.right"):
        _Pair(_Literal(1), "not-a-literal")  # type: ignore[arg-type]


def test_variant_requires_one_family_and_short_stable_tag() -> None:
    with pytest.raises(TypeError, match="kebab-case"):
        variant("Not_A_Tag")

    with pytest.raises(TypeError, match="exactly one"):

        @variant("orphan")
        class _Orphan:
            pass


def test_adt_root_is_abstract_and_late_variants_are_rejected() -> None:
    with pytest.raises(TypeError, match="is abstract"):
        _Expression()

    with pytest.raises(RuntimeError, match="sealed and cannot accept late variants"):

        @variant("late")
        class _Late(_Expression):
            pass


def test_adt_closure_requires_exact_membership() -> None:
    with pytest.raises(ValueError, match="exactly its registered variants"):
        seal_adt(_Expression, _Literal)


def test_registered_constructor_annotations_reject_unregistered_subclasses() -> None:
    class _RogueLiteral(_Literal):
        pass

    rogue = _RogueLiteral(1)

    assert not is_adt_variant(rogue, _Expression)
    with pytest.raises(TypeError, match="_Pair.left"):
        _Pair(rogue, _Literal(2))

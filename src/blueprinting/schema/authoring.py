"""Expert authoring surface for canonical records and closed algebraic data.

Application users do not need these helpers.  They are intentionally grouped
here for core schema and trusted dialect authors; registry and manifest
implementation details remain in :mod:`blueprinting.schema.deriving`.
"""

from .deriving import (
    ADTSpec,
    VariantSpec,
    adt,
    adt_manifest,
    is_adt_variant,
    record,
    require_adt_variant,
    seal_adt,
    variant,
)

__all__ = [
    "ADTSpec",
    "VariantSpec",
    "adt",
    "adt_manifest",
    "is_adt_variant",
    "record",
    "require_adt_variant",
    "seal_adt",
    "variant",
]

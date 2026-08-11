#!/usr/bin/env python3
"""Compile runtime type contracts without importing mypy."""

from __future__ import annotations

from blueprinting.contracts import compile_runtime_contracts
from blueprinting.schema import Err


def main() -> int:
    result = compile_runtime_contracts()
    if isinstance(result, Err):
        for diagnostic in result.error:
            print(diagnostic.render())
        return 1
    manifest = result.value
    print(
        "runtime type contracts: "
        f"{len(manifest.types.canonical_types)} records/enums, "
        f"{len(manifest.types.algebraic_families)} ADTs, "
        f"{len(manifest.derivations)} passes; digest={manifest.digest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

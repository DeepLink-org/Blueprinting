"""Verify that the base wheel contains presets and only the pinned PoC evidence slice."""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

MAX_UNCOMPRESSED_BYTES = 5 * 1024 * 1024


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        raise SystemExit("usage: check_wheel_contract.py DIST.whl")
    wheel = Path(argv[0])
    with zipfile.ZipFile(wheel) as archive:
        members = archive.infolist()
    names = tuple(item.filename for item in members)
    required_prefixes = (
        "blueprinting/presets/models/",
        "blueprinting/presets/systems/",
    )
    for prefix in required_prefixes:
        if not any(name.startswith(prefix) and name.endswith(".json") for name in names):
            raise SystemExit(f"wheel is missing JSON presets under {prefix}")
    evidence_prefix = "blueprinting/presets/evidence/vidur/phi2_a100_tp1/"
    required_evidence = {
        f"{evidence_prefix}LICENSE.vidur",
        f"{evidence_prefix}attention.csv",
        f"{evidence_prefix}manifest.json",
        f"{evidence_prefix}mlp.csv",
    }
    missing_evidence = required_evidence - set(names)
    if missing_evidence:
        raise SystemExit(f"wheel is missing pinned PoC evidence: {sorted(missing_evidence)!r}")
    unexpected_evidence = tuple(
        name for name in names if name.startswith("blueprinting/presets/evidence/") and name not in required_evidence
    )
    if unexpected_evidence:
        raise SystemExit(f"wheel contains unapproved evidence: {unexpected_evidence[:3]!r}")
    forbidden_prefixes = ("blueprinting/systems/", "data/evidence/")
    leaked = tuple(name for name in names if name.startswith(forbidden_prefixes))
    if leaked:
        raise SystemExit(f"wheel contains optional evidence: {leaked[:3]!r}")
    uncompressed_bytes = sum(item.file_size for item in members)
    if uncompressed_bytes > MAX_UNCOMPRESSED_BYTES:
        raise SystemExit(
            f"wheel expands to {uncompressed_bytes} bytes; base-wheel budget is {MAX_UNCOMPRESSED_BYTES} bytes"
        )
    print(f"wheel contract ok: {wheel.name}, {len(names)} files, {uncompressed_bytes} bytes uncompressed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

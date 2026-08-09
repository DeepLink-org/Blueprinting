"""Read-only access to repository configuration presets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ConfigCatalog:
    """Resolve presets by catalog membership, never by user-controlled paths."""

    data_root: Path

    def _entries(self, category: str) -> dict[str, Path]:
        directory = self.data_root / category
        return {path.name: path for path in sorted(directory.glob("*.json")) if path.is_file()}

    def names(self, category: str) -> tuple[str, ...]:
        return tuple(self._entries(category))

    def load(self, category: str, name: str) -> dict[str, Any]:
        try:
            path = self._entries(category)[name]
        except KeyError as error:
            raise KeyError(f"unknown {category} preset: {name!r}") from error
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
        if not isinstance(value, dict):
            raise TypeError(f"preset {path} must contain a JSON object")
        return value

    def preferred(self, category: str, preferred_name: str) -> str:
        names = self.names(category)
        if not names:
            raise FileNotFoundError(f"no JSON presets found in {self.data_root / category}")
        return preferred_name if preferred_name in names else names[0]


@lru_cache(maxsize=1)
def default_catalog() -> ConfigCatalog:
    package_presets = Path(__file__).resolve().parents[1] / "presets"
    if package_presets.is_dir():
        return ConfigCatalog(package_presets)
    repository_root = Path(__file__).resolve().parents[3]
    source_presets = repository_root / "data"
    if source_presets.is_dir():
        return ConfigCatalog(source_presets)
    raise FileNotFoundError("Blueprinting model and system presets are not installed")

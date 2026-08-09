from __future__ import annotations

from blueprinting.workbench.catalog import default_catalog


def test_default_catalog_exposes_packaged_model_and_system_presets() -> None:
    catalog = default_catalog()

    assert "gpt3-175B.json" in catalog.names("models")
    assert "a100_80g.json" in catalog.names("systems")
    assert catalog.load("models", "gpt3-175B.json")["hidden"] > 0
    assert catalog.load("systems", "a100_80g.json")["mem1"]["GiB"] == 80

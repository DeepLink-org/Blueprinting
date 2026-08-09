from __future__ import annotations

import importlib.util

import blueprinting.analysis as analysis


def test_analysis_is_a_top_level_blueprinting_package() -> None:
    assert analysis.__name__ == "blueprinting.analysis"
    assert importlib.util.find_spec("blueprinting.analysis") is not None
    assert importlib.util.find_spec("blueprinting.compiler.analysis") is None

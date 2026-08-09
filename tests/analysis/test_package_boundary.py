from __future__ import annotations

import importlib.util

import pytest

import blueprinting.analysis as analysis
import blueprinting.synthesizer as synthesizer


def test_analysis_is_a_top_level_blueprinting_package() -> None:
    assert analysis.__name__ == "blueprinting.analysis"
    assert importlib.util.find_spec("blueprinting.analysis") is not None
    assert importlib.util.find_spec("blueprinting.synthesizer.analysis") is None


def test_synthesizer_is_the_only_formal_synthesis_package() -> None:
    assert synthesizer.__name__ == "blueprinting.synthesizer"
    assert importlib.util.find_spec("blueprinting.synthesizer") is not None
    assert importlib.util.find_spec("blueprinting.compiler") is None
    with pytest.raises(ModuleNotFoundError):
        __import__("blueprinting.compiler")


def test_legacy_public_symbols_are_not_reexported() -> None:
    assert not hasattr(synthesizer, "CompilationSession")
    assert not hasattr(synthesizer, "CompilerError")

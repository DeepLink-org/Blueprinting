from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

import blueprinting.analysis as analysis
import blueprinting.synthesizer as synthesizer
import blueprinting.synthesizer.frontend as frontend
import blueprinting.system as system
import blueprinting.workload as workload

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "src" / "blueprinting"


def _absolute_imports(package: str) -> frozenset[str]:
    imports = set()
    for source in (PACKAGE_ROOT / package).rglob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imports.add(node.module)
    return frozenset(imports)


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


def test_workload_and_system_are_top_level_domain_packages() -> None:
    assert workload.__name__ == "blueprinting.workload"
    assert system.__name__ == "blueprinting.system"
    assert importlib.util.find_spec("blueprinting.workload") is not None
    assert importlib.util.find_spec("blueprinting.system") is not None
    assert importlib.util.find_spec("blueprinting.synthesizer.models") is None


def test_domain_ownership_is_not_hidden_by_compatibility_reexports() -> None:
    assert hasattr(workload, "TransformerModelSpec")
    assert not hasattr(workload, "build_transformer_model_ir")
    assert hasattr(frontend, "build_transformer_model_ir")
    assert hasattr(system, "SystemProfile")
    assert not hasattr(analysis, "SystemProfile")


def test_domain_packages_do_not_depend_on_each_other_or_analysis_policy() -> None:
    workload_imports = _absolute_imports("workload")
    system_imports = _absolute_imports("system")

    assert not any(name.startswith(("blueprinting.analysis", "blueprinting.system")) for name in workload_imports)
    assert not any(name.startswith(("blueprinting.analysis", "blueprinting.workload")) for name in system_imports)

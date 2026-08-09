from __future__ import annotations

import ast
import importlib
import importlib.util
from pathlib import Path

import pytest

import blueprinting.analysis as analysis
import blueprinting.mapping as mapping
import blueprinting.schema as schema
import blueprinting.synthesizer as synthesizer
import blueprinting.synthesizer.frontend as frontend
import blueprinting.system as system
import blueprinting.workload as workload

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "src" / "blueprinting"


def _imports(package: str) -> frozenset[str]:
    imports = set()
    for source in (PACKAGE_ROOT / package).rglob("*.py"):
        relative = source.relative_to(PACKAGE_ROOT).with_suffix("")
        module_parts = ("blueprinting", *relative.parts)
        if module_parts[-1] == "__init__":
            module_parts = module_parts[:-1]
        module = ".".join(module_parts)
        package_context = module if source.name == "__init__.py" else module.rpartition(".")[0]
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    relative_name = "." * node.level + (node.module or "")
                    imports.add(importlib.util.resolve_name(relative_name, package_context))
                elif node.module:
                    imports.add(node.module)
    return frozenset(imports)


def _assert_only_domain_dependencies(package: str, allowed: tuple[str, ...]) -> None:
    illegal = sorted(
        name
        for name in _imports(package)
        if name.startswith("blueprinting.") and not name.startswith(tuple(f"blueprinting.{item}" for item in allowed))
    )
    assert illegal == []


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
    assert mapping.__name__ == "blueprinting.mapping"
    assert schema.__name__ == "blueprinting.schema"
    assert importlib.util.find_spec("blueprinting.synthesizer.models") is None


def test_domain_ownership_is_not_hidden_by_compatibility_reexports() -> None:
    assert hasattr(workload, "TransformerModelSpec")
    assert not hasattr(workload, "TransformerTrainingMappingSpec")
    assert hasattr(mapping, "TransformerTrainingMappingSpec")
    assert hasattr(mapping, "NetworkTierBinding")
    assert not hasattr(workload, "build_transformer_model_ir")
    assert hasattr(frontend, "build_transformer_model_ir")
    assert hasattr(system, "SystemProfile")
    assert not hasattr(analysis, "SystemProfile")
    assert not hasattr(analysis, "PrimitiveInvocation")


def test_supported_architecture_dependencies_are_acyclic_and_layered() -> None:
    _assert_only_domain_dependencies("schema", ("schema",))
    _assert_only_domain_dependencies("workload", ("schema", "workload"))
    _assert_only_domain_dependencies("mapping", ("schema", "workload", "mapping"))
    _assert_only_domain_dependencies("system", ("schema", "system"))
    _assert_only_domain_dependencies(
        "synthesizer",
        ("schema", "workload", "mapping", "synthesizer"),
    )
    _assert_only_domain_dependencies(
        "analysis",
        ("schema", "workload", "mapping", "system", "synthesizer", "analysis"),
    )


def test_validation_is_outside_the_synthesizer_dependency_closure() -> None:
    assert importlib.util.find_spec("blueprinting.validation") is not None
    assert not any(name.startswith("blueprinting.validation") for name in _imports("synthesizer"))

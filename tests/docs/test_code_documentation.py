from __future__ import annotations

import importlib
import inspect
from pathlib import Path

from blueprinting.synthesizer.passes import DerivationPass

_ROOT = Path(__file__).resolve().parents[2]
_PASS_MODULES = (
    "blueprinting.synthesizer.stages.model.passes",
    "blueprinting.synthesizer.stages.distributed.passes",
    "blueprinting.synthesizer.stages.portable_plan.passes",
    "blueprinting.synthesizer.stages.concrete_plan.passes",
    "blueprinting.synthesizer.stages.machine.passes",
)
_IR_PAGES = {
    "model-ir": "blueprinting.synthesizer.stages.model.ir",
    "distributed-task-ir": "blueprinting.synthesizer.stages.distributed.ir",
    "portable-plan-ir": "blueprinting.synthesizer.stages.portable_plan.ir",
    "concrete-plan-ir": "blueprinting.synthesizer.stages.concrete_plan.ir",
    "machine-ir": "blueprinting.synthesizer.stages.machine.ir",
}


def _public_passes() -> tuple[type[DerivationPass], ...]:
    result = []
    for module_name in _PASS_MODULES:
        module = importlib.import_module(module_name)
        for name in module.__all__:
            value = getattr(module, name)
            if inspect.isclass(value) and issubclass(value, DerivationPass):
                assert value.__module__ == module_name, f"{name} must be defined by its owning stage, not forwarded"
                result.append(value)
    return tuple(result)


def test_every_public_stage_pass_has_renderable_theory_and_provenance() -> None:
    passes = _public_passes()
    assert {item.__name__ for item in passes} == {
        "DistributeTransformerTrainingPass",
        "DistributeTransformerInferencePass",
        "PlanTransformerTrainingPass",
        "PlanTransformerInferencePass",
        "BindReferenceQueueTargetPass",
        "BindReferenceSlotTargetPass",
    }

    for pass_type in passes:
        doc = inspect.getdoc(pass_type) or ""
        assert doc.count("$$") >= 2, f"{pass_type.__name__} needs a display equation in its source docstring"
        assert "References:" in doc, f"{pass_type.__name__} needs explicit provenance"
        assert "https://arxiv.org/" in doc or "no paper" in doc, (
            f"{pass_type.__name__} must cite primary research or explicitly disclaim a paper-derived algorithm"
        )


def test_pass_api_page_covers_every_public_stage_pass_in_both_locales() -> None:
    for locale in ("en", "zh"):
        page = (_ROOT / f"docs/reference/passes.{locale}.md").read_text()
        for pass_type in _public_passes():
            identifier = f"{pass_type.__module__}.{pass_type.__name__}"
            assert identifier in page


def test_every_canonical_ir_stage_has_a_bilingual_generated_api_page() -> None:
    for page_name, module_name in _IR_PAGES.items():
        for locale in ("en", "zh"):
            page = (_ROOT / f"docs/reference/{page_name}.{locale}.md").read_text()
            assert f"::: {module_name}" in page


def test_mkdocs_enables_source_api_and_math_rendering() -> None:
    config = (_ROOT / "mkdocs.yml").read_text()
    assert "- mkdocstrings:" in config
    assert "- pymdownx.arithmatex:" in config
    assert "- javascripts/mathjax.js" in config
    assert "reference/passes.md" in config
    mathjax = (_ROOT / "docs/javascripts/mathjax.js").read_text()
    assert "MathJax.typesetPromise()" in mathjax
    workflow = (_ROOT / ".github/workflows/docs.yml").read_text()
    assert "scripts/check_rendered_code_docs.py" in workflow
    assert '"src/blueprinting/**"' in workflow

"""Verify that generated API pages contain rendered pass theory and source."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
PASSES = (
    "DistributeTransformerTrainingPass",
    "DistributeTransformerInferencePass",
    "PlanTransformerTrainingPass",
    "PlanTransformerInferencePass",
    "BindReferenceQueueTargetPass",
    "BindReferenceSlotTargetPass",
)
IR_PAGES = (
    "model-ir",
    "distributed-task-ir",
    "portable-plan-ir",
    "concrete-plan-ir",
    "machine-ir",
)


def _read(path: Path, failures: list[str]) -> str:
    if not path.is_file():
        failures.append(f"missing generated page: {path.relative_to(ROOT)}")
        return ""
    return path.read_text()


def main() -> int:
    failures: list[str] = []
    for prefix in (Path(), Path("zh")):
        pass_page = SITE / prefix / "reference/passes/index.html"
        html = _read(pass_page, failures)
        if not html:
            continue
        for pass_name in PASSES:
            if pass_name not in html:
                failures.append(f"{pass_page.relative_to(ROOT)} omits {pass_name}")
        if html.count('class="arithmatex"') < len(PASSES):
            failures.append(f"{pass_page.relative_to(ROOT)} does not render one or more pass equations")
        if html.count("Source code in") < len(PASSES):
            failures.append(f"{pass_page.relative_to(ROOT)} does not render source for every pass")
        if "javascripts/mathjax.js" not in html:
            failures.append(f"{pass_page.relative_to(ROOT)} does not load the MathJax configuration")

        for page_name in IR_PAGES:
            ir_page = SITE / prefix / f"reference/{page_name}/index.html"
            ir_html = _read(ir_page, failures)
            if ir_html and 'class="doc doc-object' not in ir_html:
                failures.append(f"{ir_page.relative_to(ROOT)} contains no generated API object")

    generated_mathjax = _read(SITE / "javascripts/mathjax.js", failures)
    if generated_mathjax and "MathJax.typesetPromise()" not in generated_mathjax:
        failures.append("generated MathJax configuration does not typeset page content")

    if failures:
        for failure in failures:
            print(f"code-docs: {failure}")
        return 1
    print(f"code-docs: rendered {len(PASSES)} pass derivations and {len(IR_PAGES)} IR APIs in 2 locales")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

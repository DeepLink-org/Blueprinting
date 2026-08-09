from pathlib import Path
from runpy import run_path

import pytest

CHECKER = run_path(str(Path(__file__).parents[2] / "scripts" / "check_docs_i18n.py"))
markdown_body = CHECKER["markdown_body"]
validate_pair = CHECKER["validate_pair"]


def test_markdown_body_preserves_plain_markdown() -> None:
    source = "# Title\n\nBody\n"

    assert markdown_body(source) == source


def test_markdown_body_removes_yaml_front_matter() -> None:
    source = "---\ntemplate: home.html\n---\n\n# Title\n\nBody\n"

    assert markdown_body(source) == "# Title\n\nBody\n"


def test_markdown_body_rejects_unterminated_front_matter() -> None:
    with pytest.raises(ValueError, match="unterminated YAML front matter"):
        markdown_body("---\ntemplate: home.html\n# Title\n")


def test_validate_pair_compares_markdown_after_front_matter(tmp_path: Path) -> None:
    sources = {}
    for locale, title in (("en", "Title"), ("zh", "标题")):
        source = tmp_path / f"index.{locale}.md"
        source.write_text(
            f"---\ntemplate: home.html\n---\n\n# {title}\n\n## Section\n",
            encoding="utf-8",
        )
        sources[locale] = source

    assert list(validate_pair(Path("index.md"), sources)) == []

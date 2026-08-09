"""Validate the paired-source contract for Blueprinting documentation.

MkDocs navigation and Markdown links use language-neutral paths such as
``design/index.md``. The static i18n plugin resolves those paths to colocated
``*.en.md`` and ``*.zh.md`` sources. This check prevents either language from
silently drifting out of the documentation architecture.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


LOCALES: Tuple[str, ...] = ("en", "zh")
LOCALIZED_NAME = re.compile(r"^(?P<stem>.+)\.(?P<locale>en|zh)\.md$")
LOCALE_LINK = re.compile(r"\[[^]]+\]\((?!https?://|mailto:|#)[^)]+\.(?:en|zh)\.md(?:#[^)]+)?\)")
HEADING = re.compile(r"^(#{1,6})\s+", re.MULTILINE)


def localized_sources(docs_dir: Path) -> Dict[Path, Dict[str, Path]]:
    pairs: Dict[Path, Dict[str, Path]] = {}
    for source in sorted(docs_dir.rglob("*.md")):
        relative = source.relative_to(docs_dir)
        match = LOCALIZED_NAME.match(source.name)
        if match is None:
            raise ValueError(
                f"documentation source must use a locale suffix: {relative} "
                f"(expected *.en.md or *.zh.md)"
            )
        canonical = relative.with_name(f"{match.group('stem')}.md")
        locale = match.group("locale")
        pairs.setdefault(canonical, {})[locale] = source
    return pairs


def heading_shape(text: str) -> List[int]:
    return [len(match.group(1)) for match in HEADING.finditer(text)]


def validate_pair(canonical: Path, sources: Dict[str, Path]) -> Iterable[str]:
    missing = [locale for locale in LOCALES if locale not in sources]
    if missing:
        yield f"{canonical}: missing locale source(s): {', '.join(missing)}"
        return

    texts = {locale: sources[locale].read_text(encoding="utf-8") for locale in LOCALES}
    for locale, text in texts.items():
        if not text.startswith("# "):
            yield f"{sources[locale]}: page must start with one H1 heading"
        match = LOCALE_LINK.search(text)
        if match is not None:
            yield (
                f"{sources[locale]}: link uses a locale-suffixed source; "
                f"use the canonical *.md path instead: {match.group(0)}"
            )

    en_shape = heading_shape(texts["en"])
    zh_shape = heading_shape(texts["zh"])
    if en_shape != zh_shape:
        yield (
            f"{canonical}: heading hierarchy differs between translations "
            f"(en={en_shape}, zh={zh_shape})"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    args = parser.parse_args()

    try:
        pairs = localized_sources(args.docs_dir)
    except ValueError as error:
        print(f"docs-i18n: {error}")
        return 1

    errors = [
        error
        for canonical, sources in pairs.items()
        for error in validate_pair(canonical, sources)
    ]
    if errors:
        for error in errors:
            print(f"docs-i18n: {error}")
        return 1

    print(f"docs-i18n: validated {len(pairs)} bilingual page pairs ({', '.join(LOCALES)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

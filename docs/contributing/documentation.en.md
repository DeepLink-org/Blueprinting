# Documentation Guide

Blueprinting documentation is a versioned engineering product. It uses one information architecture, colocated English/Chinese source pairs, strict builds, and explicit capability status so design intent cannot silently drift away from implementation reality.

## Site architecture

The documentation is organized by reader intent:

```text
docs/
├── index.{en,zh}.md                 orientation
├── exploration/                     product thesis, design space, workflow
├── modeling/                        hardware and workload models
├── design/                          formal analysis foundations
│   ├── ir/                          representation contracts
│   ├── passes/                      transformation contracts
│   └── performance/                 evidence and simulation contracts
├── experiments/                     reproducible validation reports
├── project/                         status, roadmap, decisions
├── contributing/                    maintenance guides
├── overrides/home.html              bilingual product landing page
└── assets/
    ├── architecture/                shared technical diagrams
    └── stylesheets/site.css         site shell and landing-page visual system
```

Top-level narrative starts with the hardware question, candidate blueprints, models, simulation, and decision outputs. Formal-analysis reference pages then explain the models, derivations, verification obligations, and automated analyses that make those comparisons reproducible. Experiment pages report evidence, while project pages distinguish current state from accepted future design.

## Homepage and site shell

The homepage uses one locale-aware Material override rather than duplicating a large English and Chinese HTML surface. `index.{en,zh}.md` retains the searchable Markdown orientation and selects `home.html` through front matter; the override renders product-specific navigation for the active locale. Shared styling lives in `assets/stylesheets/site.css` and applies a restrained Blueprinting visual system to both the landing page and ordinary reference pages.

The landing page starts with the hardware decision, exploration loop, evidence ladder, capability status, and reader paths. It must not turn the product into a generic compiler, simulator, or dashboard. Product claims on the landing page remain subordinate to the capability matrix in [implementation status](../project/status.md).

## Bilingual source contract

The site uses `mkdocs-static-i18n` with its suffix structure:

- English is the default locale and is served at `/`;
- Simplified Chinese is served at `/zh/`;
- translations are colocated as `page.en.md` and `page.zh.md`;
- default-locale fallback is enabled only to reuse language-neutral static assets; the mandatory repository check requires every Markdown page in both languages, so prose cannot silently fall back;
- Material's language selector links the corresponding page in the other locale;
- search is rebuilt for each locale.

This follows the plugin's [suffix-structure guidance](https://ultrabug.github.io/mkdocs-static-i18n/setup/choosing-the-structure/) and [Material integration](https://ultrabug.github.io/mkdocs-static-i18n/setup/setting-up-material/). Do not add hand-written English/Chinese switch links inside pages.

## Canonical navigation and links

`mkdocs.yml` and all internal Markdown links use the language-neutral canonical path:

```markdown
[Derivation and verification model](../design/synthesis-model.md)
```

Never link to `synthesis-model.en.md`, `synthesis-model.zh.md`, or a generated `/zh/` URL. The i18n plugin resolves the canonical path for the active locale and keeps the language selector aligned.

External links use ordinary absolute HTTPS URLs. Shared SVGs use relative paths. A language-specific asset should use the same `.en`/`.zh` pairing convention and be justified; diagrams should prefer language-neutral labels where practical.

## Adding or moving a page

To add a page:

1. choose the owning section by reader intent;
2. create both `name.en.md` and `name.zh.md` in the same directory;
3. give both pages one H1 and the same heading hierarchy;
4. add one canonical `name.md` entry to `mkdocs.yml`;
5. add the navigation label and its Chinese `nav_translations` entry;
6. use only canonical paths in internal links;
7. run the documentation checks before committing.

When moving a page, update both source files, navigation, inbound links, and any public redirect policy together. Do not leave a second copy as an unofficial translation.

## Writing contracts

Product design pages start with the hardware architecture decision and explain design variables, workload coverage, fidelity, outputs, and claim boundaries. Engineering pages then explain invariants, data ownership, algorithms, failure behavior, observability, status, and implementation mapping. They separate:

- semantic fact from performance evidence;
- accepted target architecture from current implementation;
- canonical IR from derived view and published artifact;
- verifier obligations from profiler comparisons;
- measured result from inference or hypothesis.

Compiler must not be presented as a system component or top-level product definition. The product method is formal modeling, derivation, verification, and automated analysis. IR, lowering, and pass terminology is appropriate only where a page explains the borrowed implementation techniques behind workload mapping, semantic conservation, target binding, or current source identifiers.

Use **MUST/MUST NOT/SHOULD/SHOULD NOT/MAY** only for stable normative requirements. Chinese pages use their agreed equivalents. Code identifiers stay in their canonical spelling; prose should be natural in each language rather than mechanically word-for-word.

## Translation synchronization

The English and Chinese pages are peers in structure and technical meaning. A change is incomplete until both are updated. Tables, status labels, equations, code, links, figure identity, and claim boundaries must agree; natural-language examples may be localized.

`scripts/check_docs_i18n.py` enforces complete pairs, one H1, matching heading shapes, and canonical link usage. This gate is also what makes shared-asset fallback safe for prose. It cannot prove semantic equivalence, so reviewers still compare claims and status manually.

## Status and decision updates

A document that describes future architecture marks it **Planned** or **Contract Only** near the affected section. Only repository code plus proportionate tests may justify **Implemented**. The [implementation status](../project/status.md) is updated in the same change that connects or removes a capability.

Changes to schema ownership, lowering gates, plugin protocols, evidence semantics, serialized identity, or compatibility promises require an ADR as described in [decisions](../project/decisions.md). A roadmap item is not proof of implementation.

## Build and review

Install and validate with:

```bash
uv sync --locked --no-dev --extra docs
uv run --no-dev --extra docs python scripts/check_docs_i18n.py
uv run --no-dev --extra docs mkdocs build --strict
```

Contributors who also run the formal-analysis test suite may omit `--no-dev`; the default development dependency group includes the test and legacy-workbench dependencies.

For focused local preview:

```bash
BUILD_ONLY_LOCALE=en uv run mkdocs serve
BUILD_ONLY_LOCALE=zh uv run mkdocs serve
```

Before review, inspect both locale routes, page-to-page language switching, navigation, search, tables, code blocks, and SVG rendering. `navigation.instant` remains disabled because the i18n plugin documents it as incompatible with language reconfiguration.

## Continuous delivery

`.github/workflows/docs.yml` runs the bilingual contract check and strict site build for documentation pull requests and `main`. Pull requests retain the exact `site/` output as a versioned preview artifact. Successful `main` runs package the same output as a GitHub Pages artifact and deploy it through the protected `github-pages` environment to [deeplink-org.github.io/Blueprinting](https://deeplink-org.github.io/Blueprinting/). The Markdown under `docs/` remains the only documentation source tree; deployment artifacts are generated, immutable outputs rather than another editable source.

## Review checklist

- Both locale sources exist and have matching structure.
- Claims distinguish Implemented, Contract Only, and Planned.
- Canonical links work in both generated locale trees.
- New terms are defined or use existing project vocabulary.
- Figures have meaningful alt text and remain legible in light/dark contexts.
- Reproducible results identify commands, inputs, revisions, and claim boundaries.
- Design changes map to source/tests or explicitly state that no implementation exists.
- `check_docs_i18n.py` and `mkdocs build --strict` pass.
- The landing page and ordinary documentation pages remain usable in both color schemes and at narrow widths.

Documentation debt is handled like engineering debt: make the ownership boundary explicit, add a gate that detects regression, and remove the superseded source instead of maintaining ambiguous duplicates.

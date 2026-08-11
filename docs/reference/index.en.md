# Code documentation

This section is generated from the canonical Python source with
`mkdocstrings`. It is the discoverable API companion to the semantic design
documents, not a second definition of IR meaning.

## Reading order

1. Choose the owning representation stage under **Canonical IR API**.
2. Read **Pass Formulae and API** for every committed cross-stage derivation.
3. Use **Derivation Infrastructure API** for the decorators, transaction
   runner, lineage gate, and deterministic replay contract.

Every public canonical pass is documented from its class docstring. The same
docstring carries its equations, derivation assumptions, research references,
and explicit non-claims, so source review and rendered documentation cannot
silently describe different algorithms.

## Authority boundary

- `ir.py` is authoritative for immutable schema and structural invariants.
- `passes.py` is authoritative for public pass contracts.
- dialect `*_derivation.py` modules own pure domain derivations.
- rendered pages explain those sources and link to them; they do not create a
  parallel schema or estimator.

Math is rendered by Arithmatex and MathJax. API objects and source listings are
collected by the Python handler directly from `src/` during `mkdocs build`.


# ADR-0001: Name the Formal Derivation Package Synthesizer

- Date: 2026-08-09
- Status: Accepted; codec-identity clauses superseded by ADR-0004
- Scope: Python package identity, public symbols, report vocabulary, and one serialized field name

## Context

Blueprinting explores hardware architectures by turning workload semantics and strategy choices into progressively more specific, verified plans. The implementation borrows IR, lowering, pass, and verifier techniques from compiler engineering, but the product is not a general-purpose compiler and has no conceptual Compiler component.

The historical `blueprinting.compiler` package contradicted that boundary. It owned canonical representations, bindings, derivation transactions, model frontends, and validation experiments, while evidence-backed analysis had already become the sibling `blueprinting.analysis` package. Keeping the old name would continue to make an implementation toolbox look like the product architecture.

Here, **synthesis** means formal plan synthesis: deriving a more specific, verified representation from explicit inputs and obligations. It does not mean RTL synthesis, hardware implementation, code generation, or Blueprinting's product identity.

## Decision drivers

- Make package ownership match the formal modeling and verified-derivation architecture.
- Keep analysis and performance evidence visibly separate from canonical state transformation.
- Remove ambiguous public vocabulary before target plugins and external consumers depend on it.
- Preserve stable serialized identities wherever the semantic contract has not changed.
- Prefer one decisive migration over a long-lived parallel API hierarchy.

## Considered alternatives

**Keep `compiler`.** This minimizes import churn but preserves the product-framing error and makes future module boundaries harder to explain.

**Use `formal_analysis`.** This overlaps the existing `blueprinting.analysis` package and blurs authoritative transformations with rebuildable analyses.

**Use `planner`.** Planning is only part of the path; the package also owns semantic frontends, canonical representations, verification, lineage, and machine-level realization contracts.

**Rename to `synthesizer` with a compatibility shim.** A shim lowers immediate migration cost but keeps two discoverable public hierarchies, weakens boundary tests, and encourages indefinite use of the deprecated identity.

## Decision

Rename `blueprinting.compiler` to `blueprinting.synthesizer` as a hard Python API cut. Do not provide a `blueprinting.compiler` import shim. `blueprinting.analysis` remains a sibling package and may consume canonical synthesis contracts without mutating them.

Use synthesis vocabulary for public implementation symbols:

| Previous | Current |
|---|---|
| `CompilationSession` | `SynthesisSession` |
| `CompilerError` | `SynthesisError` |
| `CompilerPass` | `DerivationPass` |
| `compilation_session_for()` | `synthesis_session_for()` |
| `inference_compilation_session_for()` | `inference_synthesis_session_for()` |
| `compile_transformer_block()` | `derive_transformer_block()` |
| `compile_transformer_inference_block()` | `derive_transformer_inference_block()` |

Keep `Pass`, `PassManager`, `lowering`, and `IR` where they describe precise borrowed mechanisms. External APIs such as Calculon's `model.compile()` and Python's `compile()` retain their names.

The original package migration preserved the then-current codec tags and session digest domain. ADR-0004 supersedes that compatibility choice: current artifacts use domain-owned semantic identities and a `synthesis-session` digest domain.

Rename the former target ABI field to `target_abi` because the ABI belongs to a bound target. The initial migration accepted the previous spelling as a decoder alias; ADR-0004 removes that runtime alias, so current payloads use only `target_abi`.

Report vocabulary distinguishes facts from predictions: exact work uses `derived_*`, timing and memory predictions use `estimated_*`, and the Blueprinting side of comparisons uses `blueprinting`. Calculon and Vidur report schemas advance to v2 because their emitted field names change.

## Consequences

- Imports from `blueprinting.compiler` fail immediately and downstream Python callers must migrate atomically.
- Source navigation now exposes the intended split: `synthesizer` owns canonical derivation; `analysis` owns rebuildable evaluation and evidence resolution.
- The initial package migration kept its contemporary snapshots readable; ADR-0004 later establishes an explicit wire-format boundary.
- Current target-profile payloads use only `target_abi`; obsolete field spellings are not accepted.
- Pickle/module-path compatibility is not provided. Canonical JSON is the supported persistence boundary.
- Existing v1 experiment report consumers must migrate to the v2 field names.

## Migration

1. Replace Python imports from `blueprinting.compiler` with `blueprinting.synthesizer`.
2. Replace the public symbols using the table above.
3. Use `target_abi=` constructor arguments and `.target_abi` attribute reads.
4. Update Calculon consumers from `compiled` to `blueprinting`, `compiled_breakdown_seconds` to `estimated_breakdown_seconds`, and `compiled_explicit_operations` to `derived_explicit_operations`.
5. Update Vidur consumers from `compiled_*` to the corresponding `estimated_*` fields.
6. Regenerate v2 experiment artifacts.

## Validation

- A package-boundary test requires `blueprinting.synthesizer` to exist and `blueprinting.compiler` to be absent.
- Public API tests require the new symbols and reject legacy re-exports.
- Canonical round-trip tests verify that target profiles encode only `target_abi`.
- Golden target-neutral IR snapshots and baseline regression digests guard the active semantic wire identities.
- Calculon and Vidur tests guard the v2 report vocabulary and numerical equivalence.
- Ruff, the full pytest suite, bilingual documentation parity, and strict MkDocs builds are release gates.

## Status

Accepted on 2026-08-09. ADR-0004 supersedes this ADR's codec-tag and field-alias compatibility clauses while retaining the `synthesizer` package decision.

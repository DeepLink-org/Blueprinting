# ADR-0003: Represent IR Debugging as a Derived Derivation Trace

- Date: 2026-08-10
- Status: Accepted
- Scope: IR visualization, lineage mapping, debug bundles, and derived overlays

## Context

Existing checkpoints preserve immutable IRs, digests, pass records, and lineage, but the workbench exposed only stage summaries and JSON. Inferring correspondence from names in the UI, or adding layout and cost to canonical IRs, would create a second representation semantics and weaken late binding.

## Decision

Introduce an application-owned, invalidatable, rebuildable `DerivationTrace`. Five adapters register by canonical schema. Layer graphs project typed fields only; adjacent-stage correspondence consumes target lineage only. Audit rules produce non-blocking diagnostics, while verifiers and observers retain transaction-acceptance authority.

Cost, timing, and observations attach through detachable overlays carrying provider revisions. `blueprinting.derivation-debug-bundle.v0` stores one run's canonical snapshots and derived metadata and revalidates them on import. It is neither a sixth IR nor a replacement for `TimelineBundle`.

Current production runs produce only the first three stages. Adapters and valid-import support for the final two must not be described as completed Concrete or Machine producers.

## Rejected alternatives

- Add UI coordinates, colors, collapse state, or cost fields to the five IRs; this contaminates canonical semantics and digests.
- Infer correspondence from display names, array positions, or operation names; this cannot represent decomposition, fusion, or generated entities.
- Make visualization audits blocking by default; the rules are not stable proof obligations for every dialect.
- Deliver breakpoints, snapshot mutation, and cross-run diff in the first version; these require separate execution-state, matching, and safety contracts.

## Consequences and validation

Application reports gain traces and analysis-report schemas advance to v2; legacy `stages` remain as a compatibility presentation. The debug bundle has an independent version and a 50 MiB limit. Tests cover all five adapters, lineage cardinality and mismatch, deterministic round trips, tamper rejection, large-graph grouping, and NiceGUI interactions. Canonical golden digests must remain unchanged.

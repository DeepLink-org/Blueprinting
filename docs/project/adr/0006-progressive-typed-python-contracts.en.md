# ADR-0006: Progressive Typed Python Contracts

Status: Accepted and implemented foundation

## Context

Python annotations documented many local types but did not by themselves remove enum-plus-optional invalid states, make expected failure explicit, guarantee closed-match exhaustiveness, or prevent an unverified pass result from being committed. Requiring mypy at runtime would make the formal model depend on a development tool, while a custom plugin would duplicate runtime semantics behind a checker-specific API.

## Decision drivers

- One declaration source must serve runtime verification and optional static analysis.
- Base installations must check formal contracts without mypy.
- Expected validation, decoding, resolution, and derivation failures must be values rather than hidden exception flow.
- Closed core semantics and open plugin/provider semantics must use different extension mechanisms.
- Every changed committed derivation must carry executable evidence.

## Decision

- Introduce domain-free `Result`, `Checked`, and immutable diagnostics in `blueprinting.schema`.
- Make canonical `verify`/`from_json`, pass-manager `run`, and cost resolver `resolve` checked APIs; retain only explicitly named `require_*` exception adapters.
- Close core ADTs with an explicit union alias and `seal_adt`; make roots abstract and reject late constructors and unregistered subclasses. Consumers use explicit-union pattern matches plus `assert_never`; no decorator metadata claims to prove function-body coverage.
- Compile complete record/enum schema shapes plus loaded codec, ADT, and pass declarations with a mypy-independent `ContractCompiler` and CI script.
- Separate authoring APIs from ordinary consumption: schema decorators are exposed only from `blueprinting.schema.authoring`, pass decorators only from `blueprinting.synthesizer.passes.authoring`, and ordinary package roots do not forward decorators.
- Remove successful `UNVERIFIED` pass transitions. Cross-stage contracts require typed relations with independent invariants, a separately reported complete normalizer, output verification, and a stable revision.
- Keep target and provider extension points open through `Protocol`/registries.
- Publish `py.typed` and run standard mypy in a separate optional `typing` dependency group and CI job. Do not introduce a custom mypy plugin.

## Domain/schema migration

The initial migration replaces workload mode plus optional phase, duplicate flat Transformer binding facts, portable task kind, cost-support status records, and collective kind plus conditional reduction/root fields with constructor-specific ADTs and structured products. The owning canonical roots remain in unpublished schema epoch `0.0.0`; no production migration edge or decoder alias is created.

`PlanTask.kind` remains a derived presentation projection of `PlanTask.body`, not serialized truth. Open target/provider sets are not sealed by the core package.

## Consequences

Code can use structural pattern matching and monadic `map`/`and_then` composition while staying ordinary Python. Runtime checks work in the base environment; installing mypy adds source-level flow and exhaustiveness analysis without changing execution.

The contract compiler validates declarations, not scientific truth or arbitrary function bodies. Formula correctness still requires independent executable invariants, canonical construction checks, negative tests, evidence provenance, and experiments. Third-party target declarations cannot yet contribute to the compiled manifest because the general target-plugin registry is not implemented.

## Validation

Runtime tests cover result composition, diagnostic accumulation, deterministic manifest compilation, ADT closure/exact membership, authoring-surface boundaries, canonical checked verification/decoding, checked pass failures, cost support matching, and rejection of changed transitions without evidence. CI runs the runtime compiler in the base test job, mypy in an independent optional-typing job, full pytest/Ruff gates, bilingual strict docs, and a wheel check requiring `py.typed`.

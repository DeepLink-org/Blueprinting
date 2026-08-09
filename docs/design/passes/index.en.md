# Analysis and Transformation Infrastructure

A Blueprinting `Pass` is the current implementation unit for a verified transaction over immutable derivation state. The mechanism is borrowed from compiler engineering; its architectural role is to make analysis dependencies, transformation authority, proof obligations, profiler checkpoints, and cache effects explicit.

![Verified pass transaction](../../assets/architecture/pass-transaction.svg)

## Transformation contract

Every pass declares:

```text
pass_id and revision
input IR type and accepted schema range
output IR type and produced schema version
required bindings
required analyses
preserved analyses
produced analyses
mutation model
verification policy
determinism and seed usage
```

Pipeline composition follows these contracts rather than `isinstance` checks against concrete pass classes.

## Transaction sequence

`PassManager` performs:

```text
check input type/schema
  -> require declared bindings and analyses
  -> verify input snapshot
  -> execute immutable or isolated mutation
  -> check output type/schema and input immutability
  -> verify output and parent lineage
  -> create PassRecord and PassCheckpoint
  -> invoke synchronous observers
  -> atomically preserve/publish analyses
  -> commit the output as the next snapshot
```

An exception, verifier failure, observer rejection, or undeclared analysis product aborts the transition. The previous snapshot and analysis store remain visible and unchanged.

## Analysis store

Analyses are addressed by:

```text
(IR digest, AnalysisKey, SynthesisSession fingerprint)
```

A pass lists every required, preserved, and produced analysis. Preserved products are copied only to the verified output digest. Undeclared products and noncanonical addresses fail the transaction.

Derivation analysis and target evidence are separate stores. The former is keyed to representation snapshots and typed sessions; the latter is keyed to estimate requests and provider revisions.

## Checkpoints and observers

`PassCheckpoint` exposes the immutable output, schema, digest, pass record, session identity, and lineage before analysis publication. Synchronous observers can run stage-specific verification, profiler comparisons, or conservation checks.

Observers are read-only. They cannot rewrite a representation or publish analyses directly. A rejected checkpoint aborts the whole transition, so validation does not leave a partially accepted derivation.

## Measurement semantics

`PassRecord.duration_ns` measures host-side transformation wall time. It is analysis-overhead provenance and never a predicted workload duration. Workload cost lives in evidence-backed views.

## Determinism

A pass declares whether it is deterministic and how it uses a seed. A deterministic pass over the same input digest, session fingerprint, pass revision, and required analyses must emit the same output digest or diagnostic.

Search passes may be seeded and budgeted. Candidate order, pruning, and rejection reasons remain provenance so a search result can be replayed.

## Analysis and transformation documentation template

Every production analysis or transformation design must include:

1. problem and decision authority;
2. input assumptions and required analyses;
3. transformation algorithm or pseudocode;
4. before/after IR example;
5. preserved semantics and output invariants;
6. complexity and search budget;
7. diagnostics and fallback policy;
8. checkpoint and profiler integration;
9. implementation and test mapping.

## Current implementation

The repository implements `SchemaRange`, `PassContract`, `PassPipeline`, `PassManager`, content-addressed `AnalysisStore`, pass records, checkpoints, and observers in `src/blueprinting/synthesizer/passes/base.py`. Contract and failure behavior are covered by `tests/synthesizer/test_pass_manager.py`.

# ADR-0005: Algebraic Expression and Command Schemas

Status: Accepted and implemented

## Context

`ScalarExpr(op, args)` admitted invalid arities and forced every interpreter to rediscover an enum-dependent product invariant. `ConcreteCommand(kind, queue?, implementation?, wait_tokens?, signal_tokens?)` similarly admitted combinations that were meaningful only for some command kinds. Large verifiers rejected these states after construction, so annotations did not describe the legal canonical value space.

At the same time, pass contracts listed preservation properties as strings. A discovered lineage relation was counted as verified even when its callback checked only a subset of the declared properties, and callbacks could not inspect the complete source-to-target mapping needed to prove dependency topology.

## Decision

- `ScalarExpr` is a closed ADT with `Add`, `Subtract`, `Multiply`, `Divide`, `CeilDivide`, `Maximum`, and `Minimum` constructors. Constructor fields encode arity.
- `ConcreteCommand` is a graph envelope containing one command-body ADT. Executable bodies own required implementations; queue-capable bodies own their optional queue. A separate synchronization ADT represents none, wait, signal, or wait-and-signal clauses.
- A cross-stage pass declares typed lineage relations, an independent executable invariant for each relation, and one pure canonical normalizer. Transition verification resolves the complete lineage graph, re-evaluates the normalizer for canonical implementation conformance, and then evaluates relation invariants as separate semantic evidence.
- Cross-stage passes without relations or an executable normal form fail the commit gate. Any changed transition without complete executable evidence fails; only an unchanged same-stage transition may succeed as `structural_only`. A successful `unverified` state is not representable.

## Schema epoch

These constructors define the initial canonical epoch rather than a migration from unpublished prototypes. All five IR roots therefore remain at `0.0.0`; nested record and ADT identities are semantic names without independent version suffixes. The production migration registry contains no historical edges. No decoder aliases or legacy constructor classes are retained.

## Consequences

Canonical constructors now describe a substantially smaller legal state space, and closed matches can use `assert_never` for static exhaustiveness. Generic consumers may use derived `kind`, `queue`, implementation, and token properties, but those properties are projections rather than serialized discriminators.

Future graduated compatibility changes must advance the owning root schema and register an explicit migration. Component-local counters must not be introduced as a substitute for that boundary.

## Validation

Positive and round-trip tests cover every active derivation chain. Negative mutation tests change specialized shape, semantic payload, generated-buffer capacity, dependency topology, and implementation identity while retaining otherwise valid structure. Normal-form checks reject canonical construction deviations; an additional counterexample deliberately lets a normalizer accept an invalid implementation and confirms that the independent relation invariant still rejects it. Generic migration tests cover deterministic chaining, ambiguity, no-op loading, and tamper detection without claiming a production history.

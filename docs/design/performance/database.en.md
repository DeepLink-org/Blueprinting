# Performance Database

The performance database is a revisioned evidence store behind a normalized query protocol. It answers a precise question—how an architecture component or legal implementation is expected to behave in a declared context—without hiding architecture choices or calibration knobs inside a lookup table.

!!! note "Design status"
    The repository currently loads `HardwareProfile` directly. The request/result/store design on this page is the accepted migration target.

## Request contract

An `EstimateRequest` identifies all dimensions that may materially affect a result:

```text
subject identity
  portable task + legal implementation revision
workload context
  operation, shape, dtype, layout, operations, bytes, message volume
architecture context
  blueprint, component/engine, memory space, capability and implementation revisions
deployment context
  device class, topology/link class, participant count, placement, runtime/library revision
execution context
  concurrency class, queue/resource occupancy, power mode
evidence policy
  admissible providers, freshness, uncertainty and fallback limits
```

Optional fields are explicit unknowns, not omitted cache-key dimensions. Providers declare which fields they require and the domain over which their answer is valid.

## Result contract

An `EstimateResult` contains more than a scalar:

```text
metrics             latency, energy, bandwidth, utilization, counters
uncertainty         interval/distribution/confidence and sample count
validity            exact domain, interpolation, extrapolation distance
provenance          provider, raw record IDs, source and calibration revisions
method              measured, simulated, analytical, calibrated, fallback
diagnostics         missing context, assumptions, rejected alternatives
```

Normalization does not erase provider detail. Provider-specific payloads may be attached as typed extensions, while the fields needed by planners and audits remain portable.

## Storage model

The database separates immutable raw evidence from derived indexes and calibrated models:

| Collection | Contents | Update rule |
|---|---|---|
| Raw records | benchmark samples, simulator runs, profiler events | append-only |
| Environment manifests | device, driver, firmware, runtime, clocks, topology, protocol | content-addressed |
| Aggregates | cleaned samples, distributions, confidence intervals | rebuildable |
| Calibration revisions | fitted curves/models plus training set digests | new immutable revision |
| Query indexes | lookup accelerators for normalized request dimensions | rebuildable |

Every raw sample records units, warm-up, repetition count, synchronization method, clock/power state, and measurement errors when available. A bare `(op_name, latency)` pair is not sufficient evidence.

## Provider protocol

A provider exposes four operations conceptually:

```python
class EstimateProvider(Protocol):
    @property
    def revision(self) -> str: ...

    def supports(self, request: EstimateRequest) -> Support: ...
    def estimate(self, request: EstimateRequest) -> EstimateResult: ...
    def explain(self, result: EstimateResult) -> EvidenceTrace: ...
```

`supports()` reports domain coverage and required missing fields before expensive evaluation. `estimate()` is deterministic for a request and provider revision unless the result explicitly records a seed and stochastic protocol.

## Resolution policy

`CostResolver` chooses evidence; it does not stack correction factors. A typical policy may prefer:

1. compatible direct measurements;
2. compatible simulator results validated for the target revision;
3. calibrated interpolation inside a measured domain;
4. analytical estimation;
5. explicit conservative fallback.

Conflicting sources remain inspectable. The resolver records why one source won and why others were rejected. Blending is allowed only through a named, versioned model with declared inputs—not an anonymous product of coefficients.

## Cache identity and reproducibility

The result cache key includes the canonical request digest, provider revision, calibration revision, resolver-policy revision, and deterministic seed when relevant. Target, deployment, runtime, topology, and concurrency context are therefore part of identity rather than ambient state.

A published plan carries an evidence snapshot fingerprint. Rebuilding under that snapshot either reproduces the same results or reports that required evidence is no longer available; it never silently upgrades to the newest database state.

## Calibration without case fitting

Calibration learns target-wide or implementation-family response behavior from observations: operation-size efficiency, transfer-size efficiency, network efficiency, launch overhead, or contention models. Training data and validation splits are revisioned.

Forbidden inputs include a benchmark case ID, comparison-oracle total time, or a per-model correction factor whose only purpose is matching a table. Those variables do not explain a causal target behavior and cannot generalize to a new plan.

## Migration from HardwareProfile

The existing `HardwareProfile` already supplies useful versioned curves for matrix/vector throughput, memory transfer, and collectives. Migration should preserve its behavior behind providers:

1. convert portable tasks into normalized requests;
2. wrap the current profile as an analytical/system-evidence provider;
3. reproduce the current Calculon experiment through the resolver;
4. add a raw measurement provider and evidence manifests;
5. replace direct estimator/profile coupling only after equivalence tests pass.

This staged adapter keeps the validated workload analysis intact while making provenance, uncertainty, and future hardware simulators first-class.

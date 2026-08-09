# Simulation and Calibration

Simulation is where a candidate hardware blueprint becomes an executable hypothesis. It evaluates a verified architecture-bound resource plan under declared cost and contention models rather than inventing a second schedule. Calibration compares predicted and observed events and creates a new evidence revision without mutating the blueprint or plan that produced them.

!!! warning "Design status"
    Block/iteration analytical estimates are implemented. The discrete-event simulator, observation normalizer, and closed-loop calibration service described below are **Planned**.

## Simulation input

The simulator consumes:

- a verified `ConcretePlanIR` envelope: common coordination core plus typed target extensions;
- physical resources, queues, memory spaces, and topology from the deployment;
- a complete `CostedTaskView` for the selected implementations;
- simulator model and evidence revision identities;
- an optional deterministic seed for explicitly stochastic resources.

It never consumes `ModelIR` and reconstructs placement or overlap on its own. Portable and model IDs are available only through lineage for aggregation and diagnosis.

## Discrete-event state

The minimal state contains:

```text
logical clock
ready/running/completed command sets
per-queue order and availability
per-engine and per-link capacity
signal/event/barrier state
buffer allocation and lifetime state
resource contention state
ordered event queue
```

A command becomes ready only when its explicit dependencies, ordering predecessor, synchronization conditions, and required resources are satisfied. Predicted start timestamps never make a command ready; target-enforced cycles or slots are interpreted by typed target-extension rules.

## Event algorithm

A deterministic baseline algorithm is:

1. enqueue commands whose semantic readiness conditions hold;
2. choose ready commands by a stable ordering and reserve resources;
3. obtain duration/resource demand from the selected cost view;
4. schedule completion and intermediate transfer events;
5. advance to the next event, release resources, update signals and buffers;
6. repeat until all commands complete or a diagnosed deadlock is reached.

Every event carries concrete command ID, resource ID, estimate provenance, and upstream lineage. The result includes intervals, critical path, resource utilization, memory high-water marks, queue delay, and uncertainty propagation.

## Network simulation

Communication is modeled from logical collective semantics plus concrete participants, routes, algorithms, and link resources. A simple provider may use latency-bandwidth curves and collective volume models. A detailed network simulator may model topology, routing, arbitration, congestion, and failures.

Both paths return normalized results for the same request. Network time is not inserted upstream as `message_bytes / nominal_bandwidth`, and a global overlap ratio is not used to approximate command concurrency.

## Timing projection versus simulation trace

`TimingProjection` is a rebuildable analysis over concrete commands: predicted intervals, slack, critical path, and uncertainty. `SimulationTraceIR` is an interchange trace with resource events and source correlation. Neither is a canonical execution program.

`TimelineBundle` references the concrete plan, projection or trace, evidence and policy fingerprints, and diagnostics as a publishable analysis/replay package. It cannot copy and mutate the command DAG.

A replay or runtime engine follows dependencies, ordering, and synchronization in `ConcretePlanIR` or its `MachineIR` lowering. A simulator may attach predicted timestamps; a runtime profiler attaches observed timestamps. Both correlate to the same command identity, while a runtime may perform bounded backpressure and failure handling allowed by the target contract.

## Observation normalization

Profiler adapters convert target-specific counters and events into an immutable `ObservationSet`:

```text
environment manifest and run identity
artifact, MachineIR, ConcretePlanIR and evidence fingerprints
command/instruction correlation
observed intervals and resource counters
missing, duplicated, or unmatched events
measurement protocol and uncertainty
```

Raw traces remain attached or referenced. Normalization never fabricates unmatched command IDs; missing correlation is itself a diagnostic and quality metric.

## Calibration revisions

Calibration compares estimate requests/results with compatible observations, stratifies residuals by causal dimensions, and produces a new immutable model revision. Examples include efficiency curves by operation size, launch overhead by runtime revision, or collective behavior by topology and participant count.

The previous evidence snapshot and every plan derived against it remain reproducible. Adopting the new revision is an explicit re-analysis and replanning choice. If changed estimates alter placement or scheduling, the result is a new `ConcretePlanIR` digest.

## Bounded refinement loop

Some interactions require iteration:

```text
candidate implementation/placement
  -> preliminary costs
  -> schedule and contention context
  -> context-aware re-estimation
  -> reschedule if material
```

Refinement must have a deterministic convergence rule, iteration budget, and oscillation diagnostic. The final plan records every iteration and selected evidence. An unbounded feedback loop is not a valid analysis or transformation transaction.

## Correspondence obligations

Simulation and execution may claim a correspondence level only if:

1. both originate from the same verified concrete command DAG;
2. every emitted/simulated operation maps to a concrete command or declared runtime support action;
3. dependencies, queue order, synchronization, and buffer contracts are preserved;
4. stripping timing from the trace does not erase execution semantics;
5. observed deviations are represented as evidence, not retroactive IR mutation.
6. the observation policy declares an exact-event, bounded-divergence, or partial-observation level.

These obligations are stronger than matching total latency: they test whether simulation and optional emission realize the same verified architecture-bound plan. They do not prove equality between predicted and observed timestamps.

## Validation ladder

The simulation system should mature through increasing fidelity:

1. deterministic virtual-target unit tests with hand-computable DAGs;
2. analytical consistency with the existing block/iteration estimator;
3. Calculon workload and schedule comparison;
4. replayable command-engine comparison;
5. real GPU profiler correlation;
6. simulation correspondence when an LPU architecture/simulator contract exists, followed by executable correspondence when its ABI and runtime exist.

At each rung, semantic conservation and command correlation are tested before aggregate latency accuracy.

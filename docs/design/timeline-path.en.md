# The Timeline Path: A Staged Product, Not an Architectural Shortcut

“Compiling a timeline” is a valuable near-term engineering path for Blueprinting, but the phrase can collapse three different concepts: execution correctness, performance prediction, and target-specific temporal control. This page fixes those boundaries and explains how the staged path evolves toward GPU, LPU, or other backends without changing the product goal of hardware architecture exploration.

!!! warning "Design and implementation status"
    This page defines an accepted direction. The production path currently reaches only `PortablePlanIR` and an analytical estimate; a `ConcretePlanIR` producer, discrete-event simulator, `TimelineBundle`, and executable backend are not implemented. The [status page](../project/status.md) is authoritative.

## Decision summary

Blueprinting will **not** restore the old public `GraphIR -> ScheduleIR -> TimelineIR` stack, and predicted absolute timestamps will not become execution semantics. The staged product is:

```text
verified PortablePlanIR
  + candidate architecture / target / deployment
  + planning evidence and policy
  -> verified ConcretePlanIR              authoritative coordination plan
  -> TimingProjection                     predicted intervals + uncertainty
  -> SimulationTraceIR                    simulated resource events
  -> TimelineBundle                       publishable analysis/replay bundle
```

`TimelineBundle` is a fully provenance-carrying product package, not a sixth canonical IR. It references rather than copies `ConcretePlanIR` and can be consumed by the workbench, simulators, comparison experiments, and replay prototypes.

## Five concepts that must not be conflated

| Concept | Semantics it owns | Semantics it does not own |
|---|---|---|
| `ConcretePlanIR` | Command dependencies, resource bindings, queue/order, synchronization, buffer planning, and a future typed target-schedule extension | Predicted wall-clock truth |
| `TimingProjection` | Predicted intervals, slack, critical path, and uncertainty under specified evidence, contention model, and policy | Correctness or permanent target facts |
| `SimulationTraceIR` | Events and counters from one simulator run, correlated to command lineage | An independent schedule or workload semantics |
| `TimelineBundle` | Concrete digest, projection/trace, evidence and policy fingerprints, diagnostics, and artifact manifest | A new canonical semantic layer |
| `MachineIR` | Backend-owned commands or instructions, ABI, sections, entry points, and target-verifier contract | Re-deciding high-level mapping or communication dependencies |

In conversation, “timeline” may refer to any of the last three. Schemas, APIs, papers, and tests must use the precise name.

## Predictive time and prescriptive time

### Predictive time

Most GPU, network, and system-level simulation time is predicted. It depends on the implementation, deployment, concurrency context, evidence revision, and simulator policy. New evidence may produce a new `TimingProjection`, but it cannot retroactively modify workload facts.

After predictive timestamps are removed, `ConcretePlanIR` should still contain enough information to determine command readiness, resource legality, and buffer safety. That is the condition for reusing a plan across cost models.

### Prescriptive time

Some targets may make issue cycles, slots, static phases, or time-triggered protocols part of program correctness. In that case time is not a generic prediction field; it is a typed scheduling constraint introduced **after target binding**:

```text
ConcretePlanIR target extension
  -> target verifier
  -> MachineIR issue/slot encoding
```

Such a constraint must define its clock domain, tolerance, backpressure, timeout and failure behavior, and ABI revision. It cannot leak into `PortablePlanIR` or acquire correctness meaning through free-form `attributes`.

This distinction lets an LPU exploit static dataflow or deterministic issue behavior without forcing GPU, CPU, or network targets to pretend they share the same temporal semantics.

## Why a timeline remains the right near-term deliverable

Although a timeline is not canonical execution truth, it still creates the first product-relevant architecture-bound vertical slice:

1. it turns abstract work into inspectable resource occupancy and event sequences;
2. it exposes critical paths, queue delay, overlap consequences, and memory high-water marks;
3. it lets performance databases, analytical models, network simulators, and hardware simulators provide replaceable evidence for the same commands;
4. it preserves stable lineage at every derivation stage so predicted events can correlate with profiler observations;
5. it validates the planning contract with a virtual target and replay before a real target ABI exists; and
6. it leaves an entry point for continued lowering from the same concrete plan into an LPU backend.

It is a staged hardware-exploration product, not a redefinition of the project as a timeline blueprinting.

## Stages and acceptance gates

| Stage | Deliverable | What it must establish | What it explicitly does not claim |
|---|---|---|---|
| A — Current baseline | `PortablePlanIR` + analytical estimate + optional dependency-projection Chrome Trace | Work/byte/collective conservation, stable lineage, evidence provenance; the projection export is explicitly marked `executable=false` | Architecture-bound scheduling, a `TimelineBundle`, or event simulation |
| B — Virtual timeline | Virtual-target `ConcretePlanIR` + `TimingProjection` + event trace | Resource legality, buffer safety, deterministic replay, agreement on hand-checkable DAGs | Real GPU/LPU performance or ABI executability |
| C — Multi-target simulation | Multi-blueprint binding + provider-backed timeline bundle | Shared workload digest, target-specific legality, uncertainty, explainable bottlenecks | One queue/dataflow model shared by all targets |
| D — Backend replay | Target plugin + replayable `MachineIR` | Command/instruction lineage, no emitter rescheduling, trace correlation | Ability to drive silicon |
| E — Executable LPU/GPU | ABI-specific artifact + runtime/profiler adapter | Target verification, failure behavior, observed-event correspondence | Equality between predicted and observed timestamps |

An artifact from a later stage becomes evidence only after the preceding acceptance gate holds; otherwise it is demonstration code.

## LPU is not a special-case branch

In Blueprinting, an LPU is first an architecture candidate and only later an executable target. A backend adds capability by maturity level:

| Maturity | Required contract | Work enabled |
|---|---|---|
| Architecture | Capability/resource hierarchy, constraints, canonical identity | Static feasibility and design-space comparison with other blueprints |
| Simulation | Legalizer, resource model, implementation catalog, cost or simulator provider | Architecture-bound timeline generation and bottleneck analysis |
| Replay | Typed target-schedule extension, MachineIR dialect, deterministic adapter | Validation of plan-to-backend-command correspondence |
| Executable | Stable ABI, emitter, loader/runtime, target verifier, profiler normalizer | Generation and validation of a real program artifact |

The first two levels do not need to wait for LPU hardware. Conversely, interest in LPU is not a reason to insert LPU opcodes, memory banks, or fixed timing into the frontend or portable plan.

## ConcretePlanIR extension boundary

The common layer should express only a coordination kernel that is stable across targets: identity, dependencies, resource claims, buffer references, synchronization, and lineage. Spatial dataflow, routes, issue slots, collective micro-protocols, and special memory movement belong in a namespaced **typed target extension** verified by the target plugin.

The current `ConcretePlanIR` v1 now has typed queue-order and slot/dataflow extensions with deterministic virtual reference binders. This closes the schema-level queue-centric versus queue-free test, but not production scheduling, resource occupancy, target-plugin legality, simulation, or emission. It remains an implementation scaffold rather than a frozen cross-target ABI.

## How each stage connects to observation

“Every stage can be checked” does not mean every stage reads a runtime profiler directly:

| Boundary | Primary checker | Correlatable observation |
|---|---|---|
| `ModelIR` | Type/effect/shape verifier and numerical reference | Framework operation or shape reference |
| `DistributedTaskIR` | Shard reconstruction, collective matching, volume conservation | Rank-level communication trace |
| `PortablePlanIR` | Work/byte/buffer/resource-requirement conservation | Backend-independent counters, when normalization is reliable |
| `ConcretePlanIR` | Placement/queue/synchronization/capacity/target legality | Command and resource events |
| `TimingProjection` | Critical-path/event consistency and uncertainty policy | Predicted-versus-observed intervals |
| `MachineIR` / artifact | ABI/target verifier and lineage coverage | Instruction, kernel, or packet profiler events |

A profiler adapter normalizes target-specific events into an immutable `ObservationSet`. An observation may invalidate a prediction and produce a new evidence revision, but it cannot modify a published IR.

## Shortcuts that remain forbidden

- aligning only total latency instead of events and commands;
- putting predicted timestamps into readiness or buffer-correctness semantics;
- letting the simulator and emitter independently reconstruct schedules;
- carrying LPU dataflow, route, or issue legality in untyped metadata;
- claiming a “zero-decision” runtime without defining backpressure, failure, timeout, and dynamic-shape policy;
- treating schema `1.0.0` as a public compatibility promise before producers and consumers are connected; and
- changing shared workload facts for one target's convenience.

## Minimum decisions before implementation

Before implementing the virtual timeline, the project must close:

1. the interface between the `ConcretePlanIR` common coordination core and typed target extensions;
2. independent identities for planning evidence and evaluation evidence;
3. minimum semantics for resource claims, routes, and concurrency occupancy;
4. the timeline-bundle schema, invalidation rules, and uncertainty representation;
5. simulator/runtime observation correspondence levels;
6. the executor contract for dynamic duration, backpressure, and failure; and
7. the graduation gate from experimental schemas to a public compatibility contract.

These are entry conditions for architecture-bound simulation, not issues to defer until an LPU emitter is built.

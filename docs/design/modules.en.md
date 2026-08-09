# Formal Analysis Module Architecture

Modules follow hardware-experiment ownership, not merely source-file convenience. Each module has typed inputs and outputs, owns a bounded decision class, publishes diagnostics and provenance, and cannot reach into downstream state.

## Exploration facade and derivation context

The future public facade should expose architecture exploration rather than an IR pipeline:

```python
experiment = ExplorationSession(
    workloads=WorkloadSuite(...),
    architecture_space=ArchitectureSpace(...),
    mappings=MappingSpace(...),
    objectives=(latency, energy, area),
    constraints=(memory_capacity, power_envelope),
    evidence_snapshot=EvidenceRevision(...),
    fidelity_policy=FidelityPolicy(...),
    seed=0,
)

result = blueprinting.explore(experiment)
result.pareto_blueprints
result.bottlenecks
result.sensitivity
```

Internally, each candidate creates an immutable typed derivation context for workload mapping, architecture binding, and analysis addressing. The current implementation names this object `SynthesisSession`; that class and its workload/strategy bindings are implemented, while `ExplorationSession` and the end-to-end product facade are planned. Global mutable configuration is forbidden because it would invalidate experiment reproducibility.

## Workload and system domain models

`blueprinting.workload` owns target-neutral model semantics, request scenarios, and logical mapping intent. A workload object cannot contain a chip name, peak rate, empirical latency, kernel identity, or physical placement. The current slice provides typed Transformer training and inference contracts.

`blueprinting.system` owns immutable chip-local compute engines, memory capacity/bandwidth, interconnect tiers, collective volume rules, and their imported evidence revision. `SystemProfile` is the current limited compute/memory/network adapter; it is not yet the hierarchical `ArchitectureBlueprint`, physical deployment, or target binding described by the product design. Cost policy remains in `analysis`: the system contract exposes peak and evidence-bearing facts but does not choose calibration mode.

These packages are authoritative domain inputs, not alternative IR hierarchies. Canonical derivation starts only when a synthesizer frontend imports a workload contract into `ModelIR`; a system profile remains outside canonical workload state and is consumed by explicit analysis or later target binding.

## Frontends

Frontends parse model or framework input, validate target-independent types and effects, assign stable identities, and emit `ModelIR`. They own import diagnostics and source mappings.

Frontends do not read peak throughput, kernel catalogs, physical topology, or runtime observations. Framework adapters expose canonical IR rather than a parallel public IR hierarchy.

The current frontend covers typed decoder-only Transformer training plus static inference prefill/decode with explicit KV-cache semantics. Additional model families, framework importers, and online serving scenarios are planned.

## Canonical formal-representation infrastructure

The representation core—whose concrete types currently use the `*IR` suffix—provides immutable values, `NodeId` and `ValueId`, typed lineage, exact scalar expressions, canonical JSON, schema versions, feature sets, deterministic digests, and verifier diagnostics.

It has no dependency on Transformer-specific derivation, target plugins, performance providers, or simulation. Typed extensions may carry namespaced semantics; free-form metadata has no compatibility meaning.

## Analysis and transformation infrastructure

`PassManager` executes declarative `PassContract` objects. Each contract declares input/output schemas, required bindings and analyses, preserved and produced analyses, mutation model, verification policy, and determinism.

`AnalysisStore` is content-addressed by representation digest, analysis key, and session fingerprint. Checkpoint observers inspect verified immutable outputs before analyses are atomically published. See [analysis and transformation infrastructure](passes/index.md).

## Planning and search

Planning generates, filters, and ranks candidates across hardware blueprint variables, workload scenarios, mapping strategies, recomputation and pipeline policies, implementation choices, and concrete schedules.

Before architecture binding, pruning uses target-neutral work, communication, memory, and dependency measures plus safe architecture constraints. After binding, objectives may include latency, throughput, utilization, capacity, energy, area, cost, and uncertainty. Every candidate and rejection retains blueprint identity, parent digest, rule, policy, fidelity, budget, and seed.

Candidate generation and Pareto APIs are planned; the current Transformer slice receives one explicit strategy.

## Architecture and deployment subsystem

The planned `ArchitectureBlueprint` defines component hierarchy, compute/memory/interconnect resources, design variables, physical constraints, and capabilities. Existing `TargetBinding`/`TargetProfile` concepts bridge current code until that schema lands. `DeploymentProfile` defines concrete devices, topology, links, capacity, reservations, hosts, and environment revision.

`TargetPlugin` composes independent protocols:

```text
CapabilityModel
LegalityModel
ImplementationSelector
EstimateProviders
ResourceModel
MachineLowering
ArtifactEmitter
ProfilerAdapter
```

A plugin may support architecture, legality, evidence, and simulation before hardware or an emitter exists. The common mapping core owns portable workload semantics and the concrete coordination contract; typed extensions and target verifiers own target-only dataflow, route, issue, and protocol semantics.

## Resource scheduler and memory planner

The scheduler consumes target-legal tasks, deployment resources, and cost views. It jointly decides implementation instances, placement, ordering, dependencies, synchronization, routes, resource occupancy, buffer spaces, offsets, and reuse.

The memory planner reasons over lifetimes under legal overlap, not only an aggregate peak-memory formula. The output must satisfy DAG, queue, synchronization, buffer, capacity, and target-legality verifiers before becoming `ConcretePlanIR`.

Only a queue-oriented experimental schema and structural verifiers exist; typed target extensions, production target binding, and scheduling are planned.

## Products, simulation, and emission

Product modules derive purpose-specific representations from verified plans:

| Module | Product | Authority |
|---|---|---|
| Cost projection | `CostedTaskView` | Rebuildable analysis |
| Timing | `TimingProjection` | Rebuildable analysis |
| Simulator adapter | `SimulationTraceIR` | Derived interchange |
| Timeline packaging | `TimelineBundle` | Provenance-carrying published/replay artifact |
| Architecture evaluation | `EvaluationReport` | Bottleneck, utilization, objectives, uncertainty |
| Exploration analysis | Pareto/sensitivity records | Candidate-comparison artifact |
| Machine lowering | `MachineIR` | Target program semantics |
| Emitter | `ProgramArtifact` | Published executable/replay package |

Simulation is the primary hardware-exploration product path. Optional emission cannot independently reschedule work; both consume the same `ConcretePlanIR` envelope and typed target extensions.

## Observation and calibration

Profiler adapters correlate runtime events with machine instructions and concrete commands. Normalization produces immutable `ObservationSet` records. Calibration fits a domain-scoped model and publishes a new `CalibrationRevision` without overwriting raw evidence.

The same lineage supports forward and reverse queries from model operation to runtime event and back.

## Dependency direction

```text
workload ──► synthesizer/frontend ──► ModelIR
                                       │
                         lowering/passes ──► portable planning
                                       │               │
system ─────────────────────────► analysis              ▼
                                       │    architecture binding
evidence ──────────────────────────────┘               │
                                                       ▼
                                            simulation / emission
```

The dependency direction is explicit: workload contracts do not depend on system descriptions; system descriptions do not depend on analysis policy; analysis does not construct canonical plans. The synthesizer materializes workload and plan facts, while analysis evaluates those facts against system descriptions and external evidence. Callers must not treat cost evidence as an implicit lowering decision.

## Current source map

| Concern | Source | Status |
|---|---|---|
| Workload semantics and logical mapping intent | `workload/` | Implemented Transformer slice |
| Chip, memory, interconnect, and aggregate system profile | `system/` | Implemented limited profile adapter |
| IDs, expressions, codec, frozen values | `synthesizer/{ids,expr,codec,frozen}.py` | Implemented |
| Canonical formal representations (`*IR`) | `synthesizer/ir/` | Implemented contracts |
| Bindings and sessions | `synthesizer/{bindings,session}.py` | Implemented |
| Analysis/transformation transactions | `synthesizer/passes/base.py` | Implemented |
| Workload-to-IR/session frontends | `synthesizer/frontend/` | Implemented Transformer slice |
| Workload and cost analysis | `analysis/` | Implemented slice |
| Transformer derivation passes | `synthesizer/lowering/transformer.py` | Implemented through portable plan |
| Current system cost adapters | `analysis/cost_model.py`, `analysis/cost/` | Implemented slice |
| Architecture model/search, evidence service, simulation, emission | Accepted boundaries | Planned |

# Formal Analysis Module Architecture

Modules follow hardware-experiment ownership, not merely source-file convenience. Each module has typed inputs and outputs, owns a bounded decision class, publishes diagnostics and provenance, and cannot reach into downstream state.

## Exploration facade and derivation context

The future public facade should expose architecture exploration rather than an IR pipeline:

!!! note "Not yet callable"
    `ExplorationSession` and `blueprinting.explore()` are design targets, not implemented in the current repository. The code below is illustrative pseudocode.

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

## Workload, mapping, and system domain models

`blueprinting.workload` owns target-neutral model semantics and request/training scenarios. A workload object cannot contain parallel placement, a chip name, peak rate, empirical latency, kernel identity, or physical placement. The current slice provides typed Transformer model, training-workload, and inference-request contracts.

`blueprinting.mapping` owns target-neutral logical strategies such as TP/PP/DP, recomputation, and collective form. It also owns explicit deployment-side associations such as `NetworkTierBinding`; those associations are supplied to evaluation after portable planning and are never embedded in workload facts or a `PortablePlanIR`. This separation makes the same portable plan evaluable on materially different systems.

Boundary importers still accept the retained Calculon-style field names, but aliases are not a second schema: if canonical and legacy spellings are both present they must agree, otherwise import fails before derivation. Newly constructed domain objects and reports use canonical ownership and typed fields.

`blueprinting.system` owns immutable chip-local compute engines, memory capacity/bandwidth, interconnect tiers, collective volume rules, and their imported evidence revision. `SystemProfile` is the current limited compute/memory/network adapter; it is not yet the hierarchical `ArchitectureBlueprint`, physical deployment, or target binding described by the product design. Cost policy remains in `analysis`: the system contract exposes peak and evidence-bearing facts but does not choose calibration mode.

These three packages are authoritative domain inputs, not alternative IR hierarchies. Canonical derivation starts only when a synthesizer frontend imports workload and logical-strategy contracts into `ModelIR` plus a typed `SynthesisSession`; a system profile and deployment-side network binding remain outside canonical workload state and are consumed by explicit analysis or later target binding.

## Frontends

Frontends parse model or framework input, validate target-independent types and effects, assign stable identities, and emit `ModelIR`. They own import diagnostics and source mappings.

Frontends do not read peak throughput, kernel catalogs, physical topology, or runtime observations. Framework adapters expose canonical IR rather than a parallel public IR hierarchy.

The current frontend covers typed decoder-only Transformer training plus static inference prefill/decode with explicit KV-cache semantics. Additional model families, framework importers, and online serving scenarios are planned.

## Canonical formal-representation infrastructure

`blueprinting.schema` provides the dependency-free canonical codec, frozen maps, and serialization errors shared by all typed contracts. The representation core in `blueprinting.synthesizer`—whose concrete types currently use the `*IR` suffix—provides `NodeId` and `ValueId`, typed lineage, exact scalar expressions, schema headers, feature sets, deterministic digests, and verifier diagnostics.

It has no dependency on Transformer-specific derivation, target plugins, performance providers, or simulation. Typed extensions may carry namespaced semantics; free-form metadata has no compatibility meaning.

## Analysis and transformation infrastructure

`PassManager` executes declarative `PassContract` objects. Each contract declares input/output schemas, required bindings and analyses, preserved and produced analyses, mutation model, verification policy, determinism, and typed lineage rules with executable predicates. Cross-boundary verification is part of the commit gate; deterministic replay is enabled in CI and optionally at runtime.

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

The experimental contract now includes mutually exclusive queue-order and slot/dataflow typed extensions, target verifiers, and deterministic virtual reference binders. They exercise the common envelope against queue-centric and queue-free semantics; production target plugins, resource scheduling, occupancy, and hardware legality remain planned.

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
schema ──► workload ──► mapping
   │          │           │
   ├──────────┴───────────┴──► synthesizer ──► PortablePlanIR
   │                                      │             │
   └──► system ───────────────────────────┼──► analysis ◄── evidence
                         NetworkTierBinding             │
                                                       ▼
                                      application / validation
```

The dependency direction is explicit: workload contracts do not depend on mapping or system descriptions; logical mappings may validate against workload shapes but do not read systems; system descriptions do not depend on analysis policy; analysis does not construct canonical plans. The synthesizer materializes workload and plan facts, while analysis evaluates those facts against explicit system, deployment mapping, and external evidence. Validation may consume the whole supported stack but no production layer depends on validation or an external oracle.

## Current source map

| Concern | Source | Status |
|---|---|---|
| Canonical codec and frozen schema values | `schema/` | Implemented |
| Model and workload semantics | `workload/` | Implemented Transformer slice |
| Logical strategies and explicit deployment mapping | `mapping/` | Implemented Transformer/network slice |
| Chip, memory, interconnect, and aggregate system profile | `system/` | Implemented limited profile adapter |
| IDs, expressions, lineage | `synthesizer/{ids,expr}.py` | Implemented |
| Canonical formal representations (`*IR`) | `synthesizer/stages/*/ir.py` | Implemented contracts |
| Bindings and sessions | `synthesizer/{bindings,session}.py` | Implemented |
| Analysis/transformation transactions | `synthesizer/passes/base.py` | Implemented |
| Workload-to-IR/session frontends | `synthesizer/frontend/` | Implemented Transformer slice |
| Transformer exact-work dialect | `synthesizer/dialects/transformer/` | Implemented training/inference slice |
| Stage-owned derivation passes | `synthesizer/stages/*/passes.py` | Implemented through portable plan |
| Current system cost adapters | `analysis/cost_model.py`, `analysis/cost/` | Implemented slice |
| Framework-neutral orchestration and reports | `application/` | Implemented static analysis slice |
| Calculon/Vidur comparisons and regression gates | `validation/` | Implemented offline gates |
| Optional external performance bundles | `data/evidence/` | Explicitly loaded; excluded from base package |
| Architecture model/search, evidence service, simulation, emission | Accepted boundaries | Planned |

`validation/calculon.py` and `validation/vidur.py` keep external reference implementations behind post-derivation comparison boundaries; `validation/regression.py` freezes their strict drift gates. They are comparison checks, not evidence that the canonical derivation path is correct by construction.

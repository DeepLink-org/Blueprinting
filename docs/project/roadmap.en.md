# Roadmap and Acceptance Gates

The roadmap is organized around hardware-exploration outcomes. Formal representation and analysis infrastructure is delivered when it enables a new class of architecture blueprint, simulation, comparison, or calibration—not merely when another schema exists.

## Product sequencing principles

1. Hold workload semantics constant while hardware variables change.
2. Introduce a first-class architecture model before adding broad target-specific code generation.
3. Compare at least two materially different blueprints in every exploration milestone.
4. Use the cheapest fidelity that can resolve a decision, then refine selectively.
5. Validate workload conservation before calibrating hardware behavior.
6. Make bottleneck, sensitivity, uncertainty, and provenance first-class outputs.
7. Keep optional program emission correlated with, but subordinate to, the simulated plan.

## Phase overview

| Phase | Product deliverables | Acceptance focus | Status |
|---|---|---|---|
| 0 — Semantic foundation | IDs, codec, IR/verifiers, pass transactions | deterministic, immutable, observable facts | **Complete** |
| 1 — Workload baseline | Transformer decomposition, distributed/portable mapping, exact work | conservation and independent Calculon validation | **Partially complete** |
| 2 — Architecture blueprint core | hierarchical compute/memory/interconnect/system model, variables, constraints, normalized evidence adapter | two blueprints with stable identity and legal capability differences | **Planned** |
| 3 — Architecture-bound simulation | legalization, mapping, placement, typed target schedules, buffers, deterministic resource events, `TimelineBundle` | feasibility, utilization, bottleneck, latency/memory comparison | **Planned** |
| 4 — Exploration engine | bounded candidate generation, sensitivity, uncertainty, Pareto records, experiment bundle | reproduce a small-space frontier and explain dominance | **Planned** |
| 5 — Multi-fidelity calibration | performance DB, network/hardware simulator adapters, observations and revisions | consistent provider semantics, event correlation, error improvement | **Planned** |
| 6 — Program and runtime validation | target MachineIR/artifacts, replay/runtime profiler correlation | emitted/replayed events satisfy the declared correspondence level with the simulated concrete plan | **Optional downstream / Planned** |

## Phase 1 completion gate

The workload baseline becomes complete enough for exploration when:

- intermediate buffers and lifetimes are explicit;
- full-model layer/stage composition preserves lineage;
- PP/DP communication and boundary sharding are structural facts;
- initial training and inference scenario suites share common semantics;
- conservation tests cover operations, bytes, messages, recomputation, and persistent state.

No architecture latency or kernel identity enters the portable workload baseline.

## Phase 2 architecture gate

Implement a minimal `ArchitectureBlueprint` with:

- compute-engine capabilities and counts;
- a two-level memory hierarchy with capacity/connectivity;
- one local and one scale-up communication fabric;
- system multiplicity/topology;
- fixed, variable, derived, and constrained fields;
- canonical identity and verifier;
- `HardwareProfile` adapted as versioned evidence rather than architecture truth.

Acceptance requires two materially different virtual blueprints that bind the same portable workload, reject incompatible mappings with diagnostics, and preserve the same source-workload digest.

## Phase 3 simulation gate

Before constructing the first plan, close the common coordination core, typed target-extension boundary, planning/evaluation evidence identities, and runtime correspondence contract. Then construct an architecture-bound plan for each accepted blueprint with implementations, resource placement, ordering, synchronization, routes, resource occupancy, and buffer allocation. Drive a deterministic event simulator from that plan.

The first comparison must publish a `TimelineBundle` that references the same concrete digest and report feasibility, total latency, throughput basis, memory high-water marks, per-resource utilization, critical path, queue delay, uncertainty, and bottleneck attribution. Hand-computable DAGs and the existing analytical estimator act as oracles before detailed simulator integration; queue-centric and non-queue-centric virtual targets check that the common core does not overfit.

## Phase 4 exploration gate

Start with exhaustive enumeration of a small design space to establish ground truth. Then introduce hierarchical pruning and smarter search while retaining:

- candidate generator revision and variable assignments;
- hard-constraint rejection reasons;
- mapping and evidence fingerprints;
- objectives and uncertainty policy;
- search seed, budget, and evaluation fidelity;
- sensitivity results and Pareto dominance evidence.

The milestone is achieved when a user can ask a hardware question, obtain a reproducible frontier, and understand which architectural resource moves each bottleneck.

## Phase 5 calibration gate

Normalize analytical models, performance databases, network simulators, hardware simulators, and measurements behind the same request/result semantics. Add fidelity-aware resolution and explicit interpolation/extrapolation policy.

Correlate observations to architecture components and concrete commands. Calibration creates immutable evidence revisions and must improve held-out prediction error without model/case-specific correction factors. Historical experiment bundles remain reproducible.

## Phase 6 optional program gate

Program emission is not required to establish the architecture-exploration MVP. LPU, GPU, and other backends mature through Architecture, Simulation, Replay, and Executable levels; only the last requires real hardware and an ABI. A backend lowers the same verified concrete-plan envelope used by simulation.

Acceptance requires command/instruction lineage, ABI manifests, a target verifier, failure policy, and profiler correlation. Timing differences become new evidence; the emitter cannot introduce an independent high-level schedule. Target-enforced cycles or slots come from typed extensions rather than encoding a predictive timeline directly as correctness.

## Exploration MVP definition

The core product is credible when:

1. one workload suite maps to at least two materially different architecture blueprints;
2. compute, memory, and communication resources are explicit and verified;
3. simulation produces resource-correlated performance and memory behavior;
4. bottleneck and sensitivity analysis explain the difference between candidates;
5. a bounded multi-objective search reproduces a known small-space Pareto frontier;
6. every conclusion records workload, blueprint, mapping, evidence, simulator, and policy revisions;
7. changing evidence reevaluates results without changing workload or architecture semantics.

Hardware program emission strengthens validation but is not part of this minimum definition.

## Near-term delivery order

1. Define the minimal hierarchical `ArchitectureBlueprint` and verifier.
2. Wrap `HardwareProfile` behind normalized evidence requests/results.
3. Add two parameterized virtual blueprints and architecture legality.
4. Complete portable buffers/lifetimes needed for resource mapping.
5. Use queue-centric and non-queue-centric virtual targets to freeze the common coordination core and typed-extension boundary.
6. Construct a minimal architecture-bound plan, deterministic simulator, and `TimelineBundle`.
7. Produce a two-blueprint bottleneck/utilization comparison.
8. Add bounded variables, exhaustive small-space Pareto analysis, and sensitivity.
9. Integrate network/hardware simulator providers, then optional target emission.

Every delivery updates the bilingual product narrative, [implementation status](status.md), and [risk register](risks.md). Roadmap text never counts as implementation evidence; see the [timeline staging path](../design/timeline-path.md) for the complete semantics.

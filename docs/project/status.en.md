# Implementation Status

This page separates Blueprinting's hardware-exploration product goals from the engineering foundation already connected in the repository. A schema or design contract is useful groundwork, but it is not a working exploration capability until a candidate can be constructed, evaluated, and consumed end to end.

**Status date:** 2026-08-09

## Status vocabulary

| Label | Meaning |
|---|---|
| **Implemented** | Runnable production code and proportionate tests exist in this repository |
| **Implemented experiment** | A reproducible validation path exists but is not yet a general public product API |
| **Experimental Contract** | A schema and verifier run, but producer, consumer, semantic-coverage, or compatibility gates are not connected |
| **Contract Only** | A schema, verifier, protocol, or fixture exists without a connected production path |
| **Planned** | The design direction is accepted, but implementation or a stable contract is absent |

## Product capability matrix

| Hardware-exploration capability | Status | Current evidence or gap |
|---|---|---|
| Typed Transformer training workload accounting | **Implemented** | exact block operations, bytes, collectives, recomputation, and phases |
| Static Transformer inference phase planning | **Implemented slice** | independently verified prefill/decode plans, KV capacity, and decoder-block phase composition |
| Target-neutral workload/mapping plan | **Implemented slice** | Transformer path reaches `PortablePlanIR` |
| Versioned compute/memory/network efficiency profile | **Implemented adapter** | `SystemProfile` and two analytical estimate modes |
| Normalized task-cost resolution and performance-data ingestion | **Implemented slice** | immutable query/result/store, ordered resolver, roofline fallback, generic simulator tables, Vidur profiles, and four AIConfigurator table families |
| Vidur raw component-profile alignment | **Implemented experiment** | exact-key CSV lookup after independent lowering/costing, with component coverage and non-cancelling error attribution |
| Calculon/SeqSel workload and cost calibration | **Implemented experiment** | eight-case reproducible report and tests |
| First-class hierarchical `ArchitectureBlueprint` | **Planned** | documented component model; no production schema/API |
| Hardware design variables and constraint-aware candidate generation | **Planned** | no design-space generator or search session |
| Workload suite and scenario weighting | **Planned** | current path evaluates explicit individual configurations |
| Architecture capability/legalization model | **Planned** | target/deployment bindings exist; general plugin path does not |
| Architecture-bound placement, schedule, and memory plan | **Experimental Contract / Planned** | `ConcretePlanIR` has only a generic queue-oriented schema and structural verifier; producer, route/occupancy semantics, and typed target extensions do not exist |
| Discrete-event compute/memory/resource simulation | **Planned** | current result is analytical composition, not event simulation |
| Timeline analysis/replay bundle | **Planned** | `TimingProjection`, `SimulationTraceIR`, and `TimelineBundle` are design contracts only |
| General network/hardware simulator adapters | **Implemented slice / Planned** | explicit tabular ingestion and a general resolver exist; simulator execution, calibrated interpolation, contention validity, and environment manifests remain planned |
| Bottleneck, utilization, sensitivity, and what-if reports | **Planned** | no general architecture report product |
| Energy, area, power, thermal, and cost models | **Planned** | dimensions are specified but no providers exist |
| Multi-objective Pareto architecture search | **Planned** | no candidate frontier API |
| Profiler observation ingestion and calibration revisions | **Planned** | offline comparison exists; no closed-loop service |
| GPU/LPU/other program emission | **Experimental Contract / Planned** | optional downstream `MachineIR` schema only; no LPU capability/ABI, emitter, or runtime contract |

This matrix is authoritative for user-facing claims. The existence of five IR classes must not be summarized as a completed hardware simulator.

## Schema maturity is not capability maturity

The five current IR classes use `1.0.0` as an internal canonical serialization version. The number does not mean a public API or ABI is frozen, nor that every layer has a production producer and consumer. `ConcretePlanIR` and `MachineIR` in particular remain experimental scaffolds; they must pass the compatibility gate in the [risk register](risks.md) before graduating to stable contracts.

## Connected analysis path

The current runnable slices are:

```text
TransformerModelSpec + TransformerExecutionSpec
  -> exact workload decomposition
  -> ModelIR
  -> DistributedTaskIR
  -> PortablePlanIR
  -> SystemProfile analytical estimate
  -> Calculon / paper comparison report

TransformerModelSpec + inference mapping + request cohort
  -> phase-neutral inference ModelIR
  -> independently bound prefill and decode DistributedTaskIR
  -> phase-local PortablePlanIR with KV state/capacity
  -> CostResolver(exact imported evidence -> explicit roofline fallback)
  -> optional post-hoc Vidur baseline comparison
  -> static prefill / decode-step model time and analytical memory report
```

`PassManager` verifies each staged derivation and exposes immutable checkpoints. `SystemProfile` supplies evidence after the portable plan. The path does not yet construct an architecture hierarchy, bind physical resources, execute a discrete-event simulation, or search hardware candidates.

## What the current result can claim

The repository can claim that selected Transformer training workloads are decomposed into auditable target-neutral work and compared against one versioned system evidence profile without case-specific timing coefficients. It can also derive dense-MHA inference prefill and decode phase points, derive KV capacity, and compose a homogeneous request cohort with explicit evidence provenance. It provides the stable identities, verifier gates, and pass-level checkpoint hooks needed for later simulation correlation.

It cannot yet claim serving-system SLO accuracy: arrivals, queueing, continuous batching, scheduler overhead, contention, and tail distributions are absent. Nor can it yet claim that Blueprinting explores compute/memory/interconnect parameters, predicts NoC behavior, models energy/area/cost, constructs a legal concrete hardware schedule, produces sensitivity/Pareto results, or closes a calibration loop on real GPU/LPU observations.

## Engineering foundation

| Foundation | Status | Source of truth |
|---|---|---|
| Immutable values, stable IDs, lineage, codec, digests | **Implemented** | `src/blueprinting/synthesizer/{frozen,ids,codec}.py` |
| Five progressive formal-representation schemas (`*IR`) and verifiers | **Experimental Contract** | `src/blueprinting/synthesizer/ir/`; only the first three have a production derivation slice |
| Typed workload/strategy/target/deployment bindings | **Implemented** | `bindings.py`, `session.py` |
| Chip, memory, interconnect, and aggregate system profile | **Implemented adapter** | `src/blueprinting/system/`; evidence-bearing profile, not the planned `ArchitectureBlueprint` |
| Transactional analyses/transformations, checkpoints, observers | **Implemented** | `passes/base.py` |
| Transformer workload contracts, frontend, and workload algebra | **Implemented slice** | `workload/transformer.py`, `synthesizer/frontend/transformer.py`, `analysis/transformer_workload.py` |
| Distributed and portable mapping derivations | **Implemented slice** | `lowering/transformer.py` |
| Cost protocol, resolver, roofline, database, and external importers | **Implemented slice** | `analysis/cost/`, `analysis/vidur.py`; exact task latency only, not plan simulation |
| Static inference frontend, lowering, cost, and request composition | **Implemented slice** | `workload/transformer_inference.py`, `synthesizer/frontend/transformer_inference.py`, `analysis/{transformer_inference,inference_cost}.py`, `synthesizer/lowering/transformer_inference.py`, `application/inference.py` |
| Vidur raw component-profile alignment | **Implemented experiment** | `analysis/vidur.py` + `experiments/vidur.py`; a minimal licensed CI slice is pinned locally and the full upstream corpus remains external |
| Calculon experiment | **Implemented experiment** | `experiments/calculon.py` |
| External-baseline regression gate | **Implemented** | frozen contract and licensed offline fixtures under `data/validation/`; `experiments/regression.py`; `.github/workflows/quality.yml` |

These typed representations, verifiers, derivation transactions, and analyses are the formal foundation for hardware exploration. New architecture models, simulator providers, and analysis products should extend this one semantic foundation rather than establish parallel workload truth.

## Verification baseline

The current test suite covers binding consistency, canonical serialization, verifier rejection, pass transaction rollback, checkpoint observers, workload conservation, Calculon calibration, prefill/decode scaling, KV capacity, static request composition, baseline-only Vidur comparison, roofline components, exact/ambiguous database resolution, simulator unit normalization, AIConfigurator CSV/Parquet schemas, explicit Vidur ingestion, and inference resolver fallback. A dedicated CI job runs the eight-case Calculon/SeqSel and three-case pinned Vidur gates on every main-branch pull request and push. It freezes provenance, semantic policy, coverage, comparable-subtotal drift budgets, non-cancelling component errors, aggregate results, and IR digests; it cannot silently regenerate goldens. The Vidur gate is drift detection, not an accuracy certification. Documentation checks enforce complete bilingual page pairs and strict site builds.

Status promotion requires an end-to-end product test. For example, introducing `ArchitectureBlueprint` as a dataclass is Contract Only; constructing two different candidates, mapping the same workload, producing comparable results, and preserving provenance is the minimum product-level evidence.

## Next product milestone

The next milestone is a minimal two-blueprint exploration:

```text
one Transformer workload suite
  + two parameterized virtual ArchitectureBlueprints
  + normalized SystemProfile evidence
  -> legal architecture-bound plans
  -> deterministic resource simulation
  -> bottleneck + utilization + latency/memory comparison
  -> reproducible experiment bundle
```

This milestone proves the name “Blueprinting”: the system compares hardware blueprints and explains the difference. Replayable MachineIR may validate the same plan later but is not a gate for the first exploration MVP. See the [roadmap](roadmap.md).

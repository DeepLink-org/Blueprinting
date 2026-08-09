# Architecture Risk Register

This page records systemic risks that can break hardware-candidate comparability, plan executability, or conclusion credibility. It is not a meeting-specific question list. The same gates apply to compiler, chip, simulation, network, and ML-systems collaborators.

**Audit date:** 2026-08-09

## Severity

- **P0:** must close before architecture-bound scheduling begins or an end-to-end capability is claimed;
- **P1:** prototypes may proceed, but stable contracts or performance conclusions cannot be published; and
- **P2:** does not immediately break correctness but creates extension, reproducibility, or explanation cost.

## Current risks

| ID | Severity | Risk | Current evidence | Control and closure gate | State |
|---|---|---|---|---|---|
| R-01 | P0 | Timeline is interpreted as both execution truth and predicted trace | The removed legacy stack used `TimelineIR`; current code uses `ConcretePlanIR` plus derived timing | Unify terminology; predictive time does not define correctness; target-enforced time is a typed target semantic only | Documentation corrected; implementation gate pending |
| R-02 | P0 | The common concrete schema overfits a GPU queue/stream model | Current v1 has device, queue, buffer, and command only; there is no typed target-schedule extension | Validate the common coordination core with a non-queue-centric virtual target; forbid correctness-critical free-form metadata | Open |
| R-03 | P0 | Documentation presents a schema/verifier scaffold as a resource-complete plan | There is no portable-to-concrete producer; route, occupancy, and target-specific schedule semantics are incomplete | Downgrade status to experimental contract; upgrade only after producer, consumers, and end-to-end verifiers pass | Wording corrected; capability not implemented |
| R-04 | P1 | Planning-evidence and evaluation-evidence identities are conflated | `ConcretePlanIR.evidence_revision` records construction input while derived cost views also use evidence | Separate construction provenance from re-evaluation revision in the contract; verify that re-costing cannot silently mutate a plan | Open |
| R-05 | P0 | Simulator/runtime “equivalence” is read as equal temporal behavior | No real backend or conformance evidence exists | Promise only command/event correspondence; define strict, bounded-divergence, and partial-observation levels | Documentation corrected; tests pending |
| R-06 | P1 | Schema version `1.0.0` is mistaken for public stability | Five IR schemas are versioned, but target producers and consumers are not connected | State that internal serialization version is not a compatibility promise; establish a graduation checklist | Documentation corrected; policy pending |
| R-07 | P0 | “Zero-decision runtime” ignores backpressure, failure, and dynamic duration | No runtime contract exists | Forbid unbounded global replanning while retaining bounded safety and mechanism decisions; declare policy in artifacts | Documentation corrected; runtime pending |
| R-08 | P0 | LPU features leak prematurely into portable semantics | No LPU ABI, capability, or resource contract exists yet | Stage Architecture/Simulation/Replay/Executable maturity; introduce physical detail only after the target gate | Controlled |
| R-09 | P1 | Simulator and emitter each fill in a missing schedule | Neither production path exists and no cross-consumer conformance test exists | Both consume the same concrete digest plus typed target extension; emitter decision delta must be empty | Open |
| R-10 | P1 | “Formal verification” is read as theorem-proved correctness | Current assurance consists of typed schemas, executable verifiers, property tests, and external experiments | Attach assurance levels to claims; do not use certified/proven language without mechanized proof | Documentation corrected |
| R-11 | P1 | Calibration matches references with case-specific parameters | The Calculon path rejects per-case timing coefficients, but the general evidence service is not implemented | Version training/validation splits, validity domains, uncertainty, and held-out gates | Partially controlled |
| R-12 | P2 | Static plans silently exclude dynamic workloads | The current slice is an explicit static Transformer training configuration | Declare shape/workload envelopes; diagnose or explicitly replan outside them rather than silently reusing a plan | Open |

## Compatibility gate before publication

Before any canonical schema, target plugin, or timeline bundle is declared stable, it needs at least:

1. one production producer and two independent consumers;
2. canonical round-trip and unknown-extension behavior;
3. a negative verifier suite, not only happy-path fixtures;
4. at least one queue-centric and one non-queue-centric virtual target;
5. a schema migration and compatibility policy;
6. provenance, evidence invalidation, and replay tests; and
7. an `Implemented` documentation state backed by a runnable end-to-end test.

## Risk-closure rule

Renaming concepts can remove ambiguity but cannot close implementation risk. A risk moves from “documentation corrected” to “closed” only when the corresponding producer, consumers, verifier, negative tests, and reproducible experiment exist. State changes must stay synchronized with [implementation status](status.md) and the [roadmap](roadmap.md).

# Decisions and Terminology

This page is the compact index of architecture commitments, rejected alternatives, unresolved questions, and shared vocabulary. Detailed reasoning lives in the design chapters; compatibility-affecting changes require a dedicated Architecture Decision Record (ADR).

## Accepted decisions

| Topic | Decision | Consequence |
|---|---|---|
| Product identity | Hardware architecture exploration and simulation through formal derivation, verification, and automated analysis | User-facing outputs prioritize blueprints, bottlenecks, sensitivity, uncertainty, and trade-offs |
| Canonical representations | The current semantic backbone has five typed representations; no parallel public hierarchy, and any new stable boundary requires an ADR plus migration evidence | Every frontend and backend joins one derivation chain without treating an unvalidated layer count as eternal fact |
| Target binding | Only at `PortablePlanIR -> ConcretePlanIR` | Portable planning remains reusable across GPU, LPU, and other targets |
| Execution truth | The `ConcretePlanIR` envelope: common coordination core plus typed target extensions | Simulation and emission cannot reconstruct independent schedules |
| Timing | Predictive timing is derived analysis; target-enforced timing is a typed target semantic after binding | Evidence revisions can change predictions without changing correctness; LPU issue/slot constraints are not erased |
| Evidence | Immutable, versioned, and provenance-carrying | Calibration creates revisions instead of overwriting facts |
| Profiler | Observer and evidence source; never a formal-state mutator | Every derivation remains deterministic and auditable |
| Transformation engine | Typed declarative pass contracts and atomic publication | Failed verification or observation leaves no partial state |
| Target extension | Protocol-composed `TargetPlugin` | Capabilities can mature before hardware emission exists |
| Equality saturation | Bounded candidate generation only | E-graphs do not become canonical IR or physical schedulers |
| Runtime | Honor a verified plan while retaining bounded mechanism decisions allowed by the contract | Runtime does not repeat unbounded global search or pretend backpressure and failure do not exist |
| Schema maturity | Internal schema versions are not automatically public compatibility promises | A contract graduates only after producer, independent consumer, migration, and conformance gates pass |
| Documentation | Colocated suffix-based bilingual sources | Navigation and language switching remain page-aligned |
| Formal derivation package | Hard-cut Python rename to `blueprinting.synthesizer` | Source ownership matches formal plan synthesis; see [ADR-0001](adr/0001-synthesizer-package.md) |
| Domain packages | `blueprinting.workload` owns target-neutral workload contracts; `blueprinting.system` owns chip/interconnect/system profiles | Synthesis and analysis consume explicit domain inputs without owning them; see [ADR-0002](adr/0002-workload-system-domains.md) |
| Derivation debugging | Five-stage graphs, adjacent mappings, and debug bundles are derived traces rebuilt from checkpoints and lineage | UI and overlays never enter canonical IR; see [ADR-0003](adr/0003-derivation-debug-trace.md) |
| Canonical wire identity | Domain-owned `blueprinting.*` namespaces with no obsolete aliases | Serialized identities describe current semantics and old artifacts are regenerated at the hard boundary; see [ADR-0004](adr/0004-semantic-wire-identities.md) |
| Algebraic canonical constructors | Scalar operations and concrete command semantics use constructor-specific ADTs; preservation claims are executable evidence | Invalid arity/payload combinations are removed and schema changes use explicit migrations; see [ADR-0005](adr/0005-algebraic-expression-command-schemas.md) |
| Progressive typed Python | Runtime `Checked` contracts and sealed core ADTs share declarations with optional standard mypy analysis; no custom plugin | Base installs retain contract checking, expected failure is explicit, and static analysis adds coverage without becoming runtime truth; see [ADR-0006](adr/0006-progressive-typed-python-contracts.md) |

## Rejected alternatives

**One universal graph with optional fields.** This makes ownership and legality depend on conventions, permits target leakage, and lets unknowns survive until runtime. Separate IR contracts make each lowering gate verifiable.

**Duration stored on portable tasks.** A duration is not portable: it depends on implementation, target, deployment, concurrency, evidence, and revision. It belongs in a cost view.

**A predicted timeline as the execution source of truth.** Absolute timestamps conflate prediction with readiness. Common dependency, resource, and buffer contracts define execution; a predicted timeline is a projection. If a target's issue cycle or slot has correctness meaning, it enters a typed target extension and `MachineIR` after binding rather than masquerading as a generic prediction field.

**One correction coefficient per model or case.** Such a coefficient can match a report but does not explain workload, implementation, network, or schedule behavior and does not generalize.

**Early LPU or CUDA binding.** It would make the frontend and portable planner target-specific before the hardware contracts stabilize. Capability requirements preserve late specialization; LPU can mature first as an architecture and simulation target before gaining replay and executable backends.

**Independent simulator and emitter schedulers.** Matching their totals would not prove they describe the same program. Both must consume the concrete plan.

**Parallel legacy and new IR stacks.** Adapters may read legacy inputs, but maintaining two public semantic hierarchies creates ambiguous ownership and divergent fixes.

**Compiler-first product framing.** It mistakes a borrowed implementation toolbox for the system itself. Blueprinting has no conceptual Compiler component: its method is formal modeling, derivation, verification, and automated analysis. IR, lowering, and passes remain valid implementation terms inside **Formal Analysis Foundations**.

## Open decisions

The following questions remain deliberately unresolved:

1. whether and when a binary format supplements canonical JSON v1;
2. which dialects, if any, should migrate to MLIR and at what maturity point;
3. the final typed extension mechanism for portable commands;
4. target artifact containers and ABI compatibility windows;
5. process or RPC isolation for expensive/untrusted simulator providers;
6. Pareto dominance and pruning under correlated uncertainty;
7. the preferred external trace interchange schema;
8. the boundary for introducing target-specific equality exploration;
9. evidence-retention, privacy, and promotion policies for profiler data.

Each item needs an ADR before it becomes a serialized, public, or target-plugin compatibility dependency. See the [architecture risk register](risks.md) for current high-risk open items and their graduation gates.

## ADR policy

An ADR is required when a proposal changes a canonical schema, lowering gate, ownership boundary, serialized identity, plugin protocol, evidence semantics, or public compatibility promise. An ADR should contain context, decision drivers, considered alternatives, decision, consequences, migration, validation, and status.

Accepted ADRs are immutable except for status links and typo fixes. A superseding decision creates a new ADR and points back to the old one. Implementation status remains separate: accepting an ADR does not mark its design Implemented.

## Normative vocabulary

| Term | Meaning |
|---|---|
| Workload | Model semantics, training/inference mode, and workload parameters |
| Architecture blueprint | A versioned candidate hierarchy of compute, memory, interconnect, system, capability, and physical constraints |
| Strategy | Parallelism, recomputation, pipelining, fusion, layout, and algorithm choices |
| Target | A blueprint bound to a concrete hardware/software capability and ABI family |
| Deployment | Concrete devices, topology, capacities, reservations, and environment revision |
| Evidence | Measurement, simulation, analytical, or calibrated information with provenance |
| Plan | A workload mapping onto one blueprint with selected implementations and execution constraints |
| Formal representation | An authoritative typed model with a versioned contract and verifier; current canonical types use the `*IR` suffix |
| Derived view | A rebuildable projection that cannot change source representation semantics |
| Artifact | A published report, trace, executable, or replayable package |
| Binding | An explicit specialization fact supplied through a typed derivation context; current code calls it `SynthesisSession` |
| Derivation | A verified rule that resolves decisions, discharges obligations, and preserves required semantics |
| Lowering | A compiler-engineering implementation technique used for a staged derivation |
| Revision | An immutable identity for evidence, schema, analysis engine, plugin, or product state |

The words **MUST**, **MUST NOT**, **SHOULD**, **SHOULD NOT**, and **MAY** carry normative force in English pages. Their Chinese equivalents are **必须**、**不得**、**应该**、**不应该** and **可以**.

## Reference systems

The architecture is informed by—not intended to reimplement—these adjacent systems:

- [MLIR dialect conversion](https://mlir.llvm.org/docs/DialectConversion/) for progressive legality and lowering;
- [IREE design roadmap](https://iree.dev/developers/design-docs/design-roadmap/) for compiler-planned resources and multi-target executables;
- [OpenXLA architecture](https://openxla.org/xla/architecture) and [StableHLO](https://openxla.org/stablehlo/spec) for portable semantics and target lowering;
- [TVM MetaSchedule](https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/meta_schedule.html) for measurement-driven search and database separation;
- [Chakra](https://github.com/mlcommons/chakra) and [ASTRA-sim](https://github.com/astra-sim/astra-sim) for trace interchange and distributed simulation;
- [Timeloop](https://github.com/NVlabs/timeloop) for accelerator mapping and architecture modeling.

References are comparison points, not evidence that Blueprinting implements their capabilities. The [implementation status](status.md) remains authoritative for repository reality.

# Planning and Execution IR

`PortablePlanIR`, `ConcretePlanIR`, and `MachineIR` separate portable execution intent, concrete resource decisions, and target encoding. The boundary prevents target evidence from leaking upstream while allowing the backend to express accelerator-specific behavior fully.

## PortablePlanIR

### Semantic domain

`PortablePlanIR` represents a selected or candidate deployment strategy with target-neutral execution constraints.

```text
PortablePlanIR
├── tasks and dependency DAG
├── WorkloadFacts
├── abstract ResourceRequirements
├── abstract buffers and lifetimes
├── implementation capability alternatives
├── concurrency groups
├── objectives and constraints
└── provenance and lineage
```

Exact operations, read/write bytes, message bytes, reuse, and arithmetic intensity belong here when derivable. Recomputation, checkpointing, fusion, pipeline order, and abstract resource choices are explicit structural decisions.

### Forbidden information

Portable plans cannot contain physical implementation IDs, vendor libraries, physical devices, routes, queues, engines, memory banks, addresses, target-derived latency, or execution timestamps.

### Verification

The verifier checks task-reference integrity, DAG closure, buffer lifecycle consistency, resource-requirement validity, exact nonnegative work facts, objective and constraint identity, source lineage, and the absence of target-bound fields.

`PlanSet` is an immutable artifact containing candidates and dominance provenance, not a sixth canonical dialect.

## Target and deployment binding gate

The gate consumes a portable candidate, target, deployment, implementation policy, and evidence policy. It must:

1. prove target capability and ABI compatibility;
2. select legal implementation candidates;
3. map abstract resources to target resources;
4. establish physical placement and memory spaces;
5. make queue, synchronization, and buffer decisions sufficient for correctness;
6. reject unresolved or infeasible requirements with typed diagnostics.

Early `TargetRequirements` can constrain device count, capacity, dtype, or required collectives, but they do not identify a vendor, architecture, runtime, or physical cluster.

The evidence here is **planning evidence**: it explains why an implementation and schedule were selected. Later re-costing of the same plan uses **evaluation evidence**, which belongs to derived-view identity. The current `ConcretePlanIR.evidence_revision` field does not distinguish these roles by name; this contract ambiguity must close before a production producer exists.

## ConcretePlanIR

### Semantic domain

`ConcretePlanIR` is the authoritative execution-plan envelope after target and deployment binding. Its common coordination core carries the command DAG that is stable across targets; a typed target extension carries dataflow, route, issue, or protocol constraints whose correctness meaning belongs to one architecture:

```text
ConcretePlanIR
├── selected implementations
├── devices, queues, and engines
├── commands and dependency tokens
├── synchronization and collectives
├── buffer regions, offsets, lifetimes, and reuse
├── target/deployment/ABI fingerprints
├── evidence and planner revisions
├── typed target schedule extension
└── lineage to portable tasks
```

Initial commands are `Launch`, `Collective`, `Transfer`, `Barrier`, `Signal`, `Wait`, and `HostCall`. Planned allocations may be static buffer bindings and do not require runtime allocation commands.

### Operational meaning

Commands execute when dependency, ordering, synchronization, resource, and buffer contracts are satisfied. Predicted wall-clock timestamps are neither required nor authoritative. If a target makes cycles or slots correctness constraints, they belong to the target extension and later `MachineIR`, not to a generic duration annotation.

### Verification

The verifier for a complete producer must check DAG acyclicity, dependency-token production and consumption, ordering legality, cross-resource synchronization, implementation coverage, placement, buffer lifetime and overlap, address bounds, resource capacity, target fingerprints, typed extensions, and lineage.

The current repository implements only a generic device/queue/buffer/command schema and a set of structural verifiers. It has no production portable-to-concrete construction pass, route or resource-occupancy semantics, typed target extension, or end-to-end target conformance. This v1 is an experimental serialization contract, not a frozen public ABI.

## MachineIR

`MachineIR` is a target-specific lowering product. “Machine” does not promise a bare ISA: it may own instructions or runtime commands, program sections, entry points, symbols, target ABI, executable references, and target-verifier requirements.

Targets may use distinct dialects: CUDA/NCCL calls and executable references, an LPU command packet and memory descriptor format, CPU tasks, firmware commands, or an external framework package. Common concepts use protocols, but target-specific semantics are not erased for superficial uniformity.

Machine lowering cannot reinterpret portable work or add an unplanned communication dependency. Before emission, a target verifier checks legality and ABI compatibility.

## Derived products

| Product | Contents | Source of invalidation |
|---|---|---|
| `CostedTaskView` | Versioned estimates keyed to plan entities | Source, target, deployment, evidence, or calibration changes |
| `TimingProjection` | Predicted intervals, critical path, overlap, uncertainty | Concrete plan, costs, or simulation policy changes |
| `SimulationTraceIR` | Event/counter interchange with command lineage | Projection or adapter revision changes |
| `TimelineBundle` | Published concrete digest + projection/trace + evidence/policy + diagnostics | Any referenced artifact or manifest policy changes |
| `EvaluationReport` | Objectives, constraints, bottlenecks, validation | Plan, evidence, or policy changes |
| `ObservationSet` | Immutable measured or simulated records | Never mutated; superseded by a new revision |
| `ProgramArtifact` | Executable or replay package and manifest | MachineIR, ABI, library, or emitter changes |

## Program artifact contract

An artifact manifest records schema version, target and deployment requirements, target ABI, portable/concrete/machine digests, analysis-engine and plugin versions, required runtime libraries, and a debug lineage map or detachable symbol package.

Timing may be packaged for diagnostics, but it is not a loader or runtime correctness requirement.

Likewise, a `TimelineBundle` only references canonical and derived artifacts; it cannot copy and mutate the command DAG. See the [timeline path](../timeline-path.md) for the complete staging model.

## End-to-end traceability

The toolchain supports both directions:

```text
Model operation -> distributed task -> portable task
                -> concrete command -> machine instruction -> profiler event

Profiler event -> machine instruction -> concrete command
               -> portable task -> distributed task -> model operation
```

Reports and observations use stable IDs and snapshot digests, never display names as identity.

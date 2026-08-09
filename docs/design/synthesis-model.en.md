# Formal Derivation and Verification Model

The derivation model defines what is known while a workload is mapped onto a candidate hardware blueprint, which decisions remain open, which facts may change, and which checkable claims every transition must establish. It is the semantic spine shared by workload models, mapping planners, architecture models, evidence providers, simulators, optional emitters, and profilers.

!!! note "What formal means here"
    Formal means typed semantics, explicit constraints, executable verification conditions, deterministic derivations, and traceable claim boundaries. Blueprinting does not yet claim proof-assistant certification or exhaustive model checking; those require their own connected implementations and evidence.

### Assurance levels

Every use of “verified” in documentation or reports must identify its assurance level. A local schema check cannot be promoted into a system-correctness claim:

| Level | Claim it can support | Claim it cannot support |
|---|---|---|
| Structural | Types, references, DAGs, schemas, and local legality checks pass | Correct workload semantics or accurate performance |
| Conservation | Executable invariants or property tests preserve operations, bytes, shards, or dependencies | A theorem over every input |
| Cross-consumer conformance | Simulator, emitter, and replay consume the same plan digest and align event lineage | Equal real-time behavior |
| Empirical correlation | Measurement quantifies error in a declared workload, target, and evidence domain | Out-of-domain generalization or causal optimality |
| Mechanized proof | An independent formal specification and proof artifact support the claim | Blueprinting does not currently provide this level |

A schema version, test count, or matching reference total cannot by itself raise the assurance level.

## Derivation state

A derivation step operates on explicit state rather than ambient Python globals:

```text
DerivationState_i = (
    representation_snapshot,
    resolved_bindings,
    candidate_provenance,
    analyses,
    evidence_snapshot,
    open_obligations
)
```

The implementation names its typed derivation context `SynthesisSession`. It identifies workload, strategy, target requirements, deployment, calibration revision, feature set, and deterministic seed; its fingerprint participates in every analysis address. The term belongs to the `blueprinting.synthesizer` implementation boundary and does not introduce a conceptual Compiler component.

## Derivation contract

Every staged derivation is a partial function. The current code often realizes such a rule as a lowering pass:

```text
Derive_i(DerivationState_i)
  -> Verified<DerivationState_i+1>
   | Diagnostics
```

It has three obligations:

### Semantic preservation

The meaning of the source must equal the projection of the result back to the source abstraction:

```text
meaning(IR_i) = project_i(meaning(IR_{i+1}))
```

Decomposition, sharding, recomputation, implementation selection, and command encoding may add structure, but cannot silently change the workload. If a declared approximation is allowed, equality is replaced by a typed refinement relation or error bound, and the approximation policy becomes provenance.

### Monotonic constraint refinement

A derivation can add constraints or select an alternative; it cannot admit a result whose projection was illegal upstream. The obligation is semantic, not merely that more predicates were appended:

```text
project_i(Solutions(C_{i+1})) ⊆ Solutions(C_i)
```

### Owned-obligation discharge

Each boundary must discharge every obligation assigned to that boundary and increase specificity. It may introduce explicitly typed downstream obligations, so simply counting all unknowns is not a valid progress measure:

```text
owned_i(Open(State_{i+1})) = ∅
specificity(State_{i+1}) > specificity(State_i)
```

If an owned obligation remains, or a new downstream obligation cannot be represented safely, the derivation fails with a diagnostic instead of inserting a default.

## Binding lattice

| Axis | Typical values | First structural effect |
|---|---|---|
| Workload | batch, sequence, mode, micro-batch | Shape specialization and workload expansion |
| Strategy | TP/PP/DP, recompute, pipeline, fusion | Distributed graph and portable-plan topology |
| Target | architecture, runtime, primitive support, ABI | Legality and implementation selection |
| Deployment | instances, topology, capacity, reservations | Placement, routing, scheduling, and memory feasibility |
| Calibration | evidence snapshot and model revision | Estimate values only |

Portable planning may consume a target requirement envelope for pruning. A requirement is not a binding: it cannot carry a vendor identity, physical device, kernel ID, or measured latency.

## Derivation gates

| Transition | Required context | Must be resolved on success |
|---|---|---|
| Source → `ModelIR` | Model semantics | Operation meaning, value types, effects, declared dynamic dimensions |
| `ModelIR` → `DistributedTaskIR` | Strategy | Logical ownership, sharding, communication, cross-rank dependencies |
| `DistributedTaskIR` → `PortablePlanIR` | Planning policy | Task DAG, abstract resources, work facts, buffer lifecycles, objectives |
| `PortablePlanIR` → `ConcretePlanIR` | Target, deployment, implementations, evidence policy | Legality, placement, queues, synchronization, buffers, capacity |
| `ConcretePlanIR` → `MachineIR` | Target ABI and emitter | Instruction forms, sections, entry points, references |

## Automated analysis over verified states

Transformation and analysis are distinct. A transformation resolves a decision and creates a new authoritative state; an analysis derives a rebuildable result without changing source semantics. The main analysis classes are:

| Analysis class | Establishes |
|---|---|
| Exact workload analysis | Operations, bytes, messages, dependencies, and lifetimes derived from model semantics |
| Legality and feasibility | Capability coverage, shape/layout rules, topology, capacity, and scheduling constraints |
| Quantitative evaluation | Evidence-backed latency, throughput, contention, energy, area, cost, and uncertainty |
| Plan analysis | Critical path, utilization, bottlenecks, memory pressure, and overlap consequences |
| Cross-candidate analysis | Dominance, Pareto frontiers, sensitivity, robustness, and evidence-dependent ranking |

Every result records the source representation digest, binding and evidence fingerprints, analysis revision, validity domain, and diagnostics. An analysis may trigger a new planning decision, but it cannot silently mutate the state it observed.

## Concrete execution abstract machine

`ConcretePlanIR` is meaningful through an abstract machine, not through a list of predicted timestamps.

```text
MachineState = (
    command_state,
    queue_heads,
    resource_occupancy,
    buffer_state,
    synchronization_tokens,
    observations
)
```

A command is ready when:

```text
ready(c) =
    dependencies_completed(c)
    and queue_predecessors_completed(c)
    and synchronization_satisfied(c)
    and resources_available(c)
    and buffers_valid(c)
```

The command vocabulary begins with `Launch`, `Collective`, `Transfer`, `Barrier`, `Signal`, `Wait`, and `HostCall`. Commands transition from pending to ready, running, and completed or failed. Completion publishes dependency tokens and observations and releases resources according to the buffer and queue contracts.

## Simulation and runtime correspondence

Simulation and runtime must share command identities, dependency intent, resource and buffer contracts, and an observable event vocabulary. They need not share one transition implementation and do not promise equal wall-clock behavior. Their first difference is the source of completion:

```text
Simulator: completion_time := resolve_cost(command, context)
Runtime:   completion       := observe_target_event(command)
```

The simulator advances a logical clock and models resources. The runtime submits actual work and awaits target events. A runtime may perform bounded backpressure, retry, timeout, and failure handling explicitly allowed by its target contract, but it cannot silently add high-level dependencies, change the workload, or repeat unbounded global planning.

Conformance asks whether projected runtime observations belong to the trace language allowed by the concrete plan:

```text
project(runtime_events, observation_policy)
  conforms_to allowed_traces(ConcretePlanIR, target_extension)
```

The observation policy declares an exact-event, bounded-divergence, or partial-observation level. Predicted-versus-observed timing is a separate correlation result, not a proof of execution equivalence.

## Determinism and revisions

A fixed source snapshot, binding set, analysis-engine/plugin versions, evidence revision, policy, and seed must produce the same canonical digests and diagnostics. Search may explore alternatives, but its budget, seed, pruning decisions, and rejected candidates remain provenance.

Evidence and observations are immutable revisions. Re-estimation can rebuild views, and replanning can create a new concrete plan, but no service mutates a previously published IR or artifact.

## Failure model

Failures are typed and stage-specific: malformed source, failed verification, missing binding, unsupported capability, unresolved evidence, infeasible capacity, scheduling failure, target-verifier failure, and runtime observation mismatch. A failure reports stable subject IDs, source lineage, relevant fingerprints, the violated rule, and considered fallbacks.

Transformation execution is transactional: a failed verifier or observer leaves the previous representation snapshot and analysis store unchanged.

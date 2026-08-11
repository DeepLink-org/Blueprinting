# Architecture Binding and Plan Construction

Architecture binding is the formal bridge from a portable workload mapping to an architecture-bound simulation plan. It checks a verified `PortablePlanIR` against a candidate hardware blueprint, deployment, and evidence policy, then constructs one authoritative `ConcretePlanIR` envelope from which timing, simulation, and optional target programs are derived.

!!! warning "Design status"
    The late-binding boundary, experimental schemas, and deterministic queue/slot reference binders exist. The target-plugin registry, a production producer, and the downstream analysis/transformation chain remain **Planned**.

## Why this is one vertical slice

Legalization, implementation selection, cost resolution, placement, scheduling, and memory planning cannot be implemented as unrelated utilities. Each decision constrains the next one, and scheduling can invalidate an allocation or implementation choice. The first target-complete milestone therefore needs an end-to-end virtual target, not several disconnected half-passes.

The accepted sequence is:

```text
PortablePlanIR
  -> target legalization
  -> implementation candidates
  -> cost resolution analysis
  -> placement + target scheduling + memory planning
  -> ConcretePlanIR
  -> timing/simulation projection -> TimelineBundle
  -> optional replayable MachineIR artifact
```

## Binding gate

The gate requires three explicit inputs:

- `TargetBinding`: architecture, runtime, capability, library, and ABI identity;
- `DeploymentBinding`: device instances, topology, capacity, and reservations;
- evidence policy and revision: permitted providers, uncertainty budget, fallback rules, and freshness.

The input portable digest remains unchanged. All selected identities and planning-evidence fingerprints enter concrete-plan construction provenance; later evaluation evidence enters derived-analysis identity. A missing binding is a diagnostic, never a request to read a process-global default.

## Legalization and implementation selection

Legalization maps each capability requirement to legal target implementations. It checks dtype, shape, layout, alignment, memory-space, collective, runtime, library, and ABI constraints. A target plugin may expose rewrite alternatives, but it cannot change exact workload operations, bytes, or logical collective semantics.

Implementation selection may keep several candidates for search. Every candidate records the originating portable task, target capability rule, plugin revision, required resources, and rejection reasons. Unsupported work fails with a source-linked diagnostic rather than silently choosing a generic zero-cost implementation.

## Cost resolution analysis

Cost resolution builds normalized estimate requests for legal implementation candidates and resolves evidence under the experiment policy. Its result is a `CostedTaskView`, not a new semantic representation.

This separation permits the same portable or legal plan to be evaluated against measured kernels, analytical roofline models, a hardware simulator, or a network simulator without changing its digest. Out-of-domain evidence carries an explicit confidence/fallback status and cannot masquerade as a direct measurement.

## Joint scheduling and memory planning

The scheduler owns physical placement, implementation instances, ordering, synchronization, routing requirements, resource occupancy, and buffer allocation. Cross-target semantics enter the common coordination core; correctness semantics for dataflow, routes, issue slots, or micro-protocols enter typed target extensions. Memory planning is joint with scheduling because buffer interference depends on possible overlap.

At minimum, construction must enforce:

1. command dependencies are closed and acyclic;
2. queue order and signal/wait edges cannot deadlock;
3. every command has a legal implementation and placement;
4. buffer lifetimes cover all producers and consumers;
5. reused allocations do not overlap in any legal execution;
6. device, memory-space, engine, and link capacities are respected;
7. every cost used for optimization is present or handled by an explicit risk policy.

The algorithm may begin with deterministic list scheduling and interval allocation. Constraint programming or bounded refinement can be added later, but the current `ConcretePlanIR` contract remains experimental. Compatibility can be considered frozen only after queue-centric and non-queue-centric targets validate the common core.

## Timing and simulation projection

After the command DAG is verified, timing projection computes predicted intervals, contention, critical paths, and uncertainty. Simulator lowering maps the same commands and resources into discrete events.

Predicted timestamps are annotations, not readiness semantics. Removing them must leave a replayable dependency/queue program. This is what allows a new evidence revision to change expected duration without changing execution correctness.

If a target makes a cycle or slot a correctness constraint, it enters a typed target extension after binding rather than `TimingProjection`. The projection and trace are published with concrete, evidence, and policy digests in a `TimelineBundle`; see the [timeline staging path](../timeline-path.md) for the complete semantics.

## Implemented reference-binder passes

`stages/concrete_plan/passes.py` directly defines two deterministic contract-validation passes:

- `BindReferenceQueueTargetPass` produces a queue-centric `QueueScheduleExtension`;
- `BindReferenceSlotTargetPass` produces a non-queue `SlotDataflowExtension`.

Both preserve a 1:1 `PlanTask -> ConcreteCommand` identity, dependency topology, and buffer uses. Neither claims to be a production scheduler. Buffers use stable-order aligned linear allocation:

```text
offset_0 = 0
bound_i = align_up(offset_i, alignment_i)
offset_(i+1) = bound_i + size_i
```

Every command implementation and placement comes from explicit target/deployment bindings. The commit gate rebuilds the complete concrete plan with the same pure normalizer and requires equality of source-buffer identity, exact size, task lineage, command mapping, and the typed extension. Separately, relation invariants check portable-to-concrete buffer identity/capacity/alignment, dependency correspondence, buffer access, operation identity, and target ABI. The two passes map the same portable input into different typed extensions, proving that the common envelope does not assume queue-only targets.

This is a contract reference implementation, not a paper-derived performance heuristic. Source is `src/blueprinting/synthesizer/stages/concrete_plan/passes.py`; positive/negative, lineage, and deterministic replay tests are in `tests/synthesizer/test_reference_targets.py`.

## Machine lowering and artifact emission

A target plugin lowers verified concrete commands into its own `MachineIR` dialect. A virtual target should emit a deterministic replay package first; CUDA, LPU, and other hardware plugins can then add ABI-specific instructions and executable artifacts incrementally.

Every artifact manifest records target/deployment identity, portable/concrete/machine digests, analysis-engine and plugin revisions, runtime requirements, evidence fingerprint, and a detachable lineage map. Emission cannot introduce high-level scheduling decisions that are absent from the concrete plan.

## Checkpoints and diagnostics

Observers attach to legalization, concrete construction, timing, simulation, and machine verification boundaries. They check capability coverage, command correlation, cost coverage, resource conflicts, trace conservation, and ABI legality.

Diagnostics must preserve the candidate and source path:

```text
model operation -> distributed task -> portable task
  -> implementation candidate -> concrete command -> machine instruction
```

This path makes a hardware or simulator mismatch actionable at the owning layer instead of reducing it to one total-latency error.

## First implementation milestone

The next vertical slice should deliberately minimize target complexity:

1. implement one queue-centric and one non-queue-centric `VirtualTargetPlugin` to jointly validate the coordination core;
2. adapt the current `SystemProfile` behind a normalized estimate provider;
3. legalize the existing Transformer `PortablePlanIR`;
4. build a single-device or simple-TP `ConcretePlanIR` with explicit ordering, resources, buffers, and typed extensions;
5. derive a timing projection, discrete-event result, and `TimelineBundle`;
6. emit a replayable `MachineIR` package;
7. establish a command/event correspondence level between simulation and replay.

Only after this slice passes should CUDA or LPU-specific emission extend the plugin protocols. LPU can reach architecture and simulation maturity before hardware exists; executable maturity requires a stable ABI, runtime, and profiler contract.

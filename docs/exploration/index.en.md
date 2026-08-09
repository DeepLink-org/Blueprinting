# Why Blueprinting

Blueprinting is a hardware architecture exploration and simulation system for distributed AI workloads. It helps an architect describe candidate compute, memory, interconnect, and system organizations; map representative workloads onto them; estimate and simulate their behavior from traceable evidence; and compare the resulting design trade-offs before committing to hardware.

## The name is the product thesis

A blueprint is neither a loose idea nor finished silicon. It is a precise, inspectable design that is detailed enough to test, compare, revise, and hand to the next engineering stage.

That is the role of Blueprinting:

- sketch several hardware organizations without rewriting the workload model;
- make assumptions, constraints, and evidence revisions explicit;
- turn each sketch into a simulation-ready execution plan;
- expose why a design wins or loses, not only its predicted latency;
- preserve a path from early analytical models to detailed simulators and eventual hardware programs.

Typed formal models, verified derivations, and automated analyses make those blueprints comparable. IR and lowering are implementation techniques used inside that method, not the product identity.

## Mission

Given a workload suite and an architecture question, Blueprinting should answer:

> Which hardware organization best satisfies the target performance, memory, energy, area, cost, and programmability envelope—and which evidence supports that conclusion?

The system must support both **architecture exploration**, where hardware parameters are variables, and **deployment evaluation**, where an existing GPU, LPU, or other target is fixed. The former is the primary framing; the latter is one point in the same design space.

## Questions the system should answer

| Domain | Architecture question |
|---|---|
| Compute | How many matrix/vector engines, at which precision and utilization, are useful for this workload mix? |
| Memory | Which capacity, bandwidth, banking, and hierarchy remove the actual bottleneck rather than inflate peak numbers? |
| On-chip communication | Which NoC topology, link width, routing, and collective support sustain the chosen dataflow? |
| Scale-up/scale-out | How should chips, packages, nodes, and clusters connect as model and context sizes grow? |
| Mapping | Which sharding, recomputation, pipeline, placement, and scheduling choices fit the architecture? |
| System trade-offs | Where are the Pareto frontiers across latency, throughput, memory, energy, area, and cost? |
| Robustness | Does the conclusion survive uncertainty, workload variation, and evidence revision? |

A single end-to-end time cannot answer these questions. Blueprinting needs workload conservation, resource utilization, contention, critical paths, sensitivity, uncertainty, and provenance.

## Blueprint versus point estimator

A point estimator asks “how long will configuration X take?” A blueprinting system also asks:

1. Which design decisions created configuration X?
2. Which resources and dependencies make the result feasible?
3. What is the bottleneck and which parameter would move it?
4. Which result comes from exact workload facts, measurement, simulation, or assumption?
5. Which neighboring designs dominate or remain competitive under uncertainty?
6. Can the same plan be replayed or emitted when hardware becomes available?

For this reason, an experiment result is a versioned bundle of architecture, workload, mapping, evidence, simulation, metrics, and diagnostics—not a spreadsheet cell.

## Exploration loop

![Hardware architecture exploration loop](../assets/architecture/hardware-exploration-loop.svg)

The loop has five stages:

1. frame an architecture question with workloads, objectives, and constraints;
2. generate versioned candidate hardware blueprints;
3. construct legal workload mappings and execution plans;
4. resolve evidence and simulate at an appropriate fidelity;
5. compare bottlenecks, sensitivity, uncertainty, and Pareto results, then refine the candidates.

Measurements from simulators, prototypes, GPUs, LPUs, or eventual silicon create new evidence revisions. They improve later experiments without rewriting historical results.

## Product outputs

Blueprinting should produce:

- ranked and Pareto-filtered architecture blueprints;
- bottleneck and utilization reports by compute, memory, and communication resource;
- latency, throughput, memory, energy, area, and cost envelopes with uncertainty;
- sensitivity and what-if analysis showing which hardware changes matter;
- simulation traces correlated with workload and architecture identities;
- reproducible experiment manifests and evidence provenance;
- optionally, runtime configuration or target programs derived from the selected plan.

Program emission is useful, especially for LPU validation, but it is a downstream proof that the selected blueprint is realizable—not the definition of the project.

## Formal derivation as the technical foundation

Hardware experiments are easy to invalidate if every candidate uses a different workload formula, overlap assumption, or simulator schema. Blueprinting therefore expresses the problem as staged formal derivation:

```text
workload semantics
  -> distributed work and exact volumes
  -> portable mapping intent
  -> architecture-bound resources and schedule
  -> simulation trace and optional program
```

Each transition owns a declared decision class, preserves upstream workload meaning, discharges explicit constraints, and publishes a verifiable checkpoint. The result gives every candidate the same workload truth, stable lineage, explicit binding point, and checked execution plan.

The implementation uses IR to encode typed formal states, lowering to implement refinement, and passes to make analysis and transformation transactions observable. Those terms come from compiler engineering; the architectural method is formal modeling, derivation, verification, and automated analysis. Detailed contracts live under **Formal Analysis Foundations**.

## Current implementation boundary

Today the repository has a strong workload-accounting and formal-analysis foundation: Transformer training can be derived into a target-neutral portable plan, evidence-backed analytical estimates can be produced, and the Calculon experiment is reproducible.

It does not yet expose a first-class `ArchitectureBlueprint`, broad hardware design-space search, a connected discrete-event simulator, energy/area/cost models, or Pareto exploration. These are the next product-defining slices and are tracked in [implementation status](../project/status.md) and the [roadmap](../project/roadmap.md).

Continue with the [hardware design space](design-space.md), then follow the [exploration workflow](workflow.md).

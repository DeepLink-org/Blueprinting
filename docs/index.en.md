---
template: home.html
title: Blueprinting — Hardware Architecture Exploration
description: Evidence-driven hardware architecture exploration and simulation for distributed AI workloads.
hide:
  - toc
---

# Blueprinting

Blueprinting is an evidence-driven hardware architecture exploration and simulation system for distributed AI workloads. It turns candidate compute, memory, interconnect, and system designs into comparable simulation blueprints, helping architects understand bottlenecks and trade-offs before committing to hardware.

The name captures the product: create a precise architecture blueprint, map real workloads onto it, test it at several fidelity levels, and refine it from evidence.

![Blueprinting hardware architecture exploration loop](assets/architecture/hardware-exploration-loop.svg)

## What Blueprinting explores

| Design domain | Example questions |
|---|---|
| Compute substrate | Engine types/counts, precision, array shape, dataflow, useful utilization |
| Memory hierarchy | Capacity, bandwidth, banking, placement, reuse, lifetime pressure |
| Interconnect | NoC, die-to-die, scale-up/scale-out topology, routing, collectives, congestion |
| System organization | Chiplet/package/node/cluster composition, power, area, cost constraints |
| Workload mapping | Parallelism, tiling, fusion, recomputation, placement, scheduling, buffers |
| Architecture choice | Pareto frontiers, bottlenecks, sensitivity, uncertainty, robustness |

An answer is not only a latency number. It is a reproducible bundle of architecture parameters, workload facts, mapping decisions, evidence revisions, simulation traces, metrics, diagnostics, and claim boundaries.

## Exploration workflow

```text
architecture question + workload suite
  -> candidate hardware blueprints
  -> legal workload mappings
  -> evidence-backed analytical / network / hardware simulation
  -> bottleneck, sensitivity, and Pareto analysis
  -> measurement and calibration
  -> revised blueprint candidates
```

Existing GPUs, future LPUs, and other accelerators are all architecture candidates. Hardware can remain partially specified early in exploration and become progressively bound as capability, topology, simulator, and ABI details mature.

## Method: formal derivation and automated analysis

Blueprinting turns an architecture question into explicit formal objects: workload semantics, candidate hardware capabilities and resources, mapping decisions, evidence revisions, objectives, and constraints. It then derives progressively more concrete models while checking workload conservation, legality, capacity, dependency, lifetime, and provenance obligations at every boundary.

Automated analyses operate on these verified models to establish feasibility, resolve evidence-backed cost envelopes, identify bottlenecks and critical paths, evaluate sensitivity and uncertainty, and construct Pareto comparisons. Immutable checkpoints let analytical models, network or hardware simulators, and profiler observations check the result of each derivation without rewriting its semantics.

The implementation borrows typed IRs, staged lowering, transactional passes, and verifiers from compiler engineering where they are useful. In Blueprinting, these are techniques for representing and checking derivations—not a `Compiler` component and not the product definition.

## Current implementation

The current implementation has a strong foundation but does not yet deliver the complete exploration product:

- **Implemented:** typed Transformer training workload analysis, exact operation/byte/collective facts, target-neutral portable plans, derivation-level observability, evidence-backed analytical estimates, and Calculon calibration.
- **Contract foundation:** five typed formal representations for progressive mapping through concrete execution and optional target programs.
- **Planned product slices:** first-class architecture blueprints, design-space generation, target/resource binding, discrete-event simulation, network/hardware simulator adapters, energy/area/cost models, Pareto search, and profiler feedback.

See [implementation status](project/status.md) before treating a design contract as connected functionality.

## Start here

1. Read [Why Blueprinting](exploration/index.md) for the product thesis and name.
2. Study the [hardware design space](exploration/design-space.md).
3. Follow the [exploration workflow](exploration/workflow.md).
4. Review the [hardware architecture model](modeling/hardware.md) and [workload model](modeling/workload.md).
5. Use [performance evidence](design/performance/index.md) and [simulation](design/performance/simulation.md) when implementing fidelity layers.
6. Enter [Formal Analysis Foundations](design/index.md) for derivation, verification, representation, and transformation contracts.

## Project boundary

Blueprinting owns architecture descriptions, workload-to-hardware mapping, performance evidence resolution, simulation orchestration, design-space analysis, and reproducible experiment products. It integrates rather than reimplements detailed hardware/network simulators, kernels, collective libraries, RTL tools, drivers, and runtime stacks.

Optional program emission proves that a selected blueprint can eventually execute on LPU, GPU, or another target. It is one downstream realization of the same verified plan; architecture exploration and explainable analysis remain the primary product.

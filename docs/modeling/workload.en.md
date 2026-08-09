# Workload and Mapping Model

Hardware conclusions are only as credible as the workloads mapped onto the candidates. Blueprinting models a workload as structured computation, data movement, communication, dependencies, phases, and buffer lifetimes—not as one FLOP count or a model-name lookup.

## Workload semantics

A workload specification captures model operators, tensor shapes and dtypes, training/inference semantics, state, control/effects, batch and sequence regimes, optimizer or KV-cache behavior, and correctness-relevant dependencies.

These facts are architecture-independent. They describe what must happen, not which engine executes it or how long it takes.

## Scenario suites

Architecture exploration evaluates suites rather than isolated cases. A suite may include:

- training at several global/micro-batch and context sizes;
- inference prefill and decode across latency/throughput regimes;
- dense, MoE, multimodal, or communication-heavy variants;
- precision and sparsity modes;
- representative, stress, and projected future scenarios;
- scenario weights or service-level constraints.

A candidate that wins one headline benchmark but fails the expected workload distribution should not dominate the exploration.

## Exact workload facts

Progressive decomposition derives auditable quantities:

```text
operations by primitive and phase
read/write bytes by logical tensor
logical message bytes and collective semantics
dependencies and critical-path structure
persistent, checkpoint, temporary, and communication buffers
recomputation and recommunication work
```

These quantities are conserved across architecture candidates. Target-specific implementation may change physical traffic or executed operations only through an explicit, source-linked transformation.

## Mapping decisions

Mapping connects architecture-independent work to a candidate blueprint. It selects:

- tensor/data/pipeline/expert parallel decomposition;
- tiling, fusion, recomputation, and layout;
- implementation and engine assignment;
- physical placement and communication route;
- queue order, synchronization, and overlap opportunities;
- buffer memory level, allocation, reuse, and movement.

Mapping is part of the candidate evaluation because a strong architecture with a poor mapping may look weak. Blueprinting must compare architecture and mapping jointly while preserving which decision came from each layer.

## Workload versus implementation work

The model distinguishes semantic work from implementation overhead. A logical all-reduce has source message volume and reduction semantics; a ring or tree algorithm adds physical transfers and local reductions. A logical matrix multiply has semantic operations; padding, packing, or target-specific decomposition adds implementation work.

This separation allows a hardware feature to show its actual benefit and prevents target overhead from contaminating the common workload baseline.

## Temporal and memory structure

Peak memory and exposed communication cannot be inferred from totals alone. The mapping model therefore preserves dependencies, buffer producers/consumers, lifetimes, queues, and synchronization until architecture-bound simulation.

Early analytical stages may use safe bounds. Final claims about overlap, contention, or memory reuse require a concrete resource plan or a declared approximation with uncertainty.

## Profiler and simulator correlation

Stable lineage connects model operations to distributed tasks, portable work, concrete commands, simulator events, and runtime instructions. Different layers support different checks:

| Layer | Check |
|---|---|
| Workload semantics | shape, type, effect, and numerical/reference behavior |
| Distributed work | shard reconstruction, rank/mesh consistency, communication volume |
| Portable mapping | operation/byte conservation, capabilities, abstract lifetimes |
| Concrete mapping | placement, queue, synchronization, buffer and capacity legality |
| Simulation/runtime | event correlation, utilization, duration, counters, missing work |

Profiler observations calibrate evidence; they do not redefine workload semantics.

## Current Transformer coverage

The implemented slice imports typed decoder-only Transformer training specifications and decomposes one local tensor-parallel block into forward, recomputation, activation-gradient, weight-gradient, optimizer, and collective invocations. It derives exact operations and bytes and produces `PortablePlanIR`.

Full-model PP/DP graphs, complete intermediate-buffer lifetimes, inference prefill/decode, MoE, and general framework importers remain planned. The [Transformer workload-derivation reference](../design/passes/transformer.md) documents the current algorithm and limitations.

## Contract with the hardware model

The workload side supplies exact work, logical dependencies, abstract resource needs, and capability requirements. The hardware side supplies resources, capability implementations, topology, physical constraints, and evidence. The architecture-binding step constructs a legal mapped plan between them.

Neither side is allowed to smuggle in the other's facts. This boundary enables one workload suite to compare many blueprints and one blueprint to compare many workloads without duplicating semantics.

## Validation strategy

Validation proceeds from exact facts to system behavior:

1. unit-test primitive operation and byte algebra;
2. verify conservation across distribution and mapping;
3. compare aggregate work and memory with independent references;
4. compare schedule composition and bottlenecks with analytical tools;
5. correlate simulator and runtime events with the same command identities;
6. calibrate target behavior only after workload discrepancies are resolved.

The [Calculon experiment](../experiments/calculon-calibration.md) currently covers steps 1–4 for the selected Transformer training cases.

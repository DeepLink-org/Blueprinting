# Model and Distributed IR

`ModelIR` and `DistributedTaskIR` describe what the workload means and how it is logically distributed. Both are target-neutral and must remain valid without an accelerator profile or deployment.

## ModelIR

### Semantic domain

`ModelIR` represents tensor and scalar values, typed operations, regions, dataflow, state roles, and effects. It can represent training and inference semantics, symbolic dimensions, parameters, activations, optimizer state, and KV-cache state where known.

### Core entities

```text
ModelIR
├── values: ModelValue[]
├── operations: ModelOperation[]
├── regions: SemanticRegion[]
├── inputs / outputs
├── extensions
└── header
```

Operations explicitly reference input and output values. Effects such as parameter update, random state, mutation, allocation, and external call are typed rather than inferred from operation names.

### Forbidden information

`ModelIR` cannot contain TP/PP/DP placement, logical or physical devices, target kernels, estimated duration, queues, memory addresses, or target performance hints with semantic force.

### Well-formedness

The verifier checks unique IDs, definition/use closure, type compatibility, region ownership, input/output reachability, effect ordering, symbolic-domain validity, and deterministic extensions.

### Transformations

Allowed passes include import, shape/type inference, canonicalization, decomposition, semantic fusion, constant reasoning, and bounded equality exploration. Every transformation preserves observable model semantics or declares an explicit approximation contract.

## DistributedTaskIR

### Semantic domain

`DistributedTaskIR` represents a logical distributed program over virtual meshes and ranks. It says which rank owns each shard, which local task executes, which collective or P2P operation connects ranks, and how effects and state are ordered.

### Core entities

```text
DistributedTaskIR
├── meshes: LogicalMesh[]
├── ranks: LogicalRank[]
├── values: DistributedValue[]
├── tasks: ComputeTask | CollectiveTask | P2PTask | ReshardTask
├── dependencies
├── memory_facts
└── lineage to ModelIR
```

Communication operations retain logical semantics: participants, collective kind, reduction, tensor/value relation, and exact logical message bytes where derivable.

### Forbidden information

This IR cannot name CUDA, ROCm, LPU, NCCL, physical routes, physical device IDs, queue assignments, absolute timestamps, or target-specific latency.

### Well-formedness

The verifier checks mesh and rank membership, shard reconstruction, ownership, communication matching, message-volume conservation, reshard completeness, cross-rank dependency closure, and effect order.

### Transformations

Distribution passes choose sharding and replication, create logical meshes, clone local work, insert collective and P2P tasks, and establish cross-rank dependencies. They may introduce recomputation structurally, but cannot use time multipliers or target cost to redefine work.

## Boundary between the layers

`ModelIR -> DistributedTaskIR` requires a strategy or candidate. On success, every distributed value has an ownership rule, every cross-rank use has communication, and projection over ranks reconstructs the source semantics.

```text
Model linear(x, w)
  + TP strategy
    -> rank-local matmul tasks
    -> logical AllReduce
    -> reconstructed model result
```

## Current implementation

The current Transformer frontend emits one coarse decoder-training `ModelIR` operation. `DistributeTransformerTrainingPass` expands it into typed primitive invocations and a local TP task DAG with explicit collectives, recomputation phases, and aggregate block memory facts.

The current graph models one local Transformer block conservatively. Full-model PP/DP graphs, cross-stage values, richer topology, inference regions, and KV-cache distribution remain planned.

## Profiler and verification checkpoint

At `ModelIR`, observers can check shapes, types, effects, and optional numerical references. At `DistributedTaskIR`, they can check shard reconstruction, per-rank work, communication matching, and volume conservation. Neither checkpoint may introduce target duration.

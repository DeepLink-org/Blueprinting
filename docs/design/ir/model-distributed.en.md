# Model and Distributed IR

`ModelIR` and `DistributedTaskIR` describe what the workload means and how it is logically distributed. Both are target-neutral and must remain valid without an accelerator profile or deployment.

## ModelIR

### Semantic domain

`ModelIR` represents tensor values, typed operations, explicit dataflow, state roles, and effects. It can represent training and inference semantics, symbolic dimensions, parameters, activations, optimizer state, and KV-cache state where known. Semantic regions remain a future contract rather than a current field.

### Core entities

```text
ModelIR
├── values: ModelValue[]
├── operations: ModelOperation[]
├── inputs / outputs
├── attributes
└── header
```

Operations explicitly reference input and output values. Effects such as parameter update, random state, mutation, allocation, and external call are typed rather than inferred from operation names.

### Forbidden information

`ModelIR` cannot contain TP/PP/DP placement, logical or physical devices, target kernels, estimated duration, queues, memory addresses, or target performance hints with semantic force.

### Well-formedness

The current structural verifier checks unique IDs, definition/use closure, input/output references, SSA single definition, target-dialect exclusion, and reserved attributes. Cross-operation type/shape compatibility, region ownership, complete effect ordering, and numerical references still require independent checkers or future verifier work.

### Transformations

Allowed passes include import, shape/type inference, canonicalization, decomposition, semantic fusion, constant reasoning, and bounded equality exploration. Every transformation preserves observable model semantics or declares an explicit approximation contract.

## DistributedTaskIR

### Semantic domain

`DistributedTaskIR` represents a logical distributed program over virtual meshes and ranks. It says which rank owns each shard, which local task executes, which collective or P2P operation connects ranks, and how effects and state are ordered.

### Core entities

```text
DistributedTaskIR
├── mesh: LogicalMesh
├── values: DistributedValue[]
├── tasks: DistributedTask[]
├── inputs / outputs
├── attributes
└── lineage to ModelIR
```

Communication operations retain logical semantics: participants, collective kind, reduction, tensor/value relation, and exact logical message bytes where derivable.

### Forbidden information

This IR cannot name CUDA, ROCm, LPU, NCCL, physical routes, physical device IDs, queue assignments, absolute timestamps, or target-specific latency.

### Well-formedness

The current structural verifier checks mesh and rank membership, sharding rank and axes, ownership, task/value references, DAG order, and local communication-metadata legality. Shard reconstruction, collective matching, message-volume conservation, reshard completeness, and cross-rank semantic closure remain target capabilities for observers and conformance checkers.

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

The current training graph models one local Transformer block conservatively. Static inference already implements separate prefill/decode phases, KV-cache state and values, and local TP distribution; full-model PP/DP graphs, cross-stage values, richer topology, and inference regions remain planned.

## Profiler and verification checkpoint

At `ModelIR`, observers can add shape, type, effect, and optional numerical-reference checks. At `DistributedTaskIR`, they can add shard reconstruction, per-rank work, communication matching, and volume-conservation checks; these capabilities must not be inferred from the current structural verifier. Neither checkpoint may introduce target duration.

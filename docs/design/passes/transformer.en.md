# Transformer Workload Derivation

The implemented Transformer slice is deliberately narrow and auditable: it imports one typed decoder-training workload, formally derives one local tensor-parallel block, and preserves exact work in a target-neutral portable plan. It validates the first half of the analysis architecture without pretending that target scheduling already exists.

![Implemented Transformer workload-derivation path](../../assets/architecture/implemented-compile-path.svg)

## Scope and boundary

The production path is:

```text
TransformerModelSpec + TransformerExecutionSpec
  -> ModelIR
  -> DistributeTransformerTrainingPass
  -> DistributedTaskIR
  -> PlanTransformerTrainingPass
  -> PortablePlanIR
```

The red boundary in the figure is intentional. `HardwareProfile` is consumed only by a derived estimate after `PortablePlanIR`; it is not an implicit target binding and it does not make `ConcretePlanIR` available.

This slice currently models decoder-only training at block scope. Full-model PP/DP task graphs, inference prefill/decode, intermediate-buffer lifetimes, target legalization, and physical scheduling remain subsequent work.

## Typed semantic import

`TransformerModelSpec` owns dimensions and model semantics. `TransformerExecutionSpec` owns micro-batching, TP/PP/DP, recomputation, datatype, and tensor-parallel communication mode. `compilation_session_for()` turns those execution choices into explicit workload and strategy bindings.

The importer rejects invalid dimensions, head divisibility, parallel topology, and inconsistent workload facts before a pass runs. `build_transformer_model_ir()` then creates a coarse, target-neutral `transformer.decoder_training` operation. No target name, peak rate, kernel ID, or latency enters this snapshot.

## Static workload derivation

`compile_transformer_block()` decomposes a block into typed `PrimitiveInvocation` records. Each invocation has a phase, engine class, exact operations, exact read/write bytes, and—when applicable—collective kind and logical message bytes.

The analysis follows data dependencies rather than fitted ratios. For a linear layer `Y[M,K] = X[M,N] x W[N,K]`, forward, activation-gradient, and weight-gradient work are three explicit matrix multiplications. Attention, normalization, activation, dropout, residual, and optimizer work are represented separately.

Recomputation is also structural. Full recomputation clones the required forward invocations; selective recomputation clones only the selected attention path. Sequence-parallel recommunication is a distinct collective invocation. Consequently every added operation and byte remains attributable to a semantic cause.

## Distribution derivation

`DistributeTransformerTrainingPass` consumes `ModelIR` plus workload and strategy bindings and introduces:

- a logical TP mesh and logical ranks;
- forward, recompute, backward, optimizer, and recommunication tasks;
- explicit all-reduce, reduce-scatter, or all-gather collectives;
- distributed boundary values and sharding;
- stable lineage from every task and value to its model source.

The current dependency chain is conservative and serial within the local block. That is a correctness baseline, not a claim that no target can overlap work. Physical queues, routes, collective algorithms, and overlap are forbidden at this layer because they require target and deployment knowledge.

The pass must preserve workload semantics and satisfy these checks:

1. the logical mesh size agrees with the strategy;
2. every rank and dependency resolves;
3. shard specifications reconstruct the logical boundary tensor;
4. collective participants, reduction semantics, and message volumes are well formed;
5. the output records the source `ModelIR` digest.

## Portable-plan derivation

`PlanTransformerTrainingPass` converts each distributed task into a `PlanTask`. It retains `WorkloadFacts`, declares abstract resource demand, creates capability-based implementation requirements, assigns logical concurrency groups, and introduces boundary buffers and objectives.

The pass may say that a task needs `matrix-multiply`, `vector-elementwise`, or a collective capability. It may not select a CUDA kernel, LPU opcode, physical device, memory bank, queue, or duration. Those are decisions of the portable-to-concrete gate.

The resulting plan is suitable for exact-work comparison and target-neutral pruning. It is not yet a complete execution plan: the current implementation has only boundary buffers and does not encode all intermediate lifetimes.

## Checkpoints and profiler integration

Both passes run through `PassManager`. After output verification, a `PassCheckpoint` exposes the immutable IR, schema, digest, pass record, session fingerprint, and lineage to synchronous observers.

Useful stage checks include:

| Checkpoint | Independent checks |
|---|---|
| `DistributedTaskIR` | primitive count, phase coverage, mesh/rank consistency, collective volume, recompute expansion |
| `PortablePlanIR` | operations and byte conservation, resource coverage, buffer legality, absence of target fields |

An observer may compare facts with a framework trace or reference model and reject the transition. It may not mutate the representation or write undeclared analyses. Host-side transformation duration is recorded separately from predicted workload time.

## Failure model and diagnostics

Typed inconsistency is reported at the earliest boundary: missing strategy bindings, mismatched workload/execution facts, unsupported semantic operation, malformed invocation, or verifier failure. A failed pass publishes neither its output snapshot nor preserved/produced analyses.

The derivation does not compensate for a discrepancy by reading a reference latency or attaching a case-specific coefficient. A disagreement is localized to semantic import, work derivation, distribution, cost evidence, or schedule composition and fixed at that boundary.

## Implementation map

| Concern | Source | Tests |
|---|---|---|
| Typed Transformer specifications | `src/blueprinting/compiler/models/transformer.py` | binding and calibration tests |
| Workload algebra | `src/blueprinting/compiler/analysis/transformer_workload.py` | `tests/compiler/test_calculon_calibration.py` |
| Two derivation passes | `src/blueprinting/compiler/lowering/transformer.py` | canonical representation and calibration tests |
| Transaction/checkpoints | `src/blueprinting/compiler/passes/base.py` | `tests/compiler/test_pass_manager.py` |
| Evidence-derived estimates | `src/blueprinting/compiler/analysis/cost_model.py` | calibration tests |

The [Calculon calibration experiment](../../experiments/calculon-calibration.md) is the end-to-end audit of this implemented slice.

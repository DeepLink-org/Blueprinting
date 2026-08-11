# Typed Transformer Parallel Strategies

This page defines how Blueprinting maps Megatron-style tensor, pipeline, and data parallelism into formal-analysis values. These strategies are target-neutral logical choices, not physical GPU placements or NCCL algorithm selections.

## Algebraic data model

`TransformerTrainingMappingSpec` stores `parallelism` as its canonical pattern-matchable structure. `from_mapping` is the boundary adapter for flat external configuration keys; degree-named properties are derived projections and are not a second serialized contract:

```text
TransformerTrainingParallelism
├── tensor: TensorParallel(degree, communication)
├── pipeline: PipelineParallel(degree, schedule)
├── data: DataParallel(degree, optimizer_sharding)
└── recompute: RecomputePolicy

PipelineSchedule =
    SingleStage
  | OneForwardOneBackward
  | InterleavedOneForwardOneBackward(virtual_stages)
  | ForwardOnly
```

Training and inference use different top-level records. Inference uses `ReplicaParallel` so independent serving replicas are not mislabeled as training data parallelism with gradient synchronization.

Passes destructure the strategy with structural pattern matching:

```python
match mapping.parallelism:
    case TransformerTrainingParallelism(
        tensor=TensorParallel(degree=tp, communication=communication),
        pipeline=PipelineParallel(degree=pp, schedule=schedule),
        data=DataParallel(degree=dp),
        recompute=recompute,
    ):
        ...
```

Adding a schedule constructor makes omitted interpreters and lowerings visible to the mypy `exhaustive-match` gate.

## Composition and divisibility laws

Let the TP, PP, and DP degrees be `t`, `p`, and `d`:

```text
world_size = t × p × d
local_batch = global_batch / d
microbatch_count = global_batch / (d × microbatch_size)
blocks_per_pipeline_stage = transformer_blocks / p
blocks_per_virtual_chunk = transformer_blocks / (p × v)
```

The current static planner requires every division above to be integral and requires `v` to divide the block count of each physical pipeline stage. It does not silently repair an invalid strategy with padding or uneven stages.

## Tensor-parallel mapping

For `Y[M,K] = X[M,N] W[N,K]`, partitioning the output or contraction dimension with degree `t` gives ideal local GEMM work:

```text
F_local = 2 M N K / t
```

Column- and row-parallel linear layers alternate inside the block so selected intermediates remain sharded. Reconstructing a semantic boundary produces an explicit all-reduce, or reduce-scatter plus all-gather for sequence-parallel boundaries. RS+AG requires:

```text
sequence_length mod t = 0
local_sequence = sequence_length / t
```

`DistributedTaskIR` currently materializes one local TP block, so its `LogicalMesh` contains only the `tp` axis. Typed program semantics retain PP/DP choices and iteration analysis composes them later. A full-model cross-stage P2P task graph is not implemented and cannot be inferred from the local graph.

## Pipeline-schedule mapping

| Constructor | Meaning | Current consumer |
|---|---|---|
| `SingleStage` | `p = 1`; no pipeline | training / inference |
| `OneForwardOneBackward` | synchronous non-interleaved 1F1B | training iteration analysis |
| `InterleavedOneForwardOneBackward(v)` | `v` virtual chunks per physical stage | training iteration analysis |
| `ForwardOnly` | static inference forward pipeline | inference mapping contract |

The current 1F1B composition uses forward and backward critical paths derived from the portable block plan:

```text
T_chunk = T_forward_chunk + T_backward_chunk
n_bubble = (p - 1) + extra_interleaving_bubbles
T_bubble = n_bubble × T_chunk - T_imbalance_correction
```

When `microbatch_count mod p != 0`:

```text
extra_interleaving_bubbles = (v - 1) × (p - microbatch_count mod p)
```

This is an explicit analytical schedule contract, not an execution fact in `PortablePlanIR`. A future concrete scheduler must re-establish the schedule with a command DAG, P2P dependencies, and resource conflicts.

## Primary references and scope

- Shoeybi et al., [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053): column/row intra-layer Transformer tensor parallelism.
- Narayanan et al., [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473): TP × PP × DP composition, 1F1B, and interleaved pipeline scheduling.
- Huang et al., [GPipe](https://arxiv.org/abs/1811.06965): the microbatch pipeline and basic partition/bubble model.
- Korthikanti et al., [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198): sequence parallelism and selective activation recomputation.

The papers motivate algorithms and formulas; they do not prove this implementation correct. Repository tests independently constrain divisibility, work conservation, lineage, the Calculon oracle, and regression digests.

## Source and tests

| Concern | Location |
|---|---|
| typed strategy and schedule ADT | `src/blueprinting/mapping/transformer.py` |
| TP block work algebra | `src/blueprinting/synthesizer/dialects/transformer/training.py` |
| pattern-matching derivation | `src/blueprinting/synthesizer/dialects/transformer/training_derivation.py` |
| DistributedTask pass contract | `src/blueprinting/synthesizer/stages/distributed/passes.py` |
| PortablePlan pass contract | `src/blueprinting/synthesizer/stages/portable_plan/passes.py` |
| mapping/ADT tests | `tests/analysis/test_domain_contracts.py` |
| work and lineage tests | `tests/synthesizer/test_transformer_training.py` |
| Calculon alignment gate | `tests/validation/test_calculon.py`, `tests/regression/` |

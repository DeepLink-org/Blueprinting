# Transformer 并行策略的类型化表示

本页定义 Blueprinting 如何把 Megatron 风格的 tensor、pipeline、data parallelism 映射到形式化分析对象。这里的策略是 target-neutral 的逻辑选择，不是物理 GPU placement，也不是 NCCL algorithm 选择。

## 代数数据结构

`TransformerTrainingMappingSpec` 直接把 `parallelism` 保存为 canonical、可模式匹配的结构。`from_mapping` 是扁平外部配置 key 的 boundary adapter；degree 命名的 property 只是 derived projection，不是第二套 serialized contract：

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

训练与推理使用不同的顶层 record。推理使用 `ReplicaParallel`，避免把相互独立的 serving replica 错叫成带 gradient synchronization 的 training data parallelism。

Pass 通过结构化模式匹配解构策略：

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

增加新的 schedule constructor 后，mypy 的 `exhaustive-match` gate 会迫使 interpreter、lowering 和文档处理新分支。

## 组合与整除约束

设 TP、PP、DP degree 分别为 `t`、`p`、`d`：

```text
world_size = t × p × d
local_batch = global_batch / d
microbatch_count = global_batch / (d × microbatch_size)
blocks_per_pipeline_stage = transformer_blocks / p
blocks_per_virtual_chunk = transformer_blocks / (p × v)
```

当前静态 planner 要求上述除法均为整数，并要求 `v` 整除每个 physical pipeline stage 的 block 数。它不会通过 padding 或不均匀 stage 隐式修复非法配置。

## Tensor Parallel 映射

对于矩阵乘 `Y[M,K] = X[M,N] W[N,K]`，若被切分的输出或收缩维 degree 为 `t`，每个 TP rank 的理想局部 GEMM work 为：

```text
F_local = 2 M N K / t
```

column-parallel 与 row-parallel linear 在 block 内交替，使部分中间结果保持 shard；需要重建语义边界时显式产生 all-reduce，或使用 reduce-scatter + all-gather 表达 sequence-parallel 边界。RS+AG 模式下 sequence dimension 必须满足：

```text
sequence_length mod t = 0
local_sequence = sequence_length / t
```

`DistributedTaskIR` 当前物化一个 local TP block，因此其 `LogicalMesh` 只有 `tp` axis。PP/DP 选择仍以 typed program semantic 保留，并在 iteration analysis 中组合。把完整模型展开成 cross-stage P2P task graph 是尚未实现的下一层能力，不能从当前 local block graph 推断出来。

## Pipeline Schedule 映射

当前 schedule ADT 区分：

| Constructor | 含义 | 当前消费者 |
|---|---|---|
| `SingleStage` | `p = 1`，无 pipeline | training / inference |
| `OneForwardOneBackward` | synchronous non-interleaved 1F1B | training iteration analysis |
| `InterleavedOneForwardOneBackward(v)` | 每个 physical stage 有 `v` 个 virtual chunk | training iteration analysis |
| `ForwardOnly` | 静态 inference forward pipeline | inference mapping contract |

Blueprinting 当前的 1F1B composition 使用从 portable block plan 得到的 forward/backward critical path：

```text
T_chunk = T_forward_chunk + T_backward_chunk
n_bubble = (p - 1) + extra_interleaving_bubbles
T_bubble = n_bubble × T_chunk - T_imbalance_correction
```

当 `microbatch_count mod p != 0` 时：

```text
extra_interleaving_bubbles = (v - 1) × (p - microbatch_count mod p)
```

这是一条明确的 analytical schedule contract，不是 `PortablePlanIR` 的 execution fact。未来 concrete scheduler 必须以 command DAG、P2P dependency 和 resource conflict 重新证明 schedule。

## 论文依据与适用边界

- Shoeybi et al., [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)：Transformer intra-layer tensor parallel 的 column/row partition。
- Narayanan et al., [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473)：TP × PP × DP 组合、1F1B 与 interleaved pipeline schedule。
- Huang et al., [GPipe](https://arxiv.org/abs/1811.06965)：microbatch pipeline 与 bubble/partition 的基本模型。
- Korthikanti et al., [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198)：sequence parallelism 与 selective activation recomputation。

论文提供算法与建模依据，不自动证明本实现正确。对应的整除、work conservation、lineage、Calculon oracle 和 regression digest 均由仓库测试独立约束。

## 源码与测试

| 内容 | 位置 |
|---|---|
| typed strategy 与 schedule ADT | `src/blueprinting/mapping/transformer.py` |
| TP block work algebra | `src/blueprinting/synthesizer/dialects/transformer/training.py` |
| pattern-matching derivation | `src/blueprinting/synthesizer/dialects/transformer/training_derivation.py` |
| DistributedTask pass contract | `src/blueprinting/synthesizer/stages/distributed/passes.py` |
| PortablePlan pass contract | `src/blueprinting/synthesizer/stages/portable_plan/passes.py` |
| mapping/ADT tests | `tests/analysis/test_domain_contracts.py` |
| work与lineage tests | `tests/synthesizer/test_transformer_training.py` |
| Calculon alignment gate | `tests/validation/test_calculon.py`, `tests/regression/` |

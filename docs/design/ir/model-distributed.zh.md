# 模型与分布式 IR

`ModelIR` 和 `DistributedTaskIR` 描述 workload 的含义以及它如何在逻辑上分布。两者都保持 target-neutral，并且必须在没有 accelerator profile 或 deployment 时仍然有效。

## ModelIR

### 语义域

`ModelIR` 表示 tensor value、typed operation、显式 dataflow、state role 和 effect。它可以表示 training/inference semantic、symbolic dimension、parameter、activation、optimizer state，以及已知情况下的 KV-cache state。Semantic region 仍是后续 contract，不是当前字段。

### 核心 Entity

```text
ModelIR
├── values: ModelValue[]
├── operations: ModelOperation[]
├── inputs / outputs
├── attributes
└── header
```

Operation 显式引用 input/output value。Parameter update、random state、mutation、allocation 和 external call 等 effect 使用 typed representation，而不是从 operation name 推断。

### 禁止的信息

`ModelIR` 不能包含 TP/PP/DP placement、logical/physical device、target kernel、estimated duration、queue、memory address，或者具有语义效力的 target performance hint。

### Well-formedness

当前 structural verifier 检查 unique ID、definition/use closure、input/output reference、SSA single definition、target dialect exclusion 与 reserved attributes。Operation 间 type/shape compatibility、region ownership、完整 effect ordering 与 numerical reference 仍需独立 checker 或后续 verifier 实现。

### Transformation

允许的 Pass 包括 import、shape/type inference、canonicalization、decomposition、semantic fusion、constant reasoning 和 bounded equality exploration。每次 transformation 保持 observable model semantic，或者声明显式 approximation contract。

## DistributedTaskIR

### 语义域

`DistributedTaskIR` 表示运行在 virtual mesh/rank 上的 logical distributed program。它说明每个 shard 由哪个 rank 拥有、执行什么 local task、哪些 collective/P2P 连接 rank，以及 effect/state 如何排序。

### 核心 Entity

```text
DistributedTaskIR
├── mesh: LogicalMesh
├── values: DistributedValue[]
├── tasks: DistributedTask[]
├── inputs / outputs
├── attributes
└── lineage to ModelIR
```

Communication operation 保留 logical semantic：participant、collective kind、reduction、tensor/value relation，以及可推导时的 exact logical message bytes。

### 禁止的信息

这一层不能命名 CUDA、ROCm、LPU、NCCL、physical route、physical device ID、queue assignment、absolute timestamp 或 target-specific latency。

### Well-formedness

当前 structural verifier 检查 mesh/rank membership、sharding rank/axis、ownership、task/value reference、DAG 顺序和 communication metadata 的局部合法性。Shard reconstruction、collective matching、message-volume conservation、reshard completeness 与 cross-rank semantic closure 仍是 observer/conformance checker 的目标能力。

### Transformation

Distribution pass 选择 sharding/replication，创建 logical mesh，克隆 local work，插入 collective/P2P task，并建立 cross-rank dependency。它可以结构化引入 recomputation，但不能通过 time multiplier 或 target cost 重新定义 work。

## 两层之间的边界

`ModelIR -> DistributedTaskIR` 需要 strategy 或 candidate。成功后，每个 distributed value 都有 ownership rule，每个 cross-rank use 都有 communication，并且所有 rank 的 projection 可以重建 source semantic。

```text
Model linear(x, w)
  + TP strategy
    -> rank-local matmul tasks
    -> logical AllReduce
    -> reconstructed model result
```

## 当前实现

当前 Transformer frontend 输出一个 coarse decoder-training `ModelIR` operation。`DistributeTransformerTrainingPass` 把它展开为 typed primitive invocation 和 local TP task DAG，并包含显式 collective、recomputation phase 和 aggregate block memory fact。

当前 training graph 以保守方式建模一个 local Transformer block。静态 inference 已实现独立 prefill/decode phase、KV-cache state/value 与 local TP distribution；full-model PP/DP graph、cross-stage value、更丰富 topology 和 inference region 仍在规划中。

## Profiler 与 Verification Checkpoint

在 `ModelIR`，observer 可以补充检查 shape、type、effect 和可选 numerical reference。在 `DistributedTaskIR`，observer 可以补充检查 shard reconstruction、per-rank work、communication matching 和 volume conservation；这些能力不能从当前 structural verifier 自动推断。两个 checkpoint 都不能引入 target duration。

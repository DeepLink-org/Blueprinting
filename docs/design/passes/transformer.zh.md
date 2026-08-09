# Transformer 工作负载推导

当前已经实现的 Transformer 纵向切片刻意保持窄而可审计：它导入一个强类型 decoder training 工作负载，形式化推导一个本地 tensor-parallel block，并把精确工作量保存在 target-neutral portable plan 中。这条路径验证了分析架构的前半段，但不会把尚未完成的 target scheduling 描述成已实现能力。

![已经实现的 Transformer 工作负载推导路径](../../assets/architecture/implemented-compile-path.svg)

## 范围与边界

当前 production path 是：

```text
TransformerModelSpec + TransformerExecutionSpec
  -> ModelIR
  -> DistributeTransformerTrainingPass
  -> DistributedTaskIR
  -> PlanTransformerTrainingPass
  -> PortablePlanIR
```

图中的红色边界是有意保留的。`HardwareProfile` 只在 `PortablePlanIR` 之后被 derived estimate 消费；它既不是隐式 target binding，也不意味着系统已经能够生成 `ConcretePlanIR`。

当前切片只覆盖 decoder-only training 的 block scope。完整模型的 PP/DP task graph、推理 prefill/decode、中间 buffer lifetime、target legalization 和物理调度仍属于后续工作。

## 强类型语义导入

`TransformerModelSpec` 拥有模型维度与语义，`TransformerExecutionSpec` 拥有 micro-batching、TP/PP/DP、重计算、数据类型和 tensor-parallel 通信模式。`compilation_session_for()` 把这些执行选择转换成显式 workload 与 strategy binding。

Importer 会在 pass 运行前拒绝非法维度、head 不可整除、错误并行拓扑以及互相矛盾的 workload facts。随后 `build_transformer_model_ir()` 创建一个粗粒度、target-neutral 的 `transformer.decoder_training` operation。这个 snapshot 中不存在 target 名称、峰值性能、kernel ID 或 latency。

## 静态工作量推导

`compile_transformer_block()` 把一个 block 分解为强类型 `PrimitiveInvocation`。每个 invocation 都带有 phase、engine class、精确 operations、精确 read/write bytes；如果它是 collective，还会带有 collective kind 和逻辑 message bytes。

分析遵循数据依赖，而不是拟合比例。对于线性层 `Y[M,K] = X[M,N] x W[N,K]`，forward、activation-gradient 和 weight-gradient 是三个显式矩阵乘。Attention、normalization、activation、dropout、residual 与 optimizer work 也分别表示。

重计算同样是结构语义。Full recomputation 会克隆所需的 forward invocation；selective recomputation 只克隆被选择的 attention path。Sequence-parallel recommunication 是独立的 collective invocation。因此，每一项新增 operation 与 byte 都能追溯到明确的语义原因。

## 分布式推导

`DistributeTransformerTrainingPass` 消费 `ModelIR`、workload binding 与 strategy binding，并引入：

- 逻辑 TP mesh 与逻辑 rank；
- forward、recompute、backward、optimizer 和 recommunication task；
- 显式 all-reduce、reduce-scatter 或 all-gather collective；
- 分布式边界 value 与 sharding；
- 从每个 task/value 回到 model source 的稳定 lineage。

当前 block 内依赖采用保守串行链。这是正确性基线，并不意味着 target 不能并发执行。Physical queue、route、collective algorithm 与 overlap 在本层被禁止，因为它们依赖 target 和 deployment 信息。

Pass 必须保持工作负载语义，并满足以下检查：

1. logical mesh 大小与 strategy 一致；
2. 每个 rank 与 dependency 都能解析；
3. shard specification 能重构逻辑边界 tensor；
4. collective participant、reduction semantic 和 message volume 合法；
5. 输出记录 source `ModelIR` digest。

## 可移植计划推导

`PlanTransformerTrainingPass` 把每个 distributed task 转换成 `PlanTask`。它保留 `WorkloadFacts`，声明抽象 resource demand，创建基于 capability 的 implementation requirement，分配逻辑 concurrency group，并引入 boundary buffer 与 objective。

Pass 可以声明 task 需要 `matrix-multiply`、`vector-elementwise` 或某种 collective capability，但不能选择 CUDA kernel、LPU opcode、physical device、memory bank、queue 或 duration。这些决策属于 portable-to-concrete gate。

得到的 plan 可用于精确工作量对比和 target-neutral pruning，但还不是完整执行计划：当前实现只有 boundary buffer，尚未编码全部 intermediate lifetime。

## Checkpoint 与 Profiler 联动

两个 pass 都通过 `PassManager` 执行。输出完成验证后，`PassCheckpoint` 会把 immutable IR、schema、digest、pass record、session fingerprint 与 lineage 暴露给同步 observer。

各阶段适合执行的独立检查包括：

| Checkpoint | 独立检查 |
|---|---|
| `DistributedTaskIR` | primitive 数量、phase 覆盖、mesh/rank 一致性、collective volume、recompute expansion |
| `PortablePlanIR` | operation/byte 守恒、resource 覆盖、buffer 合法性、target field 不得出现 |

Observer 可以把这些 facts 与 framework trace 或 reference model 对比并拒绝 transition，但不能修改 representation 或写入未声明 analysis。Host-side transformation duration 与预测的 workload time 分开记录。

## 失败模型与诊断

强类型不一致会在最早边界报告：缺失 strategy binding、workload/execution facts 不一致、不支持的 semantic operation、错误 invocation 或 verifier failure。失败的 pass 不会发布 output snapshot，也不会发布 preserved/produced analysis。

推导不会通过读取参考 latency 或增加 case-specific coefficient 来掩盖差异。任何不一致都必须定位到 semantic import、work derivation、distribution、cost evidence 或 schedule composition，并在对应边界修复。

## 实现映射

| 关注点 | 源码 | 测试 |
|---|---|---|
| 强类型 Transformer specification | `src/blueprinting/compiler/models/transformer.py` | binding 与 calibration tests |
| 工作量代数 | `src/blueprinting/compiler/analysis/transformer_workload.py` | `tests/compiler/test_calculon_calibration.py` |
| 两个 derivation pass | `src/blueprinting/compiler/lowering/transformer.py` | canonical representation 与 calibration tests |
| 事务与 checkpoint | `src/blueprinting/compiler/passes/base.py` | `tests/compiler/test_pass_manager.py` |
| Evidence-derived estimate | `src/blueprinting/compiler/analysis/cost_model.py` | calibration tests |

[Calculon 校准实验](../../experiments/calculon-calibration.md)是这条已实现纵向切片的端到端审计。

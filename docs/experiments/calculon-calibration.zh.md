# Calculon 对比与可解释校准

## 目标与结论边界

这个实验独立验证三个问题：

1. Transformer training lowering 是否推导出正确的静态工作量；
2. 版本化 system evidence 能否解释 peak-only model 与 reference model 的差距；
3. 端到端 1F1B/interleaved schedule composition 是否保持可追溯。

编译与估算期间无法访问模型名、Calculon duration、论文 measurement 或 per-case correction。Calculon 与论文值只在这些阶段结束后作为 comparison oracle 进入。

在 8 个 SeqSel Table 5 case 上，当前 system-evidence path 与 Calculon 在 floating-point 精度内数值等价。这表示 Blueprinting 的 workload/analytical mapping path 独立复现了 reference work、system curve 与 schedule semantic；它既不是“真实硬件误差为零”的证据，也不能证明已具备广泛 architecture-exploration coverage。相对于论文报告的实测值，MAPE 为 3.65%，最大绝对误差为 8.87%。

## 实验身份与指标定义

### 冻结的 provenance

本报告的 machine-readable schema 是 `blueprinting.calculon-calibration-experiment.v0`。当前 oracle 是 vendored Calculon `0.1.0`，本地 Python source-tree SHA-256 为 `c72cf8a0a0fc9f1fb9813a2248747d6242bbc664b665abe4b5fc6b6b18f5927b`；硬件 evidence revision 为 `eb1eb9fcc4a6e414e85b0252c23ea9ad2730aae2`。JSON 产物还逐 case 保存 model、execution 与 system 输入文件的 SHA-256，以及 ModelIR、DistributedTaskIR、PortablePlanIR 和两个 PassCheckpoint digest。

Oracle 为 [Calculon](https://github.com/calculon-ai/calculon)，论文 holdout 来自 Korthikanti et al. 的 [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198) Table 5。Blueprinting 的两个 estimate 会先完成，之后才运行 Calculon；`test_oracle_runs_only_after_both_blueprinting_estimates` 对调用顺序执行负担明确的测试。

### 误差定义

对 Blueprinting 结果 $x_i$ 和 reference $r_i$，逐 case signed relative error 与跨 case MAPE 为：

$$
e_i=100\frac{x_i-r_i}{r_i},\qquad
\operatorname{MAPE}=\frac{1}{N}\sum_{i=1}^{N}|e_i|.
$$

当一个 component 的 reference 与 estimate 均为零时，其误差定义为零；若 reference 为零而 estimate 非零，则记为无穷大并使 gate 失败。Memory 另报告绝对 byte error，避免大容量使微小 byte 差异在百分比中消失。

对齐不是只看 total：每个 case 检查 19 个 operation/memory/message/capacity workload metric、总 memory、9 个 timing component、iteration total 和论文 holdout。共计 152 个 workload comparison。

### 覆盖矩阵

| Model | TP | PP | DP | Global batch | Microbatch | Interleave | 两种模式 |
|---|---:|---:|---:|---:|---:|---:|---|
| Megatron-22B | 8 | 1 | 1 | 4 | 4 | 1 | full / seqsel |
| GPT-175B | 8 | 8 | 1 | 64 | 1 | 3 | full / seqsel |
| Turing-530B | 8 | 35 | 1 | 280 | 1 | 3 | full / seqsel |
| Megatron-1T | 8 | 64 | 1 | 512 | 1 | 1 | full / seqsel |

## 实验路径

```text
model.json
   | semantic import
   v
ModelIR: transformer.decoder_training
   | transformer-distribute
   | - decompose Transformer primitives
   | - insert explicit TP collectives
   | - clone selective/full recomputation primitives
   v
DistributedTaskIR: local TP block task DAG
   | transformer-plan-work
   | - derive operations/read/write/message bytes
   | - do not bind GPU/LPU or write duration
   v
PortablePlanIR: exact WorkloadFacts
   +-- peak-only cost view
   +-- system-evidence cost view
   +-- PassCheckpoint / profiler inspection
          |
          v
1F1B schedule analysis -> comparison-only report
          |
          +-- Calculon oracle
          +-- SeqSel paper holdout
```

每层 IR snapshot 都有独立 digest。报告保存两个 pass checkpoint 的 schema 与 digest，使 profiler/checker 能消费准确的 immutable lowering boundary。

## 工作负载推导规则

### 线性层

对 `Y[M,K] = X[M,N] x W[N,K]`：

```text
forward operations = 2 M N K
dgrad operations   = 2 M N K
wgrad operations   = 2 M N K
```

Activation gradient 与 weight gradient 是两个独立矩阵乘。Workload analysis 不会根据从 Calculon 学到的比例拆分一个不透明 backward time。Memory traffic 同样从 dependency 推导：forward 读取 `X/W` 并写入 `Y`；dgrad 读取 `W/dY` 并写入 `dX`；wgrad 读取 `X/dY` 并写入 `dW`。

### Attention

Q/K/V projection、`Q x K^T`、softmax、dropout、`P x V` 和 output projection 分别成为 primitive。一个需要产生两个输入梯度的 batch matrix multiply 具有两倍 forward 的 gradient work：

```text
forward operations = 2 B M N K
dgrad operations   = 4 B M N K
```

Softmax、layer norm、GeLU、dropout、residual 与 fork 的计数也按 primitive 声明，不使用统一的 non-GEMM percentage。

### 重计算

重计算是 graph transformation，而不是 `forward_time x ratio`：

- `full` 克隆恢复所需的全部 forward compute；
- `attn_only` 只克隆 `QK matmul -> softmax -> probability dropout`；
- sequence-parallel all-gather redo 是独立的 recommunication task。

实验也暴露了 Calculon 的一个报告口径：`block_re_flops` 在遍历被重算层时累加运行中的 forward prefix，而 `block_re_time` 则对被选 operation 的 duration 求和。审计使用显式 recomputation operations 对照后者的 timing semantic，不复制 prefix counter behavior。

### 通信

TP all-reduce、reduce-scatter 与 all-gather 是显式 `DistributedTaskIR` collective。逻辑 message bytes 来自 activation shape；network profile 再应用 collective volume model：

```text
adjusted_bytes = message_bytes x volume_multiplier
adjusted_bytes += adjusted_bytes / participants x participant_offset
network_time = latency + adjusted_bytes / effective_bandwidth
```

Reduce-scatter/all-reduce 所需的 local reduction operations 与 HBM traffic 继续保留在对应 portable task 上。

### Pipeline 组合

端到端分析从 block critical path、每 stage block 数、microbatch count 与 interleaving factor 构造 1F1B bubble。Pipeline P2P message size 由 boundary-activation sharding 决定。

实验拒绝 TP/DP overlap 与 offload。这些语义必须由 concrete command dependency、queue、buffer 和 resource-conflict analysis 决定，并从 `ConcretePlanIR` 投影，不能藏在 portable-plan overlap ratio 中。

## 校准策略

两个 view 消费完全相同的 `PortablePlanIR`：

| View | Evidence | 用途 |
|---|---|---|
| `peak_only` | peak FLOP/s、bandwidth、latency、collective structure | 可证伪的朴素基线 |
| `system_evidence` | 上述信息加 operation-size、transfer-size 与 network efficiency curve | target-level calibrated result |

允许的 calibration input 只有 target-wide、datatype-specific response curve：matrix/vector operation-size efficiency、memory transfer-size efficiency、network bandwidth efficiency 与 collective volume model。

禁止输入模型/case identity、Calculon/论文 total time、per-case correction factor、从 Calculon 读取的 activation/weight-gradient ratio，以及固定 overlap ratio。8 个 case 使用同一个 canonical hardware evidence revision。

## SeqSel Table 5 结果

System profile：`a100_80g`；8 个 case 共用一个 evidence revision。

| Case | Peak-only (s) | Evidence (s) | Calculon (s) | 论文 (s) | Evidence vs 论文 |
|---|---:|---:|---:|---:|---:|
| Megatron-22B / full | 1.1691 | 1.3956 | 1.3956 | 1.42 | -1.72% |
| Megatron-22B / seqsel | 0.9351 | 1.1367 | 1.1367 | 1.10 | +3.33% |
| GPT-175B / full | 15.4997 | 18.0288 | 18.0288 | 18.13 | -0.56% |
| GPT-175B / seqsel | 11.5500 | 13.6381 | 13.6381 | 13.75 | -0.81% |
| Turing-530B / full | 44.8563 | 49.8926 | 49.8926 | 49.05 | +1.72% |
| Turing-530B / seqsel | 30.6678 | 34.4728 | 34.4728 | 37.83 | -8.87% |
| Megatron-1T / full | 81.6511 | 90.0809 | 90.0809 | 94.42 | -4.60% |
| Megatron-1T / seqsel | 59.3347 | 66.0407 | 66.0407 | 71.49 | -7.62% |

汇总：

- peak-only vs Calculon MAPE：**12.99%**；
- system-evidence vs Calculon：在 floating-point 精度内等价；
- system-evidence vs 论文 MAPE：**3.65%**；
- system-evidence vs 论文最大绝对误差：**8.87%**。

### Component-level parity

| Timing component | MAPE | 最大绝对误差 | 最大绝对时间误差 |
|---|---:|---:|---:|
| forward | 0 | 0 | 0 s |
| backward | 1.53e-14% | 2.80e-14% | 7.11e-15 s |
| optimizer | 0 | 0 | 0 s |
| recompute | 0 | 0 | 0 s |
| tensor parallel | 6.49e-15% | 1.98e-14% | 8.88e-16 s |
| pipeline parallel | 0 | 0 | 0 s |
| data parallel | 0 | 0 | 0 s |
| recommunication | 0 | 0 | 0 s |
| pipeline bubble | 2.25e-15% | 1.80e-14% | 1.78e-15 s |

所有 workload metric 和 8 个 memory total 都是 exact match；memory 最大绝对误差为 **0 byte**。非零 timing 差异的量级符合不同浮点运算结合顺序造成的 roundoff，而不是可观察的模型误差。

两个最大 seqsel case 的误差方向一致。下一轮应优先检查 sequence-parallel collective、大规模 topology 或论文环境差异，而不是增加 model-specific coefficient。

## 复现方式与产物

```bash
uv run python examples/calculon_calibration.py
uv run python examples/calculon_calibration.py \
  --output examples/calculon_calibration_result.json

# 执行 CI 使用的强制训练/推理 baseline gate。
uv run pytest -m baseline_regression tests/regression
```

8 组参数化训练回归位于 `tests/validation/test_calculon.py`。仓库级 gate 还会把 8 个 case 作为一个完整实验运行，并检查 `data/validation/baseline_regression_contract.json`：report schema、oracle source digest、输入 provenance、oracle 隔离策略、workload/component/total Calculon 等价性、memory、论文误差预算、evidence revision、case identity、aggregate golden 与每个 `PortablePlanIR` digest 被一起冻结。更新 golden 是必须经过 review 的 contract 变更；gate 不提供自动“接受当前输出”的模式。

CI 精度预算为 workload、component timing、Calculon total 均不超过 `1e-9%`，memory 最大绝对误差不超过 `1 byte`；当前结果显著严于这些阈值。完整机器报告位于 `examples/calculon_calibration_result.json`。

实现映射：

- `workload/transformer.py`：typed workload 与 execution facts；
- `synthesizer/frontend/transformer.py`：canonical import 与 binding adapter；
- `synthesizer/dialects/transformer/training.py`：静态 operation/byte derivation；
- `synthesizer/stages/distributed/passes.py`：ModelIR 到 DistributedTaskIR 的 pass contract；
- `synthesizer/stages/portable_plan/passes.py`：DistributedTaskIR 到 PortablePlanIR 的 pass contract；
- `synthesizer/dialects/transformer/training_derivation.py`：纯推导与 pattern matching；
- `analysis/cost_model.py`：peak-only 与 evidence-backed view；
- `validation/calculon.py`：oracle adapter、audit 与 report。
- `validation/regression.py`：严格的跨域 baseline gate 与诊断。

这是仓库唯一的 Blueprinting/Calculon calibration path。未来对比仍必须保证 workload construction 与 estimation 完成前无法访问 oracle data。内部 transformation contract 参见 [Transformer 工作负载推导](../design/passes/transformer.md)，未来 provider 迁移参见 [performance evidence](../design/performance/index.md)。

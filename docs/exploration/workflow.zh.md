# 探索工作流

硬件探索是一项可复现实验，而不是一系列临时 calculator edit。工作流把 question、workload、architecture variable、mapping、evidence fidelity、simulation 与 decision criterion 分开，使结论保持可解释和可修订。

## 定义决策

从真正可能改变 hardware blueprint 的决策开始：

- 增加 matrix throughput 是否有价值，还是 memory/communication 已经成为限制？
- 稀缺 area 应投入 SRAM、HBM interface、NoC bandwidth 还是 compute tile？
- 哪种 scale-up topology 能支撑 tensor/pipeline parallel workload？
- 面对 training、prefill 与 decode 时，最佳 architecture 如何变化？

声明 objective、hard constraint、uncertainty tolerance、evaluation budget，以及不在本次范围内的决策。“让它更快”不是 experiment specification。

## 定义 Workload Suite

Workload suite 包含代表性的 model family、shape、sequence length、batch regime、training/inference mode、precision、parallel strategy 与 scenario weight。它还应包含 stress case 和预期未来 workload，而不只是一个方便 benchmark。

Workload model 独立于 candidate architecture 推导精确 operation、data movement、communication、dependency 与 lifetime fact。这是所有 candidate 共享的受控 baseline。

## 生成 Candidate Blueprint

为 compute、memory、interconnect、system organization 与 physical envelope 定义 fixed fact、design variable、conditional variable 和 constraint。记录 generator revision，以及每个 variable range 为什么可信。

Candidate generation 可以组合 enumeration、parameterized template、domain rule、equality-based mapping alternative、Bayesian optimization 或专家提案。所有 candidate 在评估前都转换为普通版本化 blueprint。

## 构造合法 Mapping

对每个存活 blueprint，把 workload 映射为 implementation、shard、placement、queue、communication route、synchronization 与 buffer。本阶段先回答 feasibility，再回答 performance：

- target 是否支持每个 required primitive 与 datatype？
- tensor/layout/alignment constraint 是否满足？
- 在可能 overlap 下，buffer 是否能放入每级 memory？
- topology、queue 与 synchronization constraint 是否合法？

同一个 portable workload 在 GPU、LPU 或实验 architecture 上可以产生不同 concrete plan。非法 candidate 得到带 source lineage 的 diagnostic，而不是虚构 performance number。

## 选择 Evidence 与 Fidelity

选择足以解决当前决策的最低成本 evidence：

| Fidelity | 典型用途 |
|---|---|
| Exact bounds | capacity、operation/byte volume、dependency depth |
| Analytical model | 大范围早期 pruning 与 sensitivity |
| Performance database/surrogate | 已知 kernel、transfer 与 collective |
| Network simulator | topology、routing、congestion、collective alternative |
| Hardware simulator | engine、memory-system、pipeline、power 或 cycle detail |
| Measurement | 在 prototype 或现有 device 上做 calibration/final validation |

每个 fallback、interpolation、extrapolation 与 uncertainty bound 都被记录。High fidelity 应投入 decision boundary 附近，而不是平均用于整个空间。

## 仿真 Mapped Plan

Simulation 消费 verified architecture-bound command/resource plan。它从 dependency、queue、synchronization 与 resource capacity 计算 readiness，而不是使用 predicted timestamp。Trace 记录 utilization、contention、memory high-water mark、critical path、queue delay 与 metric provenance。

Task cost 与 network behavior 可以来自不同 provider，但 simulator 通过同一个 resource model 组合它们，避免各 subsystem 使用互不兼容的 overlap 或 scheduling semantic。

## 比较与解释

Result analysis 执行：

- feasibility filtering 与 constraint diagnostic；
- 按声明 objective 做 Pareto comparison；
- 按 resource 与 workload phase 归因 bottleneck；
- 对 architecture variable 做 sensitivity/what-if analysis；
- 跨 workload scenario 与 uncertainty 做 robustness analysis；
- 与旧 blueprint/evidence revision 做 regression comparison。

有用的 recommendation 不只说明 winning candidate，还要说明结论在什么条件下会改变。

## 验证与校准

把 predicted event/counter 与 compatible simulator、prototype、GPU/LPU 或 silicon observation 对比。在拟合新 model 前，先把 residual 关联到 workload task、concrete command、implementation 与 resource。

Calibration 创建新的 immutable evidence revision，并记录 training set、validity domain 与 validation error。历史 experiment 继续可复现；采用新 revision 时显式 reevaluate，而不是修改旧 report。

## 发布 Experiment Bundle

完整 experiment bundle 包含：

```text
question and hypothesis
workload-suite digest
candidate blueprint definitions and generator revision
mapping/concrete-plan digests
evidence and simulator revisions
objectives, constraints, uncertainty policy, seed, and budget
results, traces, diagnostics, and rejected candidates
recommendation and claim boundary
```

这个 bundle 才是持久 Blueprinting product。Target backend 存在时，可以把 optional runtime configuration 或 program artifact 附加到被选 blueprint。

## 示例推进过程

一次 LPU exploration 可以先对 matrix-engine count、SRAM capacity、HBM bandwidth 与 scale-up link 做 analytical sweep。Capacity/roofline bound 删除大部分 candidate；schedule-aware simulation 随后比较 buffer reuse 与 compute/communication overlap；network simulator 解析 finalist 的 collective behavior；prototype observation 校准 efficiency curve；修订后的 Pareto set 再决定哪种 architecture 进入下一阶段。

具体算法可以演进；experiment identity、semantic conservation 与 evidence lineage 是固定 contract。

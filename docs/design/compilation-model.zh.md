# 形式化推导与验证模型

推导模型定义 workload 映射到 candidate hardware blueprint 时已经知道什么、还有哪些决策未定、哪些事实允许变化，以及每次 transition 必须建立哪些可检查 claim。它是 workload model、mapping planner、architecture model、evidence provider、simulator、optional emitter 与 profiler 共同遵守的语义主轴。

!!! note "这里所说的形式化"
    “形式化”指 typed semantic、显式 constraint、可执行 verification condition、确定性推导与可追溯 claim boundary。Blueprinting 目前不声称已经具备 proof-assistant certification 或 exhaustive model checking；这些能力需要独立且贯通的实现与证据。

### Assurance level

文档或 report 中的“verified”必须同时指出验证层级，不能把局部 schema check 升格为系统正确性：

| 层级 | 可支持的 claim | 不能支持的 claim |
|---|---|---|
| Structural | Type、reference、DAG、schema 与局部 legality check 通过 | Workload semantic 正确或性能准确 |
| Conservation | 可执行 invariant/property test 证明 operation、byte、shard 或 dependency 守恒 | 对全部输入的 theorem proof |
| Cross-consumer conformance | Simulator/emitter/replay 消费相同 plan digest 且 event lineage 对齐 | 真实时间行为相等 |
| Empirical correlation | 在声明 workload、target、evidence domain 上误差被 measurement 量化 | 域外泛化或因果最优性 |
| Mechanized proof | 由独立 formal specification 与 proof artifact 支持 | 当前尚无此级别能力 |

Schema version、test count 或 reference total 对齐都不能单独提高 assurance level。

## 推导状态

一次推导步骤操作显式状态，而不是环境中的 Python global：

```text
DerivationState_i = (
    representation_snapshot,
    resolved_bindings,
    candidate_provenance,
    analyses,
    evidence_snapshot,
    open_obligations
)
```

当前实现把 typed derivation context 命名为 `CompilationSession`。它标识 workload、strategy、target requirement、deployment、calibration revision、feature set 与 deterministic seed；session fingerprint 参与每个 analysis address。这个名称是从编译工程沿用的实现标识，不代表概念架构中存在一个 Compiler 组件。

## 推导 Contract

每个分阶段推导都是偏函数；当前代码通常以 lowering pass 实现这类规则：

```text
Derive_i(DerivationState_i)
  -> Verified<DerivationState_i+1>
   | Diagnostics
```

它有三项义务：

### 语义保持

Source 的含义必须等于 result 投影回 source abstraction 后的含义：

```text
meaning(IR_i) = project_i(meaning(IR_{i+1}))
```

Decomposition、sharding、recomputation、implementation selection 和 command encoding 可以增加结构，但不能静默改变 workload。

如果允许已声明 approximation，则必须用 typed refinement relation 或 error bound 替代等式，并把 approximation policy 记录为 provenance。

### 约束单调细化

推导可以增加约束或选择 alternative，但不能接受投影回上游后原本非法的结果。这是 semantic obligation，而不只是语法上追加更多 predicate：

```text
project_i(Solutions(C_{i+1})) ⊆ Solutions(C_i)
```

### 消解本层义务

每个边界必须消解分配给本边界的全部 obligation，并提高 specificity。推导可以显式引入 typed downstream obligation，因此简单统计所有 unknown 的数量不是合法 progress measure：

```text
owned_i(Open(State_{i+1})) = ∅
specificity(State_{i+1}) > specificity(State_i)
```

如果本层 obligation 仍未消解，或新引入的 downstream obligation 无法安全表示，推导必须返回 diagnostic，而不是填入默认值。

## Binding Lattice

| 维度 | 典型值 | 首次产生的结构影响 |
|---|---|---|
| Workload | batch、sequence、mode、micro-batch | Shape specialization 与 workload expansion |
| Strategy | TP/PP/DP、recompute、pipeline、fusion | Distributed graph 与 portable-plan topology |
| Target | architecture、runtime、primitive support、ABI | Legality 与 implementation selection |
| Deployment | instance、topology、capacity、reservation | Placement、routing、scheduling 与 memory feasibility |
| Calibration | evidence snapshot 与 model revision | 只改变 estimate value |

Portable planning 可以消费 target requirement envelope 来剪枝。Requirement 不等于 binding：它不能携带 vendor identity、physical device、kernel ID 或 measured latency。

## 推导 Gate

| 转换 | 所需 context | 成功后必须已经解析 |
|---|---|---|
| Source → `ModelIR` | 模型语义 | Operation meaning、value type、effect、声明的 dynamic dimension |
| `ModelIR` → `DistributedTaskIR` | Strategy | Logical ownership、sharding、communication、cross-rank dependency |
| `DistributedTaskIR` → `PortablePlanIR` | Planning policy | Task DAG、abstract resource、work fact、buffer lifecycle、objective |
| `PortablePlanIR` → `ConcretePlanIR` | Target、deployment、implementation、evidence policy | Legality、placement、queue、sync、buffer、capacity |
| `ConcretePlanIR` → `MachineIR` | Target ABI 与 emitter | Instruction form、section、entry point、reference |

## 基于已验证状态的自动分析

Transformation 与 analysis 必须分离：transformation 解析一项决策并产生新的权威状态；analysis 在不改变 source semantic 的前提下产生可重建结果。主要分析类别包括：

| 分析类别 | 建立的事实 |
|---|---|
| 精确工作负载分析 | 从模型语义推导 operation、byte、message、dependency 与 lifetime |
| 合法性与可行性 | Capability coverage、shape/layout rule、topology、capacity 与 scheduling constraint |
| 定量评估 | Evidence-backed latency、throughput、contention、energy、area、cost 与 uncertainty |
| 计划分析 | Critical path、utilization、bottleneck、memory pressure 与 overlap consequence |
| 跨候选分析 | Dominance、Pareto frontier、sensitivity、robustness 与 evidence-dependent ranking |

每份结果都记录 source representation digest、binding/evidence fingerprint、analysis revision、validity domain 与 diagnostic。Analysis 可以触发新的 planning decision，但不能静默修改它观察到的状态。

## Concrete Execution Abstract Machine

`ConcretePlanIR` 的含义来自一个 abstract machine，而不是预测 timestamp 列表。

```text
MachineState = (
    command_state,
    queue_heads,
    resource_occupancy,
    buffer_state,
    synchronization_tokens,
    observations
)
```

Command 的 ready 条件为：

```text
ready(c) =
    dependencies_completed(c)
    and queue_predecessors_completed(c)
    and synchronization_satisfied(c)
    and resources_available(c)
    and buffers_valid(c)
```

初始 command vocabulary 包含 `Launch`、`Collective`、`Transfer`、`Barrier`、`Signal`、`Wait` 和 `HostCall`。Command 从 pending 转为 ready、running，最终 completed 或 failed。Completion 发布 dependency token 和 observation，并按照 buffer 与 queue contract 释放资源。

## Simulation 与 Runtime 对应关系

Simulation 与 runtime 必须共享 command identity、dependency intent、resource/buffer contract 和可观察 event vocabulary；它们不需要共享同一份 transition 实现，也不承诺 wall-clock 行为相等。区别首先在于 completion 来源：

```text
Simulator: completion_time := resolve_cost(command, context)
Runtime:   completion       := observe_target_event(command)
```

Simulator 推进 logical clock 并建模资源；runtime 提交真实工作并等待 target event。Runtime 可以执行 target contract 明确允许的 bounded backpressure、retry、timeout 与 failure handling，但不得静默增加 high-level dependency、改变 workload 或重新做无界 global planning。

Conformance 检查的是 runtime observation 投影后是否属于 concrete plan 允许的 trace language：

```text
project(runtime_events, observation_policy)
  conforms_to allowed_traces(ConcretePlanIR, target_extension)
```

Observation policy 必须声明 exact-event、bounded-divergence 或 partial-observation level。Predicted-versus-observed timing 属于单独的 correlation result，而不是 execution equivalence proof。

## 确定性与 Revision

固定 source snapshot、binding set、analysis-engine/plugin version、evidence revision、policy 和 seed 后，必须产生相同 canonical digest 与 diagnostic。Search 可以探索 alternative，但 budget、seed、pruning decision 和 rejected candidate 都必须保留 provenance。

Evidence 与 observation 都是 immutable revision。Re-estimation 可以重建 view，replanning 可以产生新的 concrete plan，但任何 service 都不能修改已经发布的 IR 或 artifact。

## 失败模型

失败必须分类型并限定阶段：malformed source、verification failure、missing binding、unsupported capability、unresolved evidence、infeasible capacity、scheduling failure、target-verifier failure 和 runtime observation mismatch。Diagnostic 必须包含 stable subject ID、source lineage、相关 fingerprint、被违反的规则和尝试过的 fallback。

Transformation execution 是事务性的：verifier 或 observer 失败时，上一份 representation snapshot 和 analysis store 保持不变。

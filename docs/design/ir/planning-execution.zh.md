# 规划与执行 IR

`PortablePlanIR`、`ConcretePlanIR` 和 `MachineIR` 分离 portable execution intent、concrete resource decision 和 target encoding。这条边界阻止 target evidence 泄漏到上游，同时允许 backend 完整表达 accelerator-specific behavior。

## PortablePlanIR

### 语义域

`PortablePlanIR` 用 target-neutral execution constraint 表示已经选择或正在候选的 deployment strategy。

```text
PortablePlanIR
├── tasks and dependency DAG
├── WorkloadFacts
├── abstract ResourceRequirements
├── abstract buffers and lifetimes
├── implementation capability alternatives
├── concurrency groups
├── objectives and constraints
└── provenance and lineage
```

可推导时，exact operation、read/write bytes、message bytes、reuse 和 arithmetic intensity 属于这一层。Recomputation、checkpointing、fusion、pipeline order 和 abstract resource choice 是显式结构决策。

### 禁止的信息

Portable plan 不能包含 physical implementation ID、vendor library、physical device、route、queue、engine、memory bank、address、target-derived latency 或 execution timestamp。

### Verification

Verifier 检查 task reference integrity、DAG closure、buffer lifecycle consistency、resource-requirement validity、exact nonnegative work fact、objective/constraint identity、source lineage，以及 target-bound field absence。

`PlanSet` 是包含 candidate 和 dominance provenance 的 immutable artifact，不是第六层 canonical dialect。

## Target 与 Deployment Binding Gate

Gate 消费 portable candidate、target、deployment、implementation policy 和 evidence policy。它必须：

1. 证明 target capability 与 ABI compatibility；
2. 选择 legal implementation candidate；
3. 把 abstract resource 映射到 target resource；
4. 建立 physical placement 和 memory space；
5. 做出足够定义 correctness 的 queue、sync 和 buffer decision；
6. 用 typed diagnostic 拒绝 unresolved 或 infeasible requirement。

提前使用 `TargetRequirements` 可以约束 device count、capacity、dtype 或 required collective，但不能标识 vendor、architecture、runtime 或 physical cluster。

这里的 evidence 是 **planning evidence**：它解释 implementation/schedule 为什么被选择。后续对同一 plan 做 re-cost 使用 **evaluation evidence**，属于 derived view identity。当前 `ConcretePlanIR.evidence_revision` 尚未在字段名上区分这两种角色；production producer 落地前必须关闭这个 contract 歧义。

## ConcretePlanIR

### 语义域

`ConcretePlanIR` 是 target/deployment binding 之后的权威 execution-plan envelope。Common coordination core 表达跨 target 稳定的 command DAG；typed target extension 表达只对某种 architecture 有 correctness 含义的 dataflow、route、issue 或 protocol constraint：

```text
ConcretePlanIR
├── selected implementations
├── devices, queues, and engines
├── commands and dependency tokens
├── synchronization and collectives
├── buffer regions, offsets, lifetimes, and reuse
├── target/deployment/ABI fingerprints
├── evidence and planner revisions
├── typed target schedule extension
└── lineage to portable tasks
```

初始 command 包括 `Launch`、`Collective`、`Transfer`、`Barrier`、`Signal`、`Wait` 和 `HostCall`。Planned allocation 可以是 static buffer binding，不要求 runtime allocation command。

### Operational Meaning

Command 在 dependency、ordering、synchronization、resource 和 buffer contract 满足时执行。Predicted wall-clock timestamp 既不是必需字段，也不是权威语义。若 target 把 cycle/slot 作为 correctness constraint，它属于 target extension 和后续 `MachineIR`，而不是通用 duration annotation。

### Verification

完整 producer 的 verifier 必须检查 DAG acyclicity、dependency-token production/consumption、ordering legality、cross-resource synchronization、implementation coverage、placement、buffer lifetime/overlap、address bound、resource capacity、target fingerprint、typed extension 和 lineage。

当前仓库只实现通用 device/queue/buffer/command schema 与一组 structural verifier；尚无 production portable-to-concrete construction pass、route/resource-occupancy semantic、typed target extension 或 end-to-end target conformance。这个 v1 是 experimental serialization contract，不是 frozen public ABI。

## MachineIR

`MachineIR` 是 target-specific lowering product。名称中的 “Machine” 不承诺一定是裸 ISA；它可以拥有 instruction/runtime command、program section、entry point、symbol、target ABI、executable reference 和 target verifier requirement。

不同 target 可以使用不同 dialect：CUDA/NCCL call 与 executable reference、LPU command packet 与 memory descriptor、CPU task、firmware command 或 external framework package。公共概念使用 protocol，但不能为了表面统一抹除 target-specific semantic。

Machine lowering 不能重新解释 portable work，也不能增加未规划的 communication dependency。Emission 前由 target verifier 检查 legality 与 ABI compatibility。

## Derived Product

| Product | 内容 | 失效来源 |
|---|---|---|
| `CostedTaskView` | 与 plan entity 关联的 versioned estimate | Source、target、deployment、evidence 或 calibration 变化 |
| `TimingProjection` | Predicted interval、critical path、overlap、uncertainty | Concrete plan、cost 或 simulation policy 变化 |
| `SimulationTraceIR` | 带 command lineage 的 event/counter interchange | Projection 或 adapter revision 变化 |
| `TimelineBundle` | Concrete digest + projection/trace + evidence/policy + diagnostics 的发布包 | 任一被引用 artifact 或 manifest policy 变化 |
| `EvaluationReport` | Objective、constraint、bottleneck、validation | Plan、evidence 或 policy 变化 |
| `ObservationSet` | Immutable measured/simulated record | 不修改；由新 revision 取代 |
| `ProgramArtifact` | Executable/replay package 与 manifest | MachineIR、ABI、library 或 emitter 变化 |

## Program Artifact Contract

Artifact manifest 记录 schema version、target/deployment requirement、target ABI、portable/concrete/machine digest、analysis-engine/plugin version、required runtime library，以及 debug lineage map 或 detachable symbol package。

Timing 可以为了诊断被打包，但不是 loader 或 runtime correctness requirement。

`TimelineBundle` 同样只能引用 canonical/derived artifact，不能复制并修改 command DAG。完整阶段路径见 [Timeline 路径](../timeline-path.md)。

## 端到端可追溯性

Toolchain 支持两个方向：

```text
Model operation -> distributed task -> portable task
                -> concrete command -> machine instruction -> profiler event

Profiler event -> machine instruction -> concrete command
               -> portable task -> distributed task -> model operation
```

Report 与 observation 使用 stable ID 和 snapshot digest，不能把 display name 当作 identity。

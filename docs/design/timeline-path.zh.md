# Timeline 路径：阶段性产品，而不是架构捷径

“编译出 timeline”是 Blueprinting 近期很有价值的工程路径，但这句话容易把三种不同事物混为一谈：执行正确性、性能预测和 target-specific 时间控制。本页固定它们的边界，并说明这条阶段性路径如何自然演进到 GPU、LPU 或其他 backend，而不改变硬件架构探索的产品目标。

!!! warning "设计与实现状态"
    本页定义已接受的演进方向。当前 production 路径只贯通到 `PortablePlanIR` 与 analytical estimate；`ConcretePlanIR` producer、discrete-event simulator、`TimelineBundle` 和 executable backend 仍未实现。具体状态以[实现状态](../project/status.md)为准。

## 决策摘要

Blueprinting **不会**恢复旧的 `GraphIR -> ScheduleIR -> TimelineIR` public stack，也不会把预测 absolute timestamp 设为执行语义。阶段性产品定义为：

```text
verified PortablePlanIR
  + candidate architecture / target / deployment
  + planning evidence and policy
  -> verified ConcretePlanIR              authoritative coordination plan
  -> TimingProjection                     predicted intervals + uncertainty
  -> SimulationTraceIR                    simulated resource events
  -> TimelineBundle                       publishable analysis/replay bundle
```

`TimelineBundle` 是一个带完整 provenance 的产品包，不是第六层 canonical IR。它引用而不复制 `ConcretePlanIR`，可以被工作台、仿真器、对比实验和 replay prototype 消费。

## 五个概念不能混用

| 概念 | 拥有的语义 | 不拥有的语义 |
|---|---|---|
| `ConcretePlanIR` | Command dependency、resource binding、queue/order、synchronization、buffer plan，以及未来的 typed target schedule extension | Predicted wall-clock truth |
| `TimingProjection` | 在指定 evidence、contention model 和 policy 下的 predicted interval、slack、critical path 与 uncertainty | Correctness、永久 target fact |
| `SimulationTraceIR` | 一次 simulator run 产生的 event/counter 与 command lineage | 独立 schedule、workload semantic |
| `TimelineBundle` | Concrete digest、projection/trace、evidence/policy fingerprint、diagnostic 与 artifact manifest | 新 canonical semantic layer |
| `MachineIR` | Backend-owned command/instruction、ABI、section、entry point 与 target verifier contract | 重新决定高层 mapping 或 communication dependency |

在口头讨论中，“timeline”可以指最后三项中的任意一个；在 schema、API、论文和测试中必须使用准确名称。

## 预测时间与强制时间

### Predictive time

大多数 GPU、network 和系统级仿真的时间是预测值。它依赖 implementation、deployment、并发 context、evidence revision 和 simulator policy。修改 evidence 后可以得到新的 `TimingProjection`，但不能追溯修改 workload fact。

删除 predictive timestamp 后，`ConcretePlanIR` 仍应保留足以判断 command readiness、resource legality 和 buffer safety 的信息。这是计划能够跨 cost model 复用的条件。

### Prescriptive time

某些 target 可能把 issue cycle、slot、静态 phase 或 time-triggered protocol 作为程序正确性的一部分。此时时间不是通用预测字段，而是 **target binding 之后**的 typed scheduling constraint：

```text
ConcretePlanIR target extension
  -> target verifier
  -> MachineIR issue/slot encoding
```

这类约束必须说明 clock domain、容差、backpressure、timeout/failure behavior 和 ABI revision。它不能泄漏到 `PortablePlanIR`，也不能通过 free-form `attributes` 获得 correctness 含义。

这个区分允许 LPU 充分利用静态 dataflow 或确定性 issue 能力，同时不强迫 GPU、CPU 或网络 target 假装拥有相同的时间语义。

## 为什么 Timeline 仍是正确的近期交付

虽然 timeline 不是 canonical execution truth，它仍然能形成第一条有产品价值的 architecture-bound 纵向切片：

1. 把 abstract work 变成可检查的 resource occupancy 和 event sequence；
2. 显示 critical path、queue delay、overlap consequence 和 memory high-water mark；
3. 让 performance database、analytical model、network simulator 和 hardware simulator 对同一 command 提供可替换 evidence；
4. 为每层 derivation 保留 stable lineage，使预测 event 能与 profiler observation 对齐；
5. 在真实 target ABI 之前，用 virtual target/replay 验证 planning contract；
6. 为 LPU backend 保留从同一 concrete plan 继续 lowering 的入口。

它是 architecture exploration 的阶段性产品，不是把项目重新定义为 timeline compiler。

## 分阶段路径与验收 Gate

| 阶段 | 交付物 | 必须证明 | 明确不声称 |
|---|---|---|---|
| A — 当前 baseline | `PortablePlanIR` + analytical estimate | Work/byte/collective conservation，stable lineage，evidence provenance | Architecture-bound schedule 或 event simulation |
| B — Virtual timeline | Virtual target `ConcretePlanIR` + `TimingProjection` + event trace | Resource legality、buffer safety、deterministic replay、手算 DAG 对齐 | 真实 GPU/LPU 性能或 ABI 可执行性 |
| C — Multi-target simulation | 多 blueprint binding + provider-backed timeline bundle | 同 workload digest、target-specific legality、uncertainty、bottleneck 可解释性 | 所有 target 共享相同 queue/dataflow 模型 |
| D — Backend replay | Target plugin + replayable `MachineIR` | Command/instruction lineage、emitter 不重新调度、trace correlation | 已能驱动硅上硬件 |
| E — Executable LPU/GPU | ABI-specific artifact + runtime/profiler adapter | Target verifier、failure behavior、observed event correspondence | Predicted timestamp 与 observed timestamp 必然相等 |

只有前一阶段的验收成立，下一阶段的 artifact 才能成为 evidence，而不是演示代码。

## LPU 不是特例分支

LPU 在 Blueprinting 中首先是 architecture candidate，其次才是 executable target。Backend 按成熟度逐步增加能力：

| 成熟度 | 必需 contract | 可以完成的工作 |
|---|---|---|
| Architecture | Capability/resource hierarchy、constraint、canonical identity | 与其他 blueprint 做静态可行性和设计空间比较 |
| Simulation | Legalizer、resource model、implementation catalog、cost/simulator provider | 生成 architecture-bound timeline 并分析瓶颈 |
| Replay | Typed target schedule extension、MachineIR dialect、deterministic adapter | 验证 plan 与 backend command 的对应关系 |
| Executable | Stable ABI、emitter、loader/runtime、target verifier、profiler normalizer | 生成并验证真实 program artifact |

因此无需等待 LPU hardware 才开始前两层，也不能因为团队对 LPU 感兴趣就提前在 frontend 或 portable plan 中加入 LPU opcode、memory bank 或固定 timing。

## ConcretePlanIR 的扩展边界

共同层只应该表达跨 target 稳定的 coordination kernel：identity、dependency、resource claim、buffer reference、synchronization 与 lineage。Spatial dataflow、route、issue slot、collective micro-protocol 或特殊 memory movement 必须进入 namespaced **typed target extension**，并由 target plugin 验证。

当前 `ConcretePlanIR` v1 只有通用 device/queue/buffer/command schema，还没有这种 typed extension，也没有 production producer。它是实现骨架，不是已经冻结的跨 target ABI。引入第一个 non-queue-centric virtual target 前，必须先验证公共 core 不会把所有架构强行拟合成 GPU stream 模型。

## 每层如何与观测联动

“每层都可 check”不等于“每层都直接读取 runtime profiler”：

| 边界 | 首要 checker | 可关联 observation |
|---|---|---|
| `ModelIR` | Type/effect/shape verifier、numerical reference | Framework op/shape reference |
| `DistributedTaskIR` | Shard reconstruction、collective matching、volume conservation | Rank-level communication trace |
| `PortablePlanIR` | Work/byte/buffer/resource-requirement conservation | Backend-independent counters，若能可靠归一化 |
| `ConcretePlanIR` | Placement/queue/sync/capacity/target legality | Command/resource event |
| `TimingProjection` | Critical-path/event consistency、uncertainty policy | Predicted-versus-observed interval |
| `MachineIR` / artifact | ABI/target verifier、lineage coverage | Instruction/kernel/packet profiler event |

Profiler adapter 负责把 target-specific event 归一化为 immutable `ObservationSet`。Observation 可以使某项 prediction 失效并产生新 evidence revision，但不能修改已经发布的 IR。

## 不可通过的捷径

- 用 total latency 对齐代替 event/command 对齐；
- 把预测 timestamp 写入 readiness 或 buffer-correctness 语义；
- 让 simulator 与 emitter 分别重建 schedule；
- 用 untyped metadata 承载 LPU dataflow、route 或 issue legality；
- 宣称 runtime “零决策”，却不定义 backpressure、failure、timeout 和 dynamic-shape policy；
- 把 schema `1.0.0` 当成 public compatibility 承诺，而 producer/consumer 还没有贯通；
- 为了一个 target 的便利改变共同 workload facts。

## 进入实现前的最小决策

下一步实现 virtual timeline 之前，必须关闭以下问题：

1. `ConcretePlanIR` common coordination core 与 typed target extension 的界面；
2. Planning evidence 与 evaluation evidence 的独立 identity；
3. Resource claim、route 和 concurrency occupancy 的最小语义；
4. Timeline bundle 的 schema、失效规则与 uncertainty 表达；
5. Simulator/runtime observation 的 correspondence level；
6. Dynamic duration、backpressure 和 failure 的 executor contract；
7. Experimental schema 到 public compatibility contract 的 graduation gate。

这些问题是 architecture-bound simulation 的入口条件，不应推迟到 LPU emitter 阶段才处理。

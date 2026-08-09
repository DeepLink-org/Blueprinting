# 硬件设计空间

硬件探索从强类型、版本化 candidate definition 开始。Candidate 不能只是 device name 或 peak-FLOP number；它是一份可组合的 resource、topology、constraint 与 implementation capability 蓝图，能够被 mapping、simulation、comparison 与 revision。

!!! note "设计状态"
    本页定义已接受的产品模型。当前仓库已有 `SystemProfile` evidence，但尚无完整 public `ArchitectureBlueprint` schema 或 search API。

## Candidate Blueprint

概念上，一份 candidate 包含：

```text
ArchitectureBlueprint
├── compute_substrate
├── memory_hierarchy
├── on_chip_fabric
├── package_and_scale_up
├── scale_out_network
├── system_organization
├── capability_and_programming_model
├── physical_envelope
└── assumptions_and_revision
```

Blueprint 拥有 architecture fact 与 variable。Measured/simulated latency 属于 evidence；某次具体 cluster allocation 属于 deployment；workload mapping 属于 experiment。区分这些对象后，同一 architecture 才能在多个 workload、deployment 与 evidence fidelity 下评估。

## Compute Substrate

Compute parameter 包括 engine kind/count、supported operation/precision、vector width、array dimension、accumulator behavior、clock domain、issue/queue structure、local reduction support、sparsity 与 dataflow constraint。

Peak throughput 只是 derived upper bound。Useful throughput 取决于 operation shape、tiling、utilization、data movement、synchronization 与 concurrency。因此 Blueprinting 把 compute capability 关联到 implementation candidate 与 evidence domain，而不是分配统一 efficiency。

## Memory Hierarchy

每级 memory 描述 capacity、bandwidth、latency、port、bank、alignment、addressability、consistency 与 supported transfer。Hierarchy 可以覆盖 register、local SRAM、shared cache、HBM、host memory 与 remote memory。

重要探索变量不只是“更大容量”和“更高带宽”。Bank conflict、reuse scope、placement、double buffering、collective staging 与 lifetime pressure 往往决定 compute engine 能否获得足够数据。Memory planning 必须使用 mapped execution plan 的真实 buffer lifetime。

## Interconnect 与通信

Communication model 覆盖多个 scope：

| Scope | 示例变量 |
|---|---|
| On-chip | NoC topology、router、virtual channel、link width、multicast/reduction support |
| Die/package | chiplet topology、die-to-die bandwidth/latency、coherence 或 message semantic |
| Scale-up | device topology、collective offload、link class、routing、oversubscription |
| Scale-out | NIC、switch、topology、rail mapping、congestion 与 failure domain |

Logical message bytes 来自 workload mapping。Physical traffic 与 exposed time 取决于 collective algorithm、route、contention 与 implementation。因此 network simulator 细化而不是重新定义 logical communication fact。

## System Organization

System candidate 把 chip 组合成 package、node、rack 与 cluster，并指定 multiplicity、affinity、topology、host involvement、storage/offload path、power domain 与 resource reservation。

Architecture 与 deployment 保持独立。“由 topology X 连接的 8 个同类 device”可以是 architecture template；“当前具有这些 reservation 的 8 个具体 device”则是 deployment。Exploration 通常先评估 template，再绑定真实 deployment。

## 物理与经济边界

Architecture decision 最终受 area、power、thermal、packaging、yield 与 cost 约束。这些 metric 和 latency 一样，需要独立版本化 provider：

- 用 analytical area/energy model 做早期 pruning；
- 用 component/floorplan model 做 refinement；
- 对选定 candidate 使用 power/thermal simulation；
- 用 BOM、packaging 与 deployment-cost model 比较系统。

缺失 model 仍是 unknown。Blueprinting 不能通过省略不方便的 objective，就推断某设计 Pareto-optimal。

## Capability 与可编程性

Blueprint 声明 supported primitive、layout、synchronization、queue、memory operation、collective capability、runtime/ABI constraint 与 extension point。这些 fact 决定 workload mapping 是否合法，以及需要多少 software specialization。

LPU、GPU 与其他 accelerator 可以具有实质不同的 programming model。Portable workload/mapping intent 保持共享；target plugin 只在 architecture-binding boundary 解释 capability。

## 设计变量与约束

每个 field 被分类为：

- fixed fact；
- enumerable 或 bounded design variable；
- derived quantity；
- hard constraint；
- soft preference；
- 需要 provider 或后续 binding 的 unresolved value。

Constraint 包括 divisibility、topology feasibility、capacity、bandwidth、power、area、implementation legality 与 experiment budget。Conditional variable 会阻止无意义组合，例如只有选择 buffered-router fabric 时才存在 router-buffer depth。

## 分层搜索

设计空间过大，不能做 flat Cartesian enumeration。Search 按层推进：

1. 使用 exact work 与 capacity bound 拒绝不可能 candidate；
2. 使用 analytical model 做粗粒度 architecture pruning；
3. 为存活 candidate 构造合法 mapping；
4. 选择性使用 schedule-aware estimate 与 network/hardware simulation；
5. 保留 Pareto 与 uncertainty-relevant candidate；
6. 在 decision boundary 附近投入 detailed simulation 或 measurement budget。

Equality saturation 可以生成等价 mapping 或 local implementation，但它只是这套 hierarchy 内的一种 candidate-generation technique，不是产品组织抽象。

## 可复现 Candidate Identity

每项 result 都记录 blueprint digest、variable assignment、schema revision、generator/search revision、workload suite、mapping digest、deployment context、evidence snapshot、simulator revision、objective、constraint、seed 与 budget。

任何重要 assumption 变化都会产生新的 candidate 或 experiment identity。这使 architecture exploration 从非正式 spreadsheet tuning 变为可复现工程流程。

Schema ownership 参见[硬件架构模型](../modeling/hardware.md)，实验构造参见[探索工作流](workflow.md)。

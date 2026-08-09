# 为什么叫 Blueprinting

Blueprinting 是一个面向分布式 AI 工作负载的硬件架构探索与仿真系统。它帮助架构师描述候选计算、存储、互连和系统组织，将代表性 workload 映射到这些架构上，使用可追溯 evidence 估算和仿真其行为，并在投入硬件实现前比较设计权衡。

## 名字就是产品判断

Blueprint 既不是松散想法，也不是已经完成的 silicon，而是一份足够精确、可检查、可比较、可修改，并能交给下一工程阶段的设计蓝图。

Blueprinting 的作用正是：

- 在不重写 workload model 的前提下勾勒多个 hardware organization；
- 显式记录 assumption、constraint 与 evidence revision；
- 把每份草图变成 simulation-ready execution plan；
- 解释一个设计为什么胜出或失败，而不只给出 predicted latency；
- 保留从早期 analytical model 到详细 simulator，再到未来 hardware program 的演进路径。

Typed formal model、verified derivation 与 automated analysis 使这些 blueprint 可以公平比较。IR 与 lowering 是这套方法内部采用的实现技术，不是产品身份。

## 使命

给定 workload suite 与 architecture question，Blueprinting 应回答：

> 哪种硬件组织最符合目标 performance、memory、energy、area、cost 与 programmability envelope？支持这个结论的证据是什么？

系统既要支持把 hardware parameter 当作变量的 **architecture exploration**，也要支持固定现有 GPU、LPU 或其他 target 的 **deployment evaluation**。前者是主要口径，后者只是同一设计空间中的一个点。

## 系统应回答的问题

| 领域 | 架构问题 |
|---|---|
| Compute | 对当前 workload mix，多少 matrix/vector engine、何种 precision 和 utilization 才真正有价值？ |
| Memory | 哪种 capacity、bandwidth、banking 与 hierarchy 能消除真实瓶颈，而不是只提高峰值指标？ |
| On-chip communication | 哪种 NoC topology、link width、routing 与 collective support 能支撑选定 dataflow？ |
| Scale-up/scale-out | 当 model/context 扩大时，chip、package、node 与 cluster 应如何连接？ |
| Mapping | 哪种 sharding、recomputation、pipeline、placement 与 scheduling 适合该架构？ |
| System trade-offs | Latency、throughput、memory、energy、area 与 cost 的 Pareto frontier 在哪里？ |
| Robustness | 当 uncertainty、workload 分布或 evidence revision 变化时，结论是否仍成立？ |

单一端到端时间无法回答这些问题。Blueprinting 还需要 workload conservation、resource utilization、contention、critical path、sensitivity、uncertainty 与 provenance。

## Blueprint 与单点估算器

单点估算器询问“configuration X 需要多久”；Blueprinting system 还要询问：

1. 哪些设计决策产生了 configuration X？
2. 哪些 resource 与 dependency 使结果可行？
3. 瓶颈是什么，修改哪个参数能移动它？
4. 哪部分结果来自 exact workload fact、measurement、simulation 或 assumption？
5. 哪些邻近设计占优，哪些在 uncertainty 下仍有竞争力？
6. 硬件可用后，同一 plan 是否能够 replay 或 emit？

因此，一项实验的结果是 architecture、workload、mapping、evidence、simulation、metric 与 diagnostic 的版本化 bundle，而不是 spreadsheet cell。

## 探索闭环

![硬件架构探索闭环](../assets/architecture/hardware-exploration-loop.svg)

闭环包含五个阶段：

1. 用 workload、objective 与 constraint 定义 architecture question；
2. 生成版本化 candidate hardware blueprint；
3. 构造合法 workload mapping 与 execution plan；
4. 解析 evidence，并以合适 fidelity 进行 simulation；
5. 比较 bottleneck、sensitivity、uncertainty 与 Pareto result，再细化 candidate。

来自 simulator、prototype、GPU、LPU 或未来 silicon 的 measurement 会生成新 evidence revision。它们改进后续实验，但不会改写历史结果。

## 产品输出

Blueprinting 应输出：

- 排序并经过 Pareto filtering 的 architecture blueprint；
- 按 compute、memory 与 communication resource 分解的 bottleneck/utilization report；
- 带 uncertainty 的 latency、throughput、memory、energy、area 与 cost envelope；
- 说明哪些硬件变化真正有价值的 sensitivity/what-if analysis；
- 与 workload/architecture identity 关联的 simulation trace；
- 可复现实验 manifest 与 evidence provenance；
- 可选的 runtime configuration 或由选定 plan 派生的 target program。

Program emission 很有价值，特别适合未来验证 LPU，但它是“选定 blueprint 可实现”的下游证明，不是项目定义。

## 以形式化推导作为技术基础

如果每个 candidate 使用不同 workload formula、overlap assumption 或 simulator schema，硬件实验很容易失效。因此 Blueprinting 把问题表达为分阶段的形式化推导：

```text
workload semantics
  -> distributed work and exact volumes
  -> portable mapping intent
  -> architecture-bound resources and schedule
  -> simulation trace and optional program
```

每次 transition 只拥有一类明确决策，保持上游 workload meaning，消解显式约束，并发布可验证 checkpoint。由此，每个 candidate 都使用同一 workload truth、稳定 lineage、显式 binding point 与经过检查的 execution plan。

实现中使用 IR 编码 typed formal state，以 lowering 实现 refinement，并通过 pass 让 analysis/transformation transaction 可观察。这些术语来自编译工程；这里真正的方法论是形式化建模、推导、验证与自动分析。详细 contract 位于 **形式化分析基础**。

## 当前实现边界

当前仓库已经建立较强的 workload accounting 与形式化分析基础：Transformer training 可以推导为 target-neutral portable plan，系统可以产生 evidence-backed analytical estimate，并可复现 Calculon experiment。

当前还没有 first-class `ArchitectureBlueprint`、广泛 hardware design-space search、贯通的 discrete-event simulator、energy/area/cost model 或 Pareto exploration。这些是下一批定义产品的纵向切片，记录在[实现状态](../project/status.md)与[路线图](../project/roadmap.md)中。

接下来阅读[硬件设计空间](design-space.md)，然后沿[探索工作流](workflow.md)继续。交互式探索入口见[探索工作空间](workspace.md)。

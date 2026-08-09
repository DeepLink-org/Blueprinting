---
template: home.html
title: Blueprinting — 硬件架构探索与仿真
description: 面向分布式 AI 工作负载的证据驱动硬件架构探索与仿真系统。
hide:
  - toc
---

# Blueprinting

Blueprinting 是一个面向分布式 AI 工作负载的证据驱动硬件架构探索与仿真系统。它把候选计算、存储、互连和系统设计转换为可比较的 simulation blueprint，帮助架构师在投入硬件实现前理解瓶颈与权衡。

这个名字表达了产品本身：创建一份精确 architecture blueprint，把真实 workload 映射到蓝图上，以多种 fidelity 检验它，再用 evidence 持续细化。

![Blueprinting 硬件架构探索闭环](assets/architecture/hardware-exploration-loop.svg)

## Blueprinting 探索什么

| 设计领域 | 示例问题 |
|---|---|
| Compute substrate | Engine 类型/数量、precision、array shape、dataflow 与 useful utilization |
| Memory hierarchy | Capacity、bandwidth、banking、placement、reuse 与 lifetime pressure |
| Interconnect | NoC、die-to-die、scale-up/scale-out topology、routing、collective 与 congestion |
| System organization | Chiplet/package/node/cluster 组合以及 power、area、cost constraint |
| Workload mapping | Parallelism、tiling、fusion、recomputation、placement、scheduling 与 buffer |
| Architecture choice | Pareto frontier、bottleneck、sensitivity、uncertainty 与 robustness |

答案不只是 latency number，而是一份可复现 bundle：architecture parameter、workload fact、mapping decision、evidence revision、simulation trace、metric、diagnostic 与 claim boundary。

## 探索工作流

```text
architecture question + workload suite
  -> candidate hardware blueprints
  -> legal workload mappings
  -> evidence-backed analytical / network / hardware simulation
  -> bottleneck, sensitivity, and Pareto analysis
  -> measurement and calibration
  -> revised blueprint candidates
```

现有 GPU、未来 LPU 与其他 accelerator 都是 architecture candidate。探索早期允许 hardware 只被部分指定；随着 capability、topology、simulator 与 ABI 逐渐成熟，再逐层完成 binding。

## 方法：形式化推导与自动分析

Blueprinting 首先把架构问题转换为一组显式形式化对象：workload 语义、候选硬件的 capability/resource、mapping 决策、evidence revision、objective 与 constraint；随后逐步推导更具体的模型，并在每个边界检查 workload 守恒、合法性、容量、依赖、生命周期与 provenance 等义务。

自动分析在这些已验证模型上判断可行性、解析 evidence-backed cost envelope、定位瓶颈与 critical path、评估 sensitivity/uncertainty，并构造 Pareto comparison。Immutable checkpoint 允许 analytical model、network/hardware simulator 与 profiler observation 检查每层推导结果，而不改写其语义。

实现中会在合适位置借用 typed IR、staged lowering、transactional pass 与 verifier 等编译工程技术。在 Blueprinting 中，它们是表达和检查推导的技术手段，不是一个名为 `Compiler` 的系统组件，也不是产品定义。

## 当前实现

当前实现已经拥有较强基础，但尚未交付完整 exploration product：

- **Implemented：** typed Transformer training workload analysis、精确 operation/byte/collective fact、target-neutral portable plan、derivation-level observability、evidence-backed analytical estimate 与 Calculon calibration。
- **Contract foundation：** 支持逐步 mapping 到 concrete execution 与可选 target program 的五层 typed formal representation。
- **Planned product slices：** first-class architecture blueprint、design-space generation、target/resource binding、discrete-event simulation、network/hardware simulator adapter、energy/area/cost model、Pareto search 与 profiler feedback。

在把 design contract 当作已贯通功能前，请检查[实现状态](project/status.md)。

## 从这里开始

1. 阅读[为什么叫 Blueprinting](exploration/index.md)，理解产品判断与名字。
2. 研究[硬件设计空间](exploration/design-space.md)。
3. 沿[探索工作流](exploration/workflow.md)构造实验。
4. 查看[硬件架构模型](modeling/hardware.md)与[工作负载模型](modeling/workload.md)。
5. 实现 fidelity layer 时使用[性能证据](design/performance/index.md)与[仿真](design/performance/simulation.md)。
6. 需要推导、验证、表示与变换 contract 时进入[形式化分析基础](design/index.md)。

## 项目边界

Blueprinting 负责 architecture description、workload-to-hardware mapping、performance evidence resolution、simulation orchestration、design-space analysis 与可复现实验产品。它集成而不重新实现详细 hardware/network simulator、kernel、collective library、RTL tool、driver 与 runtime stack。

可选 program emission 用于证明被选 blueprint 最终能在 LPU、GPU 或其他 target 上执行。它是同一 verified plan 的一种下游实现；architecture exploration 与可解释分析仍然是主要产品。

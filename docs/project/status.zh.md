# 实现状态

本页区分 Blueprinting 的硬件探索产品目标与仓库中已经贯通的工程基础。Schema 或 design contract 是有价值的基础，但只有 candidate 能够被端到端构造、评估与消费时，才算真正的 exploration capability。

**状态日期：** 2026-08-09

## 状态词汇

| 标记 | 含义 |
|---|---|
| **Implemented** | 仓库中存在可运行 production code 与相称测试 |
| **Implemented experiment** | 存在可复现 validation path，但尚不是通用 public product API |
| **Experimental Contract** | Schema/verifier 可运行，但 producer、consumer、语义覆盖或兼容性 Gate 尚未贯通 |
| **Contract Only** | Schema、verifier、protocol 或 fixture 已存在，但 production path 未贯通 |
| **Planned** | 设计方向已接受，但实现或稳定 contract 尚不存在 |

## 产品能力矩阵

| 硬件探索能力 | 状态 | 当前证据或差距 |
|---|---|---|
| Typed Transformer training workload accounting | **Implemented** | 精确 block operation、byte、collective、recomputation 与 phase |
| Target-neutral workload/mapping plan | **Implemented slice** | Transformer path 到达 `PortablePlanIR` |
| 版本化 compute/memory/network efficiency profile | **Implemented adapter** | `HardwareProfile` 与两种 analytical estimate mode |
| Calculon/SeqSel workload 与 cost calibration | **Implemented experiment** | 8 case 可复现 report 与 test |
| First-class hierarchical `ArchitectureBlueprint` | **Planned** | 已定义 component model；无 production schema/API |
| Hardware design variable 与 constraint-aware candidate generation | **Planned** | 无 design-space generator 或 search session |
| Workload suite 与 scenario weighting | **Planned** | 当前路径评估显式 individual configuration |
| Architecture capability/legalization model | **Planned** | target/deployment binding 已有；通用 plugin path 尚无 |
| Architecture-bound placement、schedule 与 memory plan | **Experimental Contract / Planned** | `ConcretePlanIR` 只有通用 queue-oriented schema 与 structural verifier；producer、route/occupancy semantic 和 typed target extension 尚无 |
| Discrete-event compute/memory/resource simulation | **Planned** | 当前结果是 analytical composition，不是 event simulation |
| Timeline analysis/replay bundle | **Planned** | `TimingProjection`、`SimulationTraceIR`、`TimelineBundle` 只有 design contract |
| Network/hardware simulator adapter | **Planned** | normalized provider protocol 只有文档设计 |
| Bottleneck、utilization、sensitivity 与 what-if report | **Planned** | 无通用 architecture report product |
| Energy、area、power、thermal 与 cost model | **Planned** | 已定义维度，但无 provider |
| Multi-objective Pareto architecture search | **Planned** | 无 candidate frontier API |
| Profiler observation ingestion 与 calibration revision | **Planned** | 存在 offline comparison，无 closed-loop service |
| GPU/LPU/其他 program emission | **Experimental Contract / Planned** | 只有可选下游 `MachineIR` schema；尚无 LPU capability/ABI、emitter 或 runtime contract |

这张矩阵是 user-facing claim 的权威来源。存在五个 IR class 不能被概括为已经完成 hardware simulator。

## Schema 成熟度不是能力成熟度

当前五个 IR class 使用 `1.0.0` 作为内部 canonical serialization version。这个数字不表示 public API/ABI 已冻结，也不表示每层都有 production producer 和 consumer。特别是 `ConcretePlanIR` 与 `MachineIR` 仍是 experimental scaffold；它们必须通过[风险登记表](risks.md)中的 compatibility Gate 才能升级为 stable contract。

## 已贯通分析路径

当前可运行切片是：

```text
TransformerModelSpec + TransformerExecutionSpec
  -> exact workload decomposition
  -> ModelIR
  -> DistributedTaskIR
  -> PortablePlanIR
  -> HardwareProfile analytical estimate
  -> Calculon / paper comparison report
```

`PassManager` 验证每次 staged derivation 并暴露 immutable checkpoint。`HardwareProfile` 在 portable plan 后提供 evidence。当前路径尚未构造 architecture hierarchy、绑定 physical resource、执行 discrete-event simulation 或搜索 hardware candidate。

## 当前结果可以声称什么

仓库可以声称：选定 Transformer training workload 被分解为可审计 target-neutral work，并在不使用 case-specific timing coefficient 的前提下与一个版本化 system evidence profile 比较；系统也提供了后续 simulation correlation 所需的 stable identity、verifier gate 与 pass-level checkpoint hook。

当前还不能声称 Blueprinting 已经探索 compute/memory/interconnect parameter、预测 NoC/network contention、建模 energy/area/cost、构造合法 concrete hardware schedule、产生 sensitivity/Pareto result，或用真实 GPU/LPU observation 闭合 calibration loop。

## 工程基础

| 基础 | 状态 | Source of truth |
|---|---|---|
| Immutable value、stable ID、lineage、codec、digest | **Implemented** | `src/blueprinting/compiler/{frozen,ids,codec}.py` |
| 五层 progressive formal-representation schema（`*IR`）与 verifier | **Experimental Contract** | `src/blueprinting/compiler/ir/`；只有前三层存在 production derivation slice |
| Typed workload/strategy/target/deployment binding | **Implemented** | `bindings.py`、`session.py` |
| Transactional analysis/transformation、checkpoint、observer | **Implemented** | `passes/base.py` |
| Transformer semantic frontend 与 workload algebra | **Implemented slice** | `models/transformer.py`、`analysis/transformer_workload.py` |
| Distributed/portable mapping derivation | **Implemented slice** | `lowering/transformer.py` |
| 当前 evidence adapter | **Implemented slice** | `analysis/cost_model.py` |
| Calculon experiment | **Implemented experiment** | `experiments/calculon.py` |

这些 typed representation、verifier、derivation transaction 与 analysis 构成 hardware exploration 的形式化基础。新的 architecture model、simulator provider 与 analysis product 应扩展这一份 semantic foundation，而不是建立平行 workload truth。

## 验证基线

当前 test suite 覆盖 binding consistency、canonical serialization、verifier rejection、pass transaction rollback、checkpoint observer、workload conservation 与 Calculon calibration result。文档检查强制完整双语 page pair 与 strict site build。

能力升级需要端到端 product test。例如只增加 `ArchitectureBlueprint` dataclass 仍是 Contract Only；至少要构造两个不同 candidate、映射同一 workload、产生可比较 result 并保持 provenance，才能形成产品级证据。

## 下一个产品里程碑

下一里程碑是最小 two-blueprint exploration：

```text
one Transformer workload suite
  + two parameterized virtual ArchitectureBlueprints
  + normalized HardwareProfile evidence
  -> legal architecture-bound plans
  -> deterministic resource simulation
  -> bottleneck + utilization + latency/memory comparison
  -> reproducible experiment bundle
```

这个 milestone 才真正证明“Blueprinting”这个名字：系统能够比较 hardware blueprint 并解释差异。Replayable MachineIR 可以在后续验证同一个 plan，但不是第一个 exploration MVP 的 gate。参见[路线图](roadmap.md)。

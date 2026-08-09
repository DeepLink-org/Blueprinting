# 路线图与验收 Gate

路线图围绕 hardware-exploration outcome 组织。只有形式化表示与分析基础设施带来新类别的 architecture blueprint、simulation、comparison 或 calibration 能力时，才算完成产品交付，而不是因为又增加了一个 schema。

## 产品排序原则

1. Hardware variable 变化时保持 workload semantic 不变。
2. 在广泛 target-specific code generation 前建立 first-class architecture model。
3. 每个 exploration milestone 至少比较两个实质不同 blueprint。
4. 使用足以解决决策的最低 fidelity，再选择性 refinement。
5. 在校准 hardware behavior 前先验证 workload conservation。
6. 让 bottleneck、sensitivity、uncertainty 与 provenance 成为 first-class output。
7. Optional program emission 与 simulated plan 关联，但从属于后者。

## Phase 总览

| Phase | 产品交付 | 验收重点 | 状态 |
|---|---|---|---|
| 0 — Semantic foundation | ID、codec、IR/verifier、pass transaction | deterministic、immutable、observable fact | **Complete** |
| 1 — Workload baseline | Transformer decomposition、distributed/portable mapping、exact work | conservation 与独立 Calculon validation | **Partially complete** |
| 2 — Architecture blueprint core | 分层 compute/memory/interconnect/system model、variable、constraint、normalized evidence adapter | 两个具有稳定 identity 和合法 capability 差异的 blueprint | **Planned** |
| 3 — Architecture-bound simulation | legalization、mapping、placement、typed target schedule、buffer、deterministic resource event、`TimelineBundle` | feasibility、utilization、bottleneck、latency/memory comparison | **Planned** |
| 4 — Exploration engine | bounded candidate generation、sensitivity、uncertainty、Pareto record、experiment bundle | 复现小空间 frontier 并解释 dominance | **Planned** |
| 5 — Multi-fidelity calibration | performance DB、network/hardware simulator adapter、observation 与 revision | 一致 provider semantic、event correlation、误差改进 | **Planned** |
| 6 — Program and runtime validation | target MachineIR/artifact、replay/runtime profiler correlation | emitted/replayed event 与 simulated concrete plan 满足声明的 correspondence level | **Optional downstream / Planned** |

## Phase 1 完成 Gate

Workload baseline 满足以下条件后，才足以支撑 exploration：

- intermediate buffer/lifetime 显式；
- full-model layer/stage composition 保持 lineage；
- PP/DP communication 与 boundary sharding 是结构 fact；
- 初始 training/inference scenario suite 共享 common semantic；
- conservation test 覆盖 operation、byte、message、recomputation 与 persistent state。

Portable workload baseline 不得出现 architecture latency 或 kernel identity。

## Phase 2 Architecture Gate

实现最小 `ArchitectureBlueprint`：

- compute-engine capability 与 count；
- 具有 capacity/connectivity 的两级 memory hierarchy；
- 一个 local communication fabric 与一个 scale-up fabric；
- system multiplicity/topology；
- fixed、variable、derived 与 constrained field；
- canonical identity 与 verifier；
- 将 `HardwareProfile` 适配为 versioned evidence，而不是 architecture truth。

验收要求两个实质不同 virtual blueprint 绑定同一 portable workload，对 incompatible mapping 给出 diagnostic，并保持相同 source-workload digest。

## Phase 3 Simulation Gate

在构造第一个 plan 前，先关闭 common coordination core、typed target extension、planning/evaluation evidence identity 和 runtime correspondence contract。随后为每个 accepted blueprint 构造包含 implementation、resource placement、ordering、synchronization、route、resource occupancy 与 buffer allocation 的 architecture-bound plan，并用它驱动 deterministic event simulator。

第一次 comparison 必须发布引用同一 concrete digest 的 `TimelineBundle`，并报告 feasibility、total latency、throughput basis、memory high-water mark、per-resource utilization、critical path、queue delay、uncertainty 与 bottleneck attribution。在接入详细 simulator 前，用可手算 DAG 与现有 analytical estimator 作为 oracle；用 queue-centric 和 non-queue-centric virtual target 检查 common core 是否过拟合。

## Phase 4 Exploration Gate

先穷举小型 design space 建立 ground truth，再引入 hierarchical pruning 和更智能 search，同时保留：

- candidate generator revision 与 variable assignment；
- hard-constraint rejection reason；
- mapping/evidence fingerprint；
- objective 与 uncertainty policy；
- search seed、budget 与 evaluation fidelity；
- sensitivity result 与 Pareto dominance evidence。

当用户能够提出 hardware question、获得可复现 frontier，并理解哪个 architecture resource 移动了每个 bottleneck 时，这个 milestone 才达成。

## Phase 5 Calibration Gate

把 analytical model、performance database、network simulator、hardware simulator 与 measurement 统一到相同 request/result semantic，增加 fidelity-aware resolution 与显式 interpolation/extrapolation policy。

将 observation 关联到 architecture component 与 concrete command。Calibration 创建 immutable evidence revision，并且必须在不使用 model/case-specific correction factor 的前提下改进 held-out prediction error。历史 experiment bundle 保持可复现。

## Phase 6 可选 Program Gate

Program emission 不是 architecture-exploration MVP 的必要条件。LPU、GPU 或其他 backend 按 Architecture、Simulation、Replay、Executable 四级成熟；只有最后一级要求真实 hardware/ABI。Backend lowering 与 simulation 相同的 verified concrete-plan envelope。

验收要求 command/instruction lineage、ABI manifest、target verifier、failure policy 与 profiler correlation。Timing difference 形成新 evidence；emitter 不得引入独立 high-level schedule。Target-enforced cycle/slot 必须来自 typed extension，而不是把 predictive timeline 直接编码成 correctness。

## Exploration MVP 定义

满足以下条件时，核心产品才可信：

1. 一个 workload suite 映射到至少两个实质不同 architecture blueprint；
2. compute、memory 与 communication resource 显式且经过验证；
3. simulation 产生 resource-correlated performance/memory behavior；
4. bottleneck/sensitivity analysis 解释 candidate 差异；
5. 有界 multi-objective search 复现已知小空间 Pareto frontier；
6. 每项结论记录 workload、blueprint、mapping、evidence、simulator 与 policy revision；
7. 改变 evidence 时重新评估 result，而不改变 workload/architecture semantic。

Hardware program emission 会增强验证，但不属于这个 minimum definition。

## 近期交付顺序

1. 定义最小 hierarchical `ArchitectureBlueprint` 与 verifier。
2. 把 `HardwareProfile` 包装到 normalized evidence request/result 后。
3. 增加两个 parameterized virtual blueprint 与 architecture legality。
4. 完成 resource mapping 所需的 portable buffer/lifetime。
5. 用 queue-centric/non-queue-centric virtual target 冻结 common coordination core 与 typed extension boundary。
6. 构造最小 architecture-bound plan、deterministic simulator 与 `TimelineBundle`。
7. 产生 two-blueprint bottleneck/utilization comparison。
8. 增加 bounded variable、exhaustive small-space Pareto analysis 与 sensitivity。
9. 接入 network/hardware simulator provider，再增加 optional target emission。

每项交付都要更新双语产品叙事、[实现状态](status.md)与[风险登记表](risks.md)。Roadmap 文字永远不能作为 implementation evidence；Timeline 的完整语义见[阶段路径](../design/timeline-path.md)。

# 形式化分析模块架构

模块边界遵循 hardware-experiment ownership，而不是单纯按照源码放置方便程度划分。每个模块拥有 typed input/output 和有限的决策权，发布 diagnostic 与 provenance，并且不能直接访问下游状态。

## Exploration Facade 与推导 Context

未来 public facade 应暴露 architecture exploration，而不是一条 IR pipeline：

!!! note "当前不可调用"
    `ExplorationSession` 与 `blueprinting.explore()` 是设计目标，当前仓库尚未实现。以下代码是说明性伪代码。

```python
experiment = ExplorationSession(
    workloads=WorkloadSuite(...),
    architecture_space=ArchitectureSpace(...),
    mappings=MappingSpace(...),
    objectives=(latency, energy, area),
    constraints=(memory_capacity, power_envelope),
    evidence_snapshot=EvidenceRevision(...),
    fidelity_policy=FidelityPolicy(...),
    seed=0,
)

result = blueprinting.explore(experiment)
result.pareto_blueprints
result.bottlenecks
result.sensitivity
```

内部会为每个 candidate 创建 immutable typed derivation context，用于 workload mapping、architecture binding 与 analysis addressing。当前实现把这个对象命名为 `SynthesisSession`；该 class 及其 workload/strategy binding 已实现，而 `ExplorationSession` 与 end-to-end product facade 仍为 planned。Global mutable configuration 被禁止，因为它会破坏 experiment reproducibility。

## Workload、Mapping 与 System 领域模型

`blueprinting.workload` 拥有 target-neutral model semantic 与 request/training scenario。Workload object 不得包含 parallel placement、chip name、peak rate、empirical latency、kernel identity 或 physical placement。当前 slice 提供 typed Transformer model、training-workload 与 inference-request contract。

`blueprinting.mapping` 拥有 TP/PP/DP、recomputation、collective form 等 target-neutral logical strategy，也拥有 `NetworkTierBinding` 这类显式 deployment-side association。后者只在 portable planning 之后提供给 evaluation，绝不嵌入 workload fact 或 `PortablePlanIR`。因此同一份 portable plan 可以在实质不同的 system 上评估。

边界 importer 仍接受保留的 Calculon 风格字段名，但 alias 不是第二套 schema：canonical 与 legacy 拼写同时出现时必须一致，否则在 derivation 前拒绝导入。新构造的 domain object 与 report 只使用 canonical ownership 和 typed field。

`blueprinting.system` 拥有 immutable chip-local compute engine、memory capacity/bandwidth、interconnect tier、collective volume rule 与导入 evidence revision。`SystemProfile` 是当前有限的 compute/memory/network adapter；它还不是产品设计中的 hierarchical `ArchitectureBlueprint`、physical deployment 或 target binding。Cost policy 继续属于 `analysis`：system contract 暴露 peak 与 evidence-bearing facts，但不选择 calibration mode。

这三个 package 是权威 domain input，不是另一套 IR hierarchy。只有 synthesizer frontend 把 workload 与 logical-strategy contract 导入 `ModelIR` 和 typed `SynthesisSession` 后，canonical derivation 才开始；system profile 与 deployment-side network binding 继续位于 canonical workload state 之外，只能被显式 analysis 或后续 target binding 消费。

## Frontend

Frontend 解析 model 或 framework input，验证 target-independent type/effect，分配 stable identity，并输出 `ModelIR`。它拥有 import diagnostic 和 source mapping。

Frontend 不读取 peak throughput、kernel catalog、physical topology 或 runtime observation。Framework adapter 暴露 canonical IR，而不是平行的 public IR hierarchy。

当前 frontend 覆盖 typed decoder-only Transformer training，以及带显式 KV-cache semantic 的 static inference prefill/decode。更多 model family、framework importer 与 online serving scenario 属于后续工作。

## Canonical 形式化表示基础设施

`blueprinting.schema` 提供所有 typed contract 共享且无领域依赖的 canonical codec、frozen map 与 serialization error。`blueprinting.synthesizer` 中的 representation core 具体类型目前使用 `*IR` 后缀；它提供 `NodeId`/`ValueId`、typed lineage、exact scalar expression、schema header、feature set、deterministic digest 和 verifier diagnostic。

它不依赖 Transformer-specific derivation、target plugin、performance provider 或 simulation。Typed extension 可以承载 namespaced semantic；free-form metadata 不具有 compatibility meaning。

## Analysis 与 Transformation 基础设施

`PassManager` 执行 declarative `PassContract`。每个 contract 声明 input/output schema、required binding/analysis、preserved/produced analysis、mutation model、verification policy、determinism，以及带可执行 predicate 的 typed lineage rule。跨 boundary 验证属于 commit gate；deterministic replay 在 CI 启用，也可以在 runtime 显式开启。

`AnalysisStore` 通过 representation digest、analysis key 和 session fingerprint 进行 content addressing。Checkpoint observer 在 analysis 原子发布前检查 verified immutable output。详见[分析与变换基础设施](passes/index.md)。

## Planning 与 Search

Planning 在 hardware blueprint variable、workload scenario、mapping strategy、recomputation/pipeline policy、implementation choice 和 concrete schedule 上生成、过滤和排序 candidate。

Architecture binding 前的 pruning 使用 target-neutral work、communication、memory、dependency measure 与 safe architecture constraint。Binding 后的 objective 可以包含 latency、throughput、utilization、capacity、energy、area、cost 与 uncertainty。每个 candidate/rejection 都保留 blueprint identity、parent digest、rule、policy、fidelity、budget 与 seed。

Candidate generation 和 Pareto API 尚在规划；当前 Transformer slice 接收一个显式 strategy。

## Architecture 与 Deployment 子系统

计划中的 `ArchitectureBlueprint` 定义 component hierarchy、compute/memory/interconnect resource、design variable、physical constraint 与 capability。在该 schema 落地前，现有 `TargetBinding`/`TargetProfile` concept 作为代码桥接。`DeploymentProfile` 定义具体 device、topology、link、capacity、reservation、host 与 environment revision。

`TargetPlugin` 组合独立 protocol：

```text
CapabilityModel
LegalityModel
ImplementationSelector
EstimateProviders
ResourceModel
MachineLowering
ArtifactEmitter
ProfilerAdapter
```

在硬件或 emitter 尚不可用时，plugin 可以先支持 architecture、legality、evidence 与 simulation。Common mapping core 拥有 portable workload semantic 与 concrete coordination contract；target-only dataflow、route、issue 或 protocol semantic 由 typed extension 和 target verifier 拥有。

## Resource Scheduler 与 Memory Planner

Scheduler 消费 target-legal task、deployment resource 和 cost view，联合决定 implementation instance、placement、ordering、dependency、synchronization、route、resource occupancy、buffer space、offset 和 reuse。

Memory planner 必须分析合法 overlap 下的 lifetime，而不只是 aggregate peak-memory 公式。输出必须通过 DAG、queue、sync、buffer、capacity 和 target-legality verifier，才能成为 `ConcretePlanIR`。

Experimental contract 现在包含互斥的 queue-order 与 slot/dataflow typed extension、target verifier 和 deterministic virtual reference binder。它们用于验证 common envelope 能同时承载 queue-centric 与 queue-free semantic；production target plugin、resource scheduling、occupancy 和硬件 legality 仍未实现。

## Product、Simulation 与 Emission

Product module 从 verified plan 派生特定用途表示：

| 模块 | 产物 | 权威性 |
|---|---|---|
| Cost projection | `CostedTaskView` | 可重建 analysis |
| Timing | `TimingProjection` | 可重建 analysis |
| Simulator adapter | `SimulationTraceIR` | Derived interchange |
| Timeline packaging | `TimelineBundle` | 带 provenance 的发布/replay artifact |
| Architecture evaluation | `EvaluationReport` | Bottleneck、utilization、objective、uncertainty |
| Exploration analysis | Pareto/sensitivity record | Candidate-comparison artifact |
| Machine lowering | `MachineIR` | Target program semantic |
| Emitter | `ProgramArtifact` | 发布的 executable/replay package |

Simulation 是主要 hardware-exploration product path。Optional emission 不能独立重新调度；两者都消费同一 `ConcretePlanIR` envelope 与 typed target extension。

## Observation 与 Calibration

Profiler adapter 把 runtime event 关联到 machine instruction 与 concrete command。Normalization 产生 immutable `ObservationSet`。Calibration 拟合 domain-scoped model 并发布新的 `CalibrationRevision`，不覆盖 raw evidence。

同一套 lineage 支持从 model operation 到 runtime event 的正向与反向查询。

## 依赖方向

```text
schema ──► workload ──► mapping
   │          │           │
   ├──────────┴───────────┴──► synthesizer ──► PortablePlanIR
   │                                      │             │
   └──► system ───────────────────────────┼──► analysis ◄── evidence
                         NetworkTierBinding             │
                                                       ▼
                                      application / validation
```

依赖方向是显式的：workload contract 不依赖 mapping 或 system description；logical mapping 可以针对 workload shape 做验证，但不能读取 system；system description 不依赖 analysis policy；analysis 不构造 canonical plan。Synthesizer 物化 workload/plan fact，analysis 再使用显式 system、deployment mapping 与外部 evidence 评估这些事实。Validation 可以消费整个 supported stack，但 production layer 不得依赖 validation 或外部 oracle。

## 当前源码映射

| 关注点 | 源码 | 状态 |
|---|---|---|
| Canonical codec 与 frozen schema value | `schema/` | Implemented |
| Model 与 workload semantic | `workload/` | Implemented Transformer slice |
| Logical strategy 与显式 deployment mapping | `mapping/` | Implemented Transformer/network slice |
| Chip、memory、interconnect 与聚合 system profile | `system/` | Implemented limited profile adapter |
| ID、expression、lineage | `synthesizer/{ids,expr}.py` | Implemented |
| Canonical 形式化表示（`*IR`） | `synthesizer/stages/*/ir.py` | Implemented contracts |
| Binding 与 session | `synthesizer/{bindings,session}.py` | Implemented |
| Analysis/transformation transaction | `synthesizer/passes/base.py` | Implemented |
| Workload-to-IR/session frontend | `synthesizer/frontend/` | Implemented Transformer slice |
| Transformer exact-work dialect | `synthesizer/dialects/transformer/` | Implemented training/inference slice |
| Stage-owned derivation pass | `synthesizer/stages/*/passes.py` | Implemented through portable plan |
| 当前 system cost adapter | `analysis/cost_model.py`、`analysis/cost/` | Implemented slice |
| Framework-neutral orchestration 与 report | `application/` | Implemented static analysis slice |
| Calculon/Vidur comparison 与 regression gate | `validation/` | Implemented offline gate |
| Optional external performance bundle | `data/evidence/` | 显式加载；从 base package 排除 |
| Architecture model/search、evidence service、simulation、emission | Accepted boundary | Planned |

`validation/calculon.py` 与 `validation/vidur.py` 把外部 reference implementation 隔离在推导后的 comparison boundary；`validation/regression.py` 冻结其严格 drift gate。它们属于 comparison check，不能证明 canonical derivation 天然正确。

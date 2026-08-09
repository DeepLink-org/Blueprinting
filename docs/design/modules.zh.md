# 形式化分析模块架构

模块边界遵循 hardware-experiment ownership，而不是单纯按照源码放置方便程度划分。每个模块拥有 typed input/output 和有限的决策权，发布 diagnostic 与 provenance，并且不能直接访问下游状态。

## Exploration Facade 与推导 Context

未来 public facade 应暴露 architecture exploration，而不是一条 IR pipeline：

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

内部会为每个 candidate 创建 immutable typed derivation context，用于 workload mapping、architecture binding 与 analysis addressing。当前实现把这个对象命名为 `CompilationSession`；该 class 及其 workload/strategy binding 已实现，而 `ExplorationSession` 与 end-to-end product facade 仍为 planned。Global mutable configuration 被禁止，因为它会破坏 experiment reproducibility。

## Frontend

Frontend 解析 model 或 framework input，验证 target-independent type/effect，分配 stable identity，并输出 `ModelIR`。它拥有 import diagnostic 和 source mapping。

Frontend 不读取 peak throughput、kernel catalog、physical topology 或 runtime observation。Framework adapter 暴露 canonical IR，而不是平行的 public IR hierarchy。

当前 frontend 覆盖 typed decoder-only Transformer training。更多 training architecture、inference prefill/decode、KV-cache semantic 和 framework importer 属于后续工作。

## Canonical 形式化表示基础设施

Representation core 的具体类型目前使用 `*IR` 后缀；它提供 immutable value、`NodeId`/`ValueId`、typed lineage、exact scalar expression、canonical JSON、schema version、feature set、deterministic digest 和 verifier diagnostic。

它不依赖 Transformer-specific derivation、target plugin、performance provider 或 simulation。Typed extension 可以承载 namespaced semantic；free-form metadata 不具有 compatibility meaning。

## Analysis 与 Transformation 基础设施

`PassManager` 执行 declarative `PassContract`。每个 contract 声明 input/output schema、required binding/analysis、preserved/produced analysis、mutation model、verification policy 和 determinism。

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

当前只有 queue-oriented experimental schema 与 structural verifier；typed target extension、production target binding 和 scheduling 尚未实现。

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
frontend ───────► ir/common
                    │
lowering/passes ────┼────► planning
bindings/session ───┘          │
                               ▼
                    architecture binding
                               │
              evidence ◄───────┼──────► scheduling
                               │
                               ▼
                    simulation / emission
                               │
                               ▼
                    observation / calibration
```

Canonical IR、binding、pass 与 lowering 基础设施位于 `src/blueprinting/compiler/`。分析子系统则是同级的 `src/blueprinting/analysis/`：compiler 产出显式 workload 与 plan facts，analysis 再用解析模型和外部证据评估这些事实。Analysis 可以依赖 canonical compiler contract，但调用方不能把 cost evidence 当作隐式 lowering 决策。

## 当前源码映射

| 关注点 | 源码 | 状态 |
|---|---|---|
| ID、expression、codec、frozen value | `compiler/{ids,expr,codec,frozen}.py` | Implemented |
| Canonical 形式化表示（`*IR`） | `compiler/ir/` | Implemented contracts |
| Binding 与 session | `compiler/{bindings,session}.py` | Implemented |
| Analysis/transformation transaction | `compiler/passes/base.py` | Implemented |
| Transformer frontend | `compiler/models/` | Implemented slice |
| Workload 与 cost analysis | `analysis/` | Implemented slice |
| Transformer derivation pass | `compiler/lowering/transformer.py` | Implemented through portable plan |
| 当前 hardware evidence adapter | `analysis/cost_model.py`、`analysis/cost/` | Implemented slice |
| Architecture model/search、evidence service、simulation、emission | Accepted boundary | Planned |

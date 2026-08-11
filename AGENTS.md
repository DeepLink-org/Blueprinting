# Blueprinting 项目执行纲领

> Blueprinting 是面向分布式 AI 工作负载的、证据驱动的硬件架构探索与仿真系统。
> 核心方法是形式化建模、逐步推导、约束验证与自动分析；IR、lowering、pass 等是借用的编译工程技术，不是产品身份或独立 Compiler 组件。
> `docs/exploration/` 定义产品目标，`docs/design/` 定义形式化分析基础；代码、测试和其他文档不得定义第二套表示语义。

## 1. 核心定位

Blueprinting 在离线探索阶段形式化模型语义与硬件蓝图，推导分布式策略和资源计划，完成目标绑定、
计划搜索与验证；仿真、可选 hardware program 生成和 profiler 观测共享同一份已验证计划。

核心闭环是：

```text
Formal Modeling -> Verified Derivation -> Architecture Binding
  -> Simulation / Optional Program Realization -> Observation -> Calibration
```

- 硬件架构洞察、可解释比较与可复现实验是核心产品；形式化推导和 planning 是实现方法。
- 硬件必须迟绑定；GPU、LPU 与其他目标通过插件接入。
- 性能数据库、解析模型、硬件模拟器、网络模拟器和 profiler 通过统一证据面接入。
- 校准只更新有版本的证据模型，不修改已经发布的语义表示。

## 2. 当前 canonical 形式化表示主干

仓库当前采用下面五层 canonical 形式化表示；当前代码类型使用 `*IR` 后缀。不得建立平行 public hierarchy；如果未来证据表明需要新增或合并稳定边界，必须通过 ADR、迁移方案和 producer/consumer conformance，而不是把“五层”当成不可修订的事实：

```text
ModelIR
  -> DistributedTaskIR
  -> PortablePlanIR
  -> [TargetProfile + DeploymentProfile]
  -> ConcretePlanIR
  -> target MachineIR
```

### ModelIR

负责模型值、类型、显式数据流、操作和副作用。不得包含并行 placement、硬件吞吐或预测时间。

### DistributedTaskIR

负责逻辑 mesh、sharding、collective 和分布式依赖。不得包含物理设备身份或目标实现选择。

### PortablePlanIR

负责已选策略的 task DAG、精确 workload facts、抽象 buffer、资源需求和目标。它是迟绑定前的可移植
计划，不得包含 kernel ID、物理 queue、经验 duration 或 wall-clock timestamp。

### ConcretePlanIR

负责目标合法化后的 execution-plan envelope。Common coordination core 表达 implementation、物理 placement、ordering、buffer region、同步和 command DAG；dataflow、route、issue slot 等 target-only correctness semantic 必须进入 namespaced typed extension。预测时间不是执行正确性的事实源。

### MachineIR

负责一个具体 target plugin 的指令、section、entry point 和 ABI。不同硬件可以拥有不同 machine dialect，
不得为了统一表面形式而把 target 语义上提到 portable 层。

## 3. Derived views 不是 canonical 形式化表示

以下对象必须是可失效、可重建且有 provenance 的派生产物：

- `CostedTaskView`
- `TimingProjection`
- `SimulationTraceIR`
- `TimelineBundle`
- `EvaluationReport`
- `ObservationSet`

精确 FLOPs、bytes、shape 和 collective payload 属于 canonical representation。目标相关 latency、efficiency、uncertainty
和 fallback 属于 evidence/cost view。任何估算器都不得把预测结果回写成 workload facts。

## 4. 推导、变换与 profiler 契约

形式化推导规则在当前实现中通常由 pass/lowering 承载。每个 pass 必须声明：

- 输入/输出 IR 类型及 schema 范围；
- 所需 binding 和 analysis；
- preserved/invalidated analyses；
- mutation model、验证策略、确定性和 seed；
- 失败诊断与 provenance。

每个 derivation boundary 必须：

1. 生成不可变结果；
2. 在 commit 前运行 verifier；
3. 保留 stable ID 和 lineage；
4. 生成 canonical digest；
5. 通过 `PassCheckpoint` 暴露给 observer、validator 和差分测试；只有 target-bound event 才由 profiler adapter 归一化关联。

Pass pipeline 依据声明式 contract 编排，不得依赖具体 Python pass 类的硬编码判断。这里的 pass 是 verified analysis/transformation transaction，不代表系统中存在 Compiler 组件。

## 5. Binding 原则

Binding 分为独立维度：workload、strategy、target、deployment、calibration。

- workload/strategy 可以逐步特化，但必须显式记录在 typed derivation context；当前代码名为 `SynthesisSession`；
- target/deployment 只能在 portable plan 之后进入；
- 同一个 `PortablePlanIR` 必须能绑定到多个实质不同的硬件目标；
- target 变化不得改变 `ModelIR`、`DistributedTaskIR` 或 `PortablePlanIR` digest；
- 缺失 binding 必须在 derivation gate 失败，不能读取隐式全局状态补齐。

## 6. Performance Evidence Plane

统一 estimate request 至少包含：

- operation/collective identity；
- exact workload facts 与数据类型；
- target、deployment、runtime/library revision；
- requested fidelity、uncertainty policy 和 measurement protocol。

统一 estimate result 至少包含：

- latency/resource distribution 或区间；
- provider 类型与 revision；
- 输入 digest、置信度、样本量和环境信息；
- cache status、fallback reason 和适用域。

证据优先级由 `CostResolver` 显式决定。禁止模型名、case ID、对照结果、逐 case correction factor 或不可追溯
的 overlap ratio 进入 cost model。低成本解析模型用于剪枝，高成本数据库/模拟器/实测用于候选精化。

## 7. Simulation 与 program emission

```text
ConcretePlanIR envelope
  ├── TimingProjection -> SimulationTraceIR -> EvaluationReport
  └── target realization -> MachineIR -> program artifact
```

两条路径必须消费同一个 concrete digest、command DAG 和 typed target extension。执行器不得依赖 sleep 到预测时间，
也不得重复无界 global planning；它可以执行 target contract 允许的 bounded backpressure、retry、timeout 和 failure
handling。若 LPU 的 issue cycle/slot 具有 correctness 含义，它在 target binding 后进入 typed extension 与 MachineIR，
不能混入 predictive `TimingProjection`。LPU 尚未具备硬件契约时，可以先实现 architecture、legality、evidence
和 simulation maturity，无需伪造 executable backend。

## 8. Search 与 e-graph

- 帕累托前沿是离线探索产物；目标、约束、搜索预算、剪枝和 uncertainty policy 必须可复现。
- 先做 target-neutral strategy frontier，再按具体 target/deployment 生成 target-bound frontier。
- equality saturation 可以生成语义等价或分布式等价候选，但 e-graph 不是 canonical IR，也不负责最终资源调度。
- 所有候选在进入下一层前必须提取为 canonical IR 并通过 verifier。

## 9. 当前实现边界

无领域依赖的 codec 与 immutable schema primitive 位于 `src/blueprinting/schema/`；target-neutral workload contract
位于 `src/blueprinting/workload/`，逻辑策略与显式 deployment mapping 位于 `src/blueprinting/mapping/`，芯片、memory、
interconnect 与 system profile 位于 `src/blueprinting/system/`；canonical 表示与形式化推导机制位于
`src/blueprinting/synthesizer/`，分析与证据评估位于同级 `src/blueprinting/analysis/`，外部 baseline 与回归 gate
位于 `src/blueprinting/validation/`。Synthesizer 表示 formal plan synthesis 的实现边界，不是产品身份、RTL 综合器
或独立 Compiler 组件。已经实现：

- 五层 canonical IR 的 immutable schema、serialization 和 structural verifier；其中后两层仍是 experimental contract；
- stable ID、lineage、typed scalar expression、binding/session；
- pass contract、analysis cache/invalidation 和 derivation checkpoint；
- decoder-only Transformer workload contract 与 training/inference frontend adapter；
- compute、memory、interconnect 和聚合 `SystemProfile` contract；
- `ModelIR -> DistributedTaskIR -> PortablePlanIR` 的 TP、recompute、workload 与 buffer derivation；
- peak-only / system-evidence cost view 和 Calculon/SeqSel 校准实验。

尚未完成：

- target plugin registry 与 portable-to-concrete derivation；
- resource scheduler、memory planner、timing projection 与 trace simulation；
- CUDA/LPU/其他 MachineIR emitter；
- Pareto search、runtime execution 和真实 profiler ingestion。

文档和 API 必须如实区分“contract 已存在”与“端到端能力已实现”。

## 10. 代码与仓库规则

- 无领域依赖的 canonical codec、frozen value 与 schema error 进入 `src/blueprinting/schema/`；workload semantic/request contract 进入 `src/blueprinting/workload/`；逻辑 strategy 与 deployment mapping 进入
  `src/blueprinting/mapping/`；芯片、memory、interconnect 与 system contract 进入 `src/blueprinting/system/`；workload-to-IR adapter、canonical 表示与推导进入
  `src/blueprinting/synthesizer/`；cost/evidence analysis 进入 `src/blueprinting/analysis/`。不得新建平行表示栈。
- `SystemProfile` 是当前有限的 compute/memory/network evidence-bearing adapter，不得被描述成已经实现的完整
  `ArchitectureBlueprint`；`src/blueprinting/types/system/` 只服务 legacy calculator，新代码不得依赖它。
- Canonical codec tag 必须位于与领域 ownership 一致的 `blueprinting.*` namespace；不得新增历史 package-derived
  tag 或兼容 alias。Pre-graduation 阶段 nested type identity 不携带独立版本号，五层 IR root 统一使用 `0.0.0`；
  首次 schema increment 必须有 ADR、真实 migration、产物再生成方案和 conformance gate。
- IR 对象默认 frozen；语义字段使用 typed dataclass/enum/ID，不使用自由字典代替 contract。
- 普通应用代码不得依赖 authoring decorator；canonical schema/dialect 作者只从 `blueprinting.schema.authoring`
  使用 `record/adt/variant`，pass/target extension 作者只从 `blueprinting.synthesizer.passes.authoring`
  使用 `derivation/relation/claim`。Codec registry、manifest 与 pass registry 是内部实现，不得从 package root 转发。
- 所有公共 derivation/transformation 和 verifier 必须有 positive、negative、round-trip 与 lineage 测试。
- Python 最低版本为 3.10；不得使用只在更高版本解析的语法，除非先更新 packaging contract。
- 修改后至少运行相关 pytest 与 Ruff；文档修改运行双语一致性检查和 `mkdocs build --strict`。
- Calculon 是 retained comparison oracle，不是 formal-analysis backend，也不得向推导结果注入参考答案。
- 不得声称未实现的 Pareto、真实执行、硬件 emission 或实测精度已经完成。

## 11. 架构完成标准

当且仅当以下条件全部成立，核心架构才算端到端建立：

- 当前 canonical IR 主干均有经过 graduation Gate 的 stable schema、verifier 和迁移策略；
- 同一 portable plan 无前端变化地绑定至少两个不同目标；
- 数据库和至少一种模拟器通过同一 evidence protocol 工作；
- concrete plan 同时产出 trace 与 target artifact/replay package；
- profiler observation 能沿 lineage 回到 command、task 和 model operation；
- 小搜索空间能与穷举最优对比，大搜索报告预算、剪枝、重现信息和不确定性。

未关闭的跨模块风险及 stable-contract graduation 条件以 `docs/project/risks.*.md` 为准。文档措辞修正不能替代 producer、consumer、negative test 和可复现实验。

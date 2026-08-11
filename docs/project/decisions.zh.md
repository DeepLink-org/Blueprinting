# 设计决策与术语

本页集中索引架构承诺、被拒绝方案、未决问题和共享词汇。详细推理位于各设计章节；影响兼容性的变化必须通过独立 Architecture Decision Record（ADR）处理。

## 已接受决策

| 主题 | 决策 | 结果 |
|---|---|---|
| 产品身份 | 通过形式化推导、验证与自动分析进行硬件架构探索和仿真 | User-facing output 优先呈现 blueprint、bottleneck、sensitivity、uncertainty 与 trade-off |
| Canonical representation | 当前 semantic backbone 采用五层 typed representation；不得建立平行 public hierarchy，新增稳定边界必须通过 ADR 和 migration evidence | 所有 frontend/backend 汇入同一 derivation chain，但不把未验证的层数当成永恒事实 |
| Target binding | 只发生在 `PortablePlanIR -> ConcretePlanIR` | Portable planning 可跨 GPU、LPU 和其他 target 复用 |
| Execution truth | `ConcretePlanIR` envelope：common coordination core + typed target extension | Simulation/emission 不得重建独立 schedule |
| Timing | Predictive timing 是 derived analysis；target-enforced timing 是 binding 后的 typed target semantic | Evidence revision 可改变预测而不改变 correctness；LPU issue/slot constraint 不被抹除 |
| Evidence | Immutable、versioned 且携带 provenance | Calibration 创建 revision，而不是覆盖 facts |
| Profiler | Observer 与 evidence source，绝不修改 formal state | 每层 derivation 保持确定和可审计 |
| Transformation engine | 强类型 declarative pass contract 与 atomic publication | Verification/observation 失败不留下 partial state |
| Target extension | Protocol-composed `TargetPlugin` | Hardware emission 不存在时也可先完善 capability |
| Equality saturation | 仅用于有界 candidate generation | E-graph 不成为 canonical IR 或 physical scheduler |
| Runtime | 遵守 verified plan，并保留 contract 允许的 bounded mechanism decision | Runtime 不重复无界 global search，也不假装 backpressure/failure 不存在 |
| Schema maturity | Internal schema version 不自动构成 public compatibility promise | Producer、独立 consumer、migration 与 conformance Gate 通过后才毕业为 stable contract |
| Documentation | 同目录 suffix-based 双语 source | Navigation 与 language switching 始终按页面对齐 |
| 形式化推导 package | Python path 硬切为 `blueprinting.synthesizer` | Source ownership 对齐 formal plan synthesis；见 [ADR-0001](adr/0001-synthesizer-package.md) |
| Domain package | `blueprinting.workload` 拥有 target-neutral workload contract；`blueprinting.system` 拥有 chip/interconnect/system profile | Synthesis/analysis 消费显式 domain input，但不拥有它们；见 [ADR-0002](adr/0002-workload-system-domains.md) |
| Derivation 调试 | 五层图、相邻映射与调试包是从 checkpoint/lineage 重建的 derived trace | UI/overlay 不进入 canonical IR；见 [ADR-0003](adr/0003-derivation-debug-trace.md) |
| Canonical wire identity | 按领域归属的 `blueprinting.*` namespace，不保留废弃 alias | 序列化 identity 直接表达当前语义，旧产物在硬切边界重新生成；见 [ADR-0004](adr/0004-semantic-wire-identities.md) |
| 代数化 canonical constructor | Scalar operation 与 concrete command semantic 使用 constructor-specific ADT；preservation claim 具有可执行 evidence | 消除非法 arity/payload 组合，schema 变更通过显式 migration 完成；见 [ADR-0005](adr/0005-algebraic-expression-command-schemas.md) |
| 渐进式 typed Python | Runtime `Checked` contract 与 sealed core ADT 和可选标准 mypy analysis 共享 declaration；不使用自定义 plugin | Base installation 保留 contract check，预期失败显式化，static analysis 增加覆盖但不成为 runtime truth；见 [ADR-0006](adr/0006-progressive-typed-python-contracts.md) |

## 被拒绝方案

**一个带大量 optional field 的 universal graph。** 这种设计把 ownership/legality 交给约定，允许 target leakage，并让 unknown 一直存活到 runtime。独立 IR contract 能使每个 lowering gate 可验证。

**在 portable task 上保存 duration。** Duration 不可移植；它依赖 implementation、target、deployment、concurrency、evidence 与 revision，因此属于 cost view。

**把 predicted timeline 作为 execution source of truth。** Absolute timestamp 会混淆 prediction 与 readiness。Common dependency/resource/buffer contract 定义执行；预测 timeline 是 projection。若某个 target 的 issue cycle/slot 具有 correctness 含义，它必须在 binding 后进入 typed target extension 与 `MachineIR`，不能伪装成通用预测字段。

**为每个 model/case 增加 correction coefficient。** 这种系数可以匹配报告，却无法解释 workload、implementation、network 或 schedule behavior，也不能泛化。

**过早绑定 LPU 或 CUDA。** 在硬件 contract 稳定前，这会让 frontend/portable planner target-specific。Capability requirement 能保留 late specialization；LPU 可以先作为 architecture/simulation target 成熟，再增加 replay/executable backend。

**Simulator 与 emitter 各自调度。** 即使二者 total 匹配，也不能证明它们描述同一个 program。两者必须消费 concrete plan。

**并行维护 legacy/new IR stack。** Adapter 可以读取 legacy input，但两套 public semantic hierarchy 会造成 ownership 歧义与修复分叉。

**Compiler-first 产品口径。** 这种表述把借用的实现工具箱误当成系统本身。Blueprinting 在概念架构中没有 Compiler 组件；核心方法是形式化建模、推导、验证与自动分析。IR、lowering 与 pass 仍可作为 **形式化分析基础**内部的实现术语。

## 开放决策

以下问题被有意保持未决：

1. 是否以及何时用 binary format 补充 canonical JSON v1；
2. 哪些 dialect 应迁移到 MLIR，以及合适的成熟阶段；
3. portable command 的最终 typed extension mechanism；
4. target artifact container 与 ABI compatibility window；
5. 昂贵或不可信 simulator provider 的 process/RPC isolation；
6. correlated uncertainty 下的 Pareto dominance 与 pruning；
7. 首选 external trace interchange schema；
8. 引入 target-specific equality exploration 的边界；
9. profiler data 的 evidence retention、privacy 与 promotion policy。

每项决策在成为 serialized/public/target-plugin compatibility dependency 前都需要 ADR。当前高风险未决项及其 graduation Gate 见[架构风险登记表](risks.md)。

## ADR 规则

当提议改变 canonical schema、lowering gate、ownership boundary、serialized identity、plugin protocol、evidence semantic 或 public compatibility promise 时，必须创建 ADR。ADR 应包含 context、decision driver、considered alternative、decision、consequence、migration、validation 与 status。

Accepted ADR 除 status link 和 typo 外保持不可变。取代旧决策时创建新 ADR，并反向链接旧 ADR。Implementation status 继续独立维护：接受 ADR 不表示设计已经 Implemented。

## 规范词汇

| 术语 | 含义 |
|---|---|
| Workload | 模型语义、training/inference mode 与 workload parameter |
| Architecture blueprint | Compute、memory、interconnect、system、capability 与 physical constraint 的版本化 candidate hierarchy |
| Strategy | Parallelism、recomputation、pipelining、fusion、layout 与 algorithm choice |
| Target | 已绑定到具体 hardware/software capability 与 ABI family 的 blueprint |
| Deployment | 具体 device、topology、capacity、reservation 与 environment revision |
| Evidence | 携带 provenance 的 measurement、simulation、analytical 或 calibrated information |
| Plan | 某个 workload 到一份 blueprint 的 mapping，以及被选 implementation/execution constraint |
| Formal representation | 具有版本化 contract 与 verifier 的权威 typed model；当前 canonical type 使用 `*IR` 后缀 |
| Derived view | 不改变 source representation semantic 的可重建 projection |
| Artifact | 已发布 report、trace、executable 或 replayable package |
| Binding | 通过 typed derivation context 显式提供的 specialization fact；当前代码名为 `SynthesisSession` |
| Derivation | 解析决策、消解 obligation 并保持所需语义的 verified rule |
| Lowering | 用于实现 staged derivation 的编译工程技术 |
| Revision | Evidence、schema、analysis engine、plugin 或 product state 的 immutable identity |

英文页中的 **MUST**、**MUST NOT**、**SHOULD**、**SHOULD NOT** 与 **MAY** 具有规范效力；中文对应词是 **必须**、**不得**、**应该**、**不应该** 与 **可以**。

## 参考系统

架构参考但不意图重新实现以下相邻系统：

- [MLIR dialect conversion](https://mlir.llvm.org/docs/DialectConversion/)：渐进 legality 与 lowering；
- [IREE design roadmap](https://iree.dev/developers/design-docs/design-roadmap/)：compiler-planned resource 与 multi-target executable；
- [OpenXLA architecture](https://openxla.org/xla/architecture) 和 [StableHLO](https://openxla.org/stablehlo/spec)：portable semantic 与 target lowering；
- [TVM MetaSchedule](https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/meta_schedule.html)：measurement-driven search 与 database separation；
- [Chakra](https://github.com/mlcommons/chakra) 和 [ASTRA-sim](https://github.com/astra-sim/astra-sim)：trace interchange 与 distributed simulation；
- [Timeloop](https://github.com/NVlabs/timeloop)：accelerator mapping 与 architecture modeling。

这些 reference 只是比较对象，不能证明 Blueprinting 已实现相同能力。仓库现实以[实现状态](status.md)为准。

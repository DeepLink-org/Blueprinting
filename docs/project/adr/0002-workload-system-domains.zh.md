# ADR-0002：拆分 Workload 与 System 领域 Package

- 日期：2026-08-09
- 状态：Accepted
- 范围：domain ownership、Python package path、frontend adapter 与 system-profile naming

## 背景

Blueprinting 在两个独立 domain input 之间推导 mapping：workload 与 candidate system。原有源码结构没有表达这种对称关系。Transformer semantic/request contract 位于 `blueprinting.synthesizer.models`，让 derivation engine 看起来拥有输入领域；compute、memory 与 network profile 位于 `blueprinting.analysis.cost_model`，让 cost analysis 看起来拥有被评估的系统。

这种放置方式掩盖了 late binding，也产生了错误的依赖压力：新的 workload importer 会不断堆入 synthesizer，新的 chip/interconnect abstraction 会不断堆入 cost estimator。同时，system fact 与 analysis policy、有限 evidence profile 与计划中的 `ArchitectureBlueprint` 都难以区分。

## 决策驱动因素

- 让 workload/system 成为 exploration 显式且对称的顶层输入。
- 让 canonical derivation mechanics 与 domain contract 分离。
- 防止 cost analysis 拥有 chip、memory 或 interconnect semantic。
- 保持 target late binding，防止 system fact 泄漏进 workload state。
- 在源码重组时保留 canonical tag 与 target-neutral derivation digest。
- 避免 compatibility re-export 继续制造 ownership 歧义。

## 备选方案

**继续把 workload type 放在 `synthesizer.models`。** Import 改动最少，但 synthesizer 会同时拥有输入和 transformation semantic。

**继续把 system profile 放在 `analysis.cost_model`。** Module 数量较少，却把被评估对象和某一种 evaluation policy 耦合起来。

**增加 facade package 并 re-export 旧实现。** 这样只有新路径更好看，authority 并未迁移；新旧 package 都会继续像 source of truth。

**立即实现完整 `ArchitectureBlueprint`。** 当前数据和 consumer 尚不支持 component hierarchy、NoC、power/area/cost、design variable、legality 或 physical deployment。把有限 profile 命名成最终 architecture contract 会夸大实现成熟度。

## 决策

创建两个权威顶层 domain package：

- `blueprinting.workload` 拥有 target-neutral model semantic、request scenario 与 logical mapping intent；当前包含 typed Transformer training/inference contract。
- `blueprinting.system` 拥有 chip compute engine、memory、interconnect tier、collective volume rule，以及从现有 system data 导入的聚合 `SystemProfile`。

把 workload-to-canonical-state adapter 移入 `blueprinting.synthesizer.frontend`。这些 adapter 构造 `ModelIR` 与显式 `SynthesisSession` binding；workload contract 自身不执行这些工作。Lowering 消费 workload contract，但继续属于 synthesizer。

将 `HardwareProfile` 重命名为 `SystemProfile`，并移除 `blueprinting.analysis` 中的 re-export。Analysis 通过显式 policy argument 决定是否应用 profile efficiency evidence；system package 不导入也不选择 `CalibrationMode`。

不提供 `blueprinting.synthesizer.models` 或 `blueprinting.analysis.SystemProfile` compatibility facade。原 `blueprinting.types.system` package 已随 calculator path 删除；受支持代码直接导入 `blueprinting.system`。

最初的 package split 保留了当时已有的 codec tag。ADR-0004 取代这一选择：workload 与 system record 现在分别使用 `blueprinting.workload.*` 和 `blueprinting.system.*` 语义 identity。

`SystemProfile` 被明确限定为 evidence-bearing compute/memory/network adapter；它不是未来 hierarchical `ArchitectureBlueprint`、deployment description 或 target binding。

## 影响

- Import 会直接表达 domain ownership：workload、system、synthesis frontend 与 analysis 使用不同路径。
- Framework/model importer 可以在 `workload` 下扩展，而不会变成 derivation pass。
- Chip/interconnect contract 可以在 `system` 下演进，而不依赖 roofline/database provider。
- 现有 Python caller 必须迁移旧 package path 与 `HardwareProfile` class name。
- ADR-0004 定义后续 wire-format 边界；仍不支持 pickle/module-path compatibility。
- 当前 logical execution spec 仍混合 workload scenario 与 mapping intent。若后续拆分会改变 serialized contract，需要新的 ADR。

## 迁移

1. 从 `blueprinting.workload` 导入 Transformer contract。
2. 从 `blueprinting.synthesizer.frontend` 导入 `build_transformer_*_model_ir()` 与 synthesis-session helper。
3. 从 `blueprinting.system` 导入 `SystemProfile` 与 chip/interconnect component profile。
4. Costing API 继续位于 `blueprinting.analysis`，调用时显式传入 `SystemProfile`。
5. 不得新增对 `blueprinting.types.system` 的依赖，也不得在旧路径下重建 compatibility export。

## 验证

- Package-boundary test 要求两个顶层 domain package 存在，并拒绝 `blueprinting.synthesizer.models`。
- Ownership test 要求 frontend builder 不出现在 `workload`，`SystemProfile` 不出现在 `analysis`。
- System test 验证 chip、memory、interconnect、capacity、policy selection 与 canonical round trip。
- 现有 golden IR snapshot 和 training/inference baseline gate 验证源码迁移不改变 target-neutral semantic 或 estimate。
- Ruff、完整 pytest、双语文档一致性与 strict MkDocs build 继续作为 release gate。

## 状态

本 ADR 于 2026-08-09 被接受，约束初始 workload/system package split。完整 architecture-blueprint schema 与 logical execution spec 的进一步拆分属于独立未来决策。

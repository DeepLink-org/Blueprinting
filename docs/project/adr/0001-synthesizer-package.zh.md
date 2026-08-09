# ADR-0001：将形式化推导包命名为 Synthesizer

- 日期：2026-08-09
- 状态：Accepted
- 范围：Python package identity、public symbol、report vocabulary 与一个 serialized field name

## 背景

Blueprinting 通过把 workload semantic 与 strategy choice 逐步转化为更具体、经过验证的计划来探索硬件架构。实现借用了编译工程中的 IR、lowering、pass 与 verifier 技术，但产品不是通用编译器，概念架构中也没有独立 Compiler 组件。

历史上的 `blueprinting.compiler` package 与这个边界冲突。它同时承载 canonical representation、binding、derivation transaction、model frontend 与 validation experiment，而 evidence-backed analysis 已经提升为同级的 `blueprinting.analysis` package。继续保留旧名称，会持续把实现工具箱误解为产品架构。

这里的 **synthesis** 指 formal plan synthesis：根据显式输入与 obligation，推导出更具体且经过验证的 representation。它不表示 RTL synthesis、硬件实现、code generation，也不是 Blueprinting 的产品身份。

## 决策驱动因素

- 让 package ownership 与形式化建模、verified derivation 架构一致。
- 让 analysis/performance evidence 与 canonical state transformation 保持清晰分离。
- 在 target plugin 和外部 consumer 形成依赖前，消除有歧义的 public vocabulary。
- 在 semantic contract 未变化时保留稳定 serialized identity。
- 采用一次明确迁移，避免长期维护平行 API hierarchy。

## 备选方案

**保留 `compiler`。** Import churn 最小，但会保留产品口径错误，也让后续 module boundary 更难解释。

**使用 `formal_analysis`。** 它会与现有 `blueprinting.analysis` package 重叠，并混淆 authoritative transformation 与 rebuildable analysis。

**使用 `planner`。** Planning 只是路径的一部分；该 package 还承载 semantic frontend、canonical representation、verification、lineage 与 machine-level realization contract。

**重命名为 `synthesizer` 并提供兼容 shim。** Shim 能降低即时迁移成本，却会保留两套可发现 public hierarchy、削弱 boundary test，并鼓励已弃用身份长期存在。

## 决策

将 `blueprinting.compiler` 硬切重命名为 `blueprinting.synthesizer`，不提供 `blueprinting.compiler` import shim。`blueprinting.analysis` 保持同级 package，可以消费 canonical synthesis contract，但不得修改它们。

Public implementation symbol 使用 synthesis vocabulary：

| 旧名称 | 新名称 |
|---|---|
| `CompilationSession` | `SynthesisSession` |
| `CompilerError` | `SynthesisError` |
| `CompilerPass` | `DerivationPass` |
| `compilation_session_for()` | `synthesis_session_for()` |
| `inference_compilation_session_for()` | `inference_synthesis_session_for()` |
| `compile_transformer_block()` | `derive_transformer_block()` |
| `compile_transformer_inference_block()` | `derive_transformer_inference_block()` |

当 `Pass`、`PassManager`、`lowering` 与 `IR` 精确描述借用机制时继续保留。Calculon 的 `model.compile()`、Python 的 `compile()` 等外部 API 也保持原名。

保留所有既有 `compiler.*` canonical codec tag 与 `compilation-session` digest domain，把它们视为稳定 wire identity。它们是历史 opaque identifier，不是当前 Python package name。改写这些 tag 会在语义未变化时破坏 snapshot/digest，没有额外价值。

将 `TargetProfile.compiler_abi` 重命名为 `target_abi`，因为 ABI 属于绑定后的 target，而不属于 Blueprinting Compiler 组件。Decoder 将旧 field 作为 alias 接受，遇到两种拼写同时出现时拒绝 payload；encoder 只输出 `target_abi`。这个有意的 field-level schema change 会改变包含 `TargetProfile` 的 value digest；target-neutral representation digest 必须保持不变。

Report vocabulary 区分事实与预测：exact work 使用 `derived_*`，timing/memory prediction 使用 `estimated_*`，comparison 中 Blueprinting 一侧使用 `blueprinting`。Calculon 与 Vidur report 因输出 field name 改变升级为 v2。

## 影响

- `blueprinting.compiler` import 立即失败，下游 Python caller 必须原子迁移。
- 源码结构明确表达目标边界：`synthesizer` 负责 canonical derivation，`analysis` 负责 rebuildable evaluation 与 evidence resolution。
- 使用历史 `compiler.*` tag 的 canonical snapshot 仍可读取。
- 包含 `compiler_abi` 的旧 canonical JSON 仍可读取，但新序列化 target profile 与 target-bound session fingerprint 会改变。
- 不提供 pickle/module-path compatibility；canonical JSON 是受支持的 persistence boundary。
- 既有 v1 experiment report consumer 必须迁移到 v2 field name。

## 迁移

1. 将 Python import 从 `blueprinting.compiler` 替换为 `blueprinting.synthesizer`。
2. 按上表替换 public symbol。
3. 将 `compiler_abi=` constructor argument 与 attribute read 改为 `target_abi=` 和 `.target_abi`。
4. Calculon consumer 将 `compiled` 改为 `blueprinting`、`compiled_breakdown_seconds` 改为 `estimated_breakdown_seconds`、`compiled_explicit_operations` 改为 `derived_explicit_operations`。
5. Vidur consumer 将 `compiled_*` 改为对应的 `estimated_*` field。
6. 重新生成 v2 experiment artifact；不要仅为了替换 opaque codec tag 而改写旧 canonical input snapshot。

## 验证

- Package-boundary test 要求 `blueprinting.synthesizer` 存在且 `blueprinting.compiler` 不存在。
- Public API test 要求新 symbol 存在，并拒绝 legacy re-export。
- Canonical round-trip test 解码旧 `compiler_abi` payload，并验证新输出只包含 `target_abi`。
- Golden target-neutral IR snapshot 与 baseline regression digest 守护被保留的 wire tag 与 digest domain。
- Calculon/Vidur test 守护 v2 report vocabulary 与数值等价性。
- Ruff、完整 pytest、双语文档一致性与 strict MkDocs build 是 release gate。

## 状态

本 ADR 于 2026-08-09 被接受，约束随 package rename 一起交付的 hard-cut migration。未来若修改 package boundary、保留的 codec tag 或 alias policy，必须创建 superseding ADR。

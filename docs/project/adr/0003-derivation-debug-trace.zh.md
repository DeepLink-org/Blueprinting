# ADR-0003：将 IR 调试表示为派生 Derivation Trace

- 日期：2026-08-10
- 状态：Accepted
- 范围：IR visualization、lineage mapping、debug bundle 与 derived overlay

## 背景

现有 checkpoint 保存了 immutable IR、digest、pass record 和 lineage，但工作台只能展示阶段摘要与 JSON。若界面自行按名称猜测映射，或把布局/cost 写入 canonical IR，就会创建第二套表示语义并破坏 late binding。

## 决策

建立 application-owned、可失效可重建的 `DerivationTrace`。五层 adapter 按 canonical schema 注册，层内图只投影 typed fields；相邻层映射只消费 target lineage。审计规则产生非阻断诊断，transaction acceptance 仍由 verifier/observer 决定。

Cost、timing 和 observation 通过带 provider revision 的 detachable overlay 接入。`blueprinting.derivation-debug-bundle.v0` 保存一次运行的 canonical snapshots 与派生 metadata，导入时重新验证。它不是第六层 IR，也不替代 `TimelineBundle`。

当前 production run 只产生前三层；后两层 adapter 与合法导入支持不得被描述为 Concrete/Machine producer 已完成。

## 被拒绝方案

- 在五层 IR 中增加 UI 坐标、颜色、折叠状态或 cost 字段；这会污染 canonical semantic 和 digest。
- 按 display name、数组位置或 operation 名称推测跨层映射；这无法处理 decomposition、fusion 和 generated entity。
- 让可视化审计默认阻断 pass；当前规则尚未成为所有 dialect 的稳定 proof obligation。
- 首版同时实现 breakpoint、snapshot mutation 和双运行 diff；这需要独立执行状态、匹配与安全 contract。

## 影响与验证

Application report 增加 trace，分析报告 schema 升级到 v2；旧 `stages` 继续作为兼容展示。Debug bundle 有独立版本和 50 MiB 限制。五层 adapter、lineage cardinality、mismatch、deterministic round trip、tamper rejection、大图分组和 NiceGUI 交互由测试覆盖；canonical golden digest 不得改变。

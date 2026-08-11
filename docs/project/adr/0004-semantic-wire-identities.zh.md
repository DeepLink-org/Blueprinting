# ADR-0004：使用领域归属明确的语义化 Wire Identity

- 日期：2026-08-11
- 状态：Accepted
- 范围：canonical record/enum tag、digest domain 与兼容策略
- 取代：ADR-0001、ADR-0002 中保留 codec tag 的条款

## 背景

Canonical tag 仍然反映一个已经废弃的实现 package。虽然该名称已不再对应
Blueprinting 的任何领域，它仍散落在 IR 定义、workload/system contract、
analysis record、测试与生成 JSON 中。把该前缀永久视作 opaque identity，会让
源码更难阅读，并让偶然形成的历史架构持续进入每一个新产物。

## 决策

Canonical identity 直接表达语义归属：

| Namespace | 所有者 |
|---|---|
| `blueprinting.ir.*` | 公共 primitive 与五层 canonical IR |
| `blueprinting.binding.*` | 显式 derivation binding |
| `blueprinting.workload.*` | Target-neutral workload contract |
| `blueprinting.mapping.*` | Logical strategy 与 deployment mapping |
| `blueprinting.system.*` | Chip、memory、interconnect 与 system profile |
| `blueprinting.analysis.*` | Evidence 与可重建 analysis record |
| `blueprinting.synthesis.*` | Derivation session state |
| `blueprinting.expression.*` | Typed scalar expression |

名称使用 kebab-case component，不维护独立版本号。ADT constructor 仍只写短 local
tag，由 family 在语义 namespace 下展开。Compatibility 由所属 IR root schema 管理，
不在每个嵌套类型名称中重复维护第二套版本。

这是一次有意的 wire-format 硬切。Runtime codec 不注册废弃 tag alias，也不保留
field alias。已有持久化产物必须从权威 workload、mapping、system 与 evidence
输入重新生成。Canonical digest 会按设计发生变化；数值事实与推导语义不变。

Schema migration 机制保留给未来完成 graduation 的 root-schema 变更。当前 production
registry 为空，因为 pre-graduation 中间状态不构成已发布的 compatibility history。

## 影响

- IR 定义无需历史背景就能说明领域归属。
- 新编码值只包含语义化的 `blueprinting.*` identity。
- 旧序列化快照会 fail closed，不会通过 alias 静默进入当前推导。
- Golden digest 与生成的实验产物必须在同一变更中重新生成。
- 外部 consumer 必须把本次发布视作 wire-format 边界。

## 验证

- Source gate 拒绝 tag 不以 `blueprinting.` 开头的 canonical decorator，并阻止
  废弃前缀重新进入源码。
- Canonical round-trip、schema migration、determinism 与 baseline test 使用重新
  生成的语义 namespace 产物。
- Ruff、mypy、完整 pytest、双语文档与 wheel contract 继续作为 release gate。

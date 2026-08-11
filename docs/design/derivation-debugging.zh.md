# IR 推导可视化与回放审计

IR Explorer 把一次已经完成验证的形式化推导转换为可交互、可导出且可重新构建的调试视图。它消费 canonical snapshot、pass contract/record 和 typed lineage；不会修改 representation，也不会把界面布局、预测时间或 cost 写回 IR。

## 五层统一视图

工作台按 canonical 顺序提供五个 stage slot：

```text
ModelIR -> DistributedTaskIR -> PortablePlanIR -> ConcretePlanIR -> MachineIR
```

当前 training production path 产生前三层。`ConcretePlanIR` 与 `MachineIR` 没有 production producer，因此正常运行时显示明确的未生成状态；合法的五层调试包可以验证后导入同一浏览框架。这个 consumer 能力不表示 target binding、scheduler 或 emitter 已实现。

每层 adapter 只投影本层已经拥有的结构：Model 的 operation/value 和 SSA dataflow；Distributed 的 task/value、mesh/rank、sharding 与 collective；Portable 的 task/buffer、精确 workload、resource requirement 与 dependency；Concrete 的 command、device/queue、buffer allocation 与 synchronization；Machine 的 section、instruction、entry point 与 dependency。

每层同时提供两种只读文本表达。Short form 是面向推导阅读的 typed pseudo-syntax：保留 schema、interface、关键 binding 和本层核心实体，但压缩重复 task。Detailed form 枚举本层 canonical typed 字段：stable ID、dependency、workload facts、resource/implementation requirement、placement、ABI 和 lineage 等。它们都由 immutable IR 重建，不是新的 wire format，也不能替代 canonical JSON/digest。

单层视图默认先解释该层回答的问题、当前 snapshot 结果和不拥有的语义。结构图不使用无语义的 force layout，而是采用稳定的二维语义矩阵：列从左到右对应 derivation phase，行固定对应 `subsystem × entity kind`，缺失组合保留为空白。distributed/portable Transformer 因此能在同一水平泳道上比较不同阶段的 attention、MLP、compute 与 collective；结构依赖默认淡化，仅在悬停时强调邻接关系。用户可以通过 semantic group 或搜索展开局部；单次渲染最多检查 1000 个实体。

相邻 boundary 默认使用 `Source | Pass | Target` 三列 lowering 对应表。Source 与 Pass 按规则实例纵向合并，Target 逐行显示展开后的语义组。每个完整表达式由一个随文字换行的外层圆角 `span` 承载，内部 name、参数和括号使用连续的嵌套 `span` 分块；结构、类型、拓扑、精确 workload 与映射参数使用不同语义色，例如 `[MLP.forward([ranks=2,][ops=4304896])]`。内外 span 都使用 inline 换行和 `box-decoration-break: clone`，因此每个视觉行的高亮仅跟随文字，不会生成覆盖整个单元格的大边框。Entity type 决定 name 和外层基色，但不显示为独立的 head。

Short expression 不直接展示内容寻址 ID 的长 hash。对于没有语义名称的 `value:*`、`node:*` 和 `buffer:*`，派生视图按照 snapshot 中的 canonical 顺序分配 `value#1`、`node#1`、`buffer#1` 形式的本地别名。别名不进入 canonical IR、lineage 或 digest；完整 stable ID 仍用于搜索、entity inspector、mapping audit 和调试包。

Source/Target 只将该层拥有的主要参数编码进表达：Model 的 shape/dtype、Distributed 的 logical ranks/sharding/collective、Portable 的 exact operations/bytes、Concrete 的 placement/queue/implementation，以及 Machine 的 opcode/operands。中间 Pass 表达用 `<pass>.<rule>(relation=..., cardinality=N → M)`描述变换，typed signature 和 rewrite 摘要作为辅助信息。规则身份和 signature 来自 typed pass contract，运行基数来自 canonical lineage evidence。原始 canonical ID 映射和完整 pass contract 收纳在可展开审计区。所有聚合只改变 presentation，不创建 canonical entity。

## 相邻层映射

跨层关系只有在 commit gate 的 `TransitionVerifier` 解析目标实体的 `Lineage.sources`、根据 pass rule 检查 `Lineage.kind`/`Lineage.transform` 并执行规则 predicate 后才会被 `PassManager` 接受。`source_value`、`source_buffer` 和 `source_command` 用于交叉检查，不替代 lineage。Display name、容器位置和生成顺序都不是映射依据。

每个相邻 boundary 同样提供 Short / Detailed lowering 表达。Short form 首先显示 pass 的 input/output schema、required binding/analysis、produced/preserved analysis、mutation model、verification、determinism/seed 和 rewrite rule。Detailed form 进一步显示 source/target digest、transaction commit gate、analysis invalidation，以及每条 transform 的 `signature / rewrite / preserves / introduces / forbids` 语义和 lineage evidence。这些规则由 synthesizer 的 typed `PassRule` contract 声明并随调试包传递，不由前端根据名字猜测；没有声明的 transform 必须显示为 lineage-only evidence。对应表只是规则的分组证据，不定义 lowering 语义。

每个相邻 boundary 还提供关系表和以下统计：

- source/target coverage；
- 1:1、1:N、N:1 与 N:M 基数；
- generated、missing、dangling 与 explicit-source mismatch；
- pass schema、binding/analysis requirement、verification policy 与 host-side duration。

通用检查之外，当前规则会执行具名 claim，覆盖绑定后的 tensor type、role、精确 local buffer size、producer/consumer、operation identity、logical rank、精确 workload fact、dependency topology、buffer use 和 selected implementation。每条通过的 relation 都携带实际执行过的 claim evidence；仅发现 lineage relation 不计为 verified。

已声明规则的失败是阻断式 transaction diagnostic：analysis product、observer 和 checkpoint 都不会发布。`TransitionReport` 区分 `structural_only`、`canonical_conformant` 与 `relation_verified`，并将 canonical normalizer conformance 与逐 relation evidence 分开保存。IR Explorer 会同时展示两者，不把 reconstruction equality 表述成 semantic proof。所有跨层 pass 都必须注册独立 executable relation invariant 与完整 normalizer；未改变内容的同层 pass 可以是 `structural_only`，任何发生变化但没有完整 executable evidence 的 transition 都会在 commit 前失败。系统不存在成功的 unverified transition。

## Derived overlay

Canonical topology 始终是底图。Portable task cost 通过默认关闭的 overlay 显示，并携带 provider 与 revision。未来 `TimingProjection`、simulation trace 或 observation 可以实现同一 overlay protocol，但必须以 IR digest 和 stable entity ID 寻址。Overlay 不能增加 dependency、改变 workload fact 或影响 canonical digest。

## Trace 与调试包

`DerivationTrace` 是 application 层派生视图。它保存 stage instance、branch、canonical snapshot、pass metadata、boundary mapping 和可选 overlay。训练产生一条 chain；推理可以在共享 ModelIR 下分别记录 prefill 与 representative final-decode branch。

`blueprinting.derivation-debug-bundle.v0` 是确定性 JSON 调试包。导入会重新执行 canonical decode、digest、verifier、stage/schema、父 digest 和相邻 boundary 检查。包大小限制为 50 MiB。它用于一次运行的离线回放，不是新的 canonical representation、`TimelineBundle` 或跨运行 diff 格式。

## 当前边界

第一版不暂停或单步执行 `PassManager`，不允许在 UI 中修改 snapshot 后继续 lowering，也不比较两次运行。真正的 breakpoint、runtime profiler correlation、Concrete producer、Machine emitter 和 simulation trace 仍需独立 contract 与实现。

该设计决策见 [ADR-0003](../project/adr/0003-derivation-debug-trace.md)。

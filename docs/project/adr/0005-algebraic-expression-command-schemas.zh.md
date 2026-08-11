# ADR-0005：代数化表达式与 Command Schema

状态：已接受并实现

## 背景

`ScalarExpr(op, args)` 可以表示错误 arity，迫使每个 interpreter 重复恢复由 enum 决定的 product invariant。`ConcreteCommand(kind, queue?, implementation?, wait_tokens?, signal_tokens?)` 同样允许只对部分 command kind 有意义的组合。大型 verifier 只能在构造后拒绝这些状态，因此 annotation 并没有描述 canonical value 的合法空间。

与此同时，pass contract 以字符串列举 preservation property。即使 callback 只检查了声明的一部分，发现 lineage relation 也会被计为 verified；callback 还看不到证明 dependency topology 所需的完整 source-to-target mapping。

## 决策

- `ScalarExpr` 改为封闭 ADT，constructor 为 `Add`、`Subtract`、`Multiply`、`Divide`、`CeilDivide`、`Maximum` 与 `Minimum`；constructor field 直接编码 arity。
- `ConcreteCommand` 是包含一个 command-body ADT 的图 envelope。可执行 body 自己拥有必需的 implementation；支持 queue 的 body 自己拥有 optional queue。独立 synchronization ADT 表达 none、wait、signal 与 wait-and-signal。
- 跨层 pass 声明 typed lineage relation、每条 relation 的独立 executable invariant，以及一个纯 canonical normalizer。Transition verifier 先解析完整 lineage graph，再重新求值 normalizer 以检查 canonical implementation conformance，随后把 relation invariant 作为独立 semantic evidence 执行。
- 没有 relation 或 executable normal form 的跨层 pass 在 commit gate 失败。任何发生变化但缺少完整 executable evidence 的 transition 都会失败；只有内容未变化的同层 transition 可以以 `structural_only` 成功。成功的 `unverified` 状态不可表示。

## Schema epoch

这些 constructor 定义初始 canonical epoch，不是从未发布 prototype 迁移而来。因此五层 IR root 统一保持 `0.0.0`；嵌套 record 与 ADT identity 使用无独立版本后缀的语义名。Production migration registry 不包含历史 edge，也不保留 decoder alias 或 legacy constructor class。

## 影响

Canonical constructor 描述的合法状态空间显著缩小，closed match 可以使用 `assert_never` 获得静态穷尽检查。通用 consumer 仍可读取派生的 `kind`、`queue`、implementation 与 token property，但这些 property 是 projection，不是 serialized discriminator。

未来完成 graduation 后的 compatibility 变更必须提升所属 root schema，并注册显式 migration；不得用 component-local counter 替代这一边界。

## 验证

Positive 与 round-trip test 覆盖全部活动 derivation chain。Negative mutation test 在保持其余结构合法的情况下修改 specialized shape、semantic payload、generated buffer capacity、dependency topology 与 implementation identity。Normal-form check 拒绝 canonical construction 偏差；另一个反例会故意让 normalizer 接受非法 implementation，并确认独立 relation invariant 仍能拒绝它。Generic migration test 覆盖 deterministic chaining、ambiguity、no-op load 与 tamper detection，不声称存在 production history。

# ADR-0006：渐进式 Typed Python Contract

状态：已接受，基础实现已完成

## 背景

Python annotation 可以记录许多局部类型，但本身不能消除 enum-plus-optional 非法状态、显式表达预期失败、保证 closed match 穷尽，或阻止 unverified pass result commit。Runtime 强依赖 mypy 会让形式模型依赖开发工具；自定义 plugin 则会把 runtime semantic 重复实现到 checker-specific API 后面。

## 决策驱动因素

- 同一份 declaration source 必须同时服务 runtime verification 与可选 static analysis。
- Base installation 必须能在没有 mypy 时检查形式 contract。
- Validation、decode、resolution 与 derivation 的预期失败必须是 value，而不是隐藏异常控制流。
- 封闭 core semantic 与开放 plugin/provider semantic 必须使用不同 extension mechanism。
- 每个发生变化且 commit 的 derivation 都必须携带 executable evidence。

## 决策

- 在 `blueprinting.schema` 中引入 domain-free `Result`、`Checked` 与 immutable diagnostic。
- Canonical `verify`/`from_json`、pass-manager `run` 与 cost resolver `resolve` 使用 checked API；只保留具名 `require_*` exception adapter。
- Core ADT 使用显式 union alias 与 `seal_adt` 封闭；root 不可直接构造，late constructor 与未注册 subclass 会被拒绝；consumer 使用显式 union pattern match 与 `assert_never`，不注册无法证明函数体覆盖率的 decorator metadata。
- 使用不依赖 mypy 的 `ContractCompiler` 与 CI script 编译完整 record/enum schema shape，以及已加载 codec、ADT 与 pass declaration。
- 将 authoring API 与普通消费接口分开：schema decorator 只由 `blueprinting.schema.authoring` 暴露，pass decorator 只由 `blueprinting.synthesizer.passes.authoring` 暴露；普通 package root 不转发 decorator。
- 删除成功的 `UNVERIFIED` pass transition。跨层 contract 必须具备带独立 invariant 的 typed relation、分开报告的完整 normalizer、output verification 与稳定 revision。
- Target/provider extension point 继续通过 `Protocol`/registry 保持开放。
- 发布 `py.typed`，在独立可选 `typing` dependency group 与 CI job 中运行标准 mypy；不引入自定义 mypy plugin。

## 领域与 Schema 迁移

初始迁移用 constructor-specific ADT 与 structured product 替换 workload mode + optional phase、binding 中重复的扁平 Transformer fact、portable task kind、cost-support status record，以及 collective kind + 条件性 reduction/root field。所属 canonical root 继续处于未发布的 `0.0.0` schema epoch；不创建 production migration edge 或 decoder alias。

`PlanTask.kind` 继续作为 `PlanTask.body` 的 derived presentation projection，不是 serialized truth。Open target/provider 集合不由 core package 封闭。

## 影响

代码可以使用 structural pattern matching 与 monadic `map`/`and_then` composition，同时仍是普通 Python。Runtime check 在 base environment 工作；安装 mypy 后增加 source-level flow/exhaustiveness analysis，但不改变执行。

Contract compiler 验证 declaration，不证明 scientific truth 或任意 function body。Formula correctness 仍需要独立 executable invariant、canonical construction check、negative test、evidence provenance 与 experiment。通用 target-plugin registry 尚未实现，因此第三方 target declaration 当前不能贡献 compiled manifest。

## 验证

Runtime test 覆盖 result composition、diagnostic accumulation、deterministic manifest compilation、ADT closure/exact membership、authoring surface boundary、canonical checked verification/decoding、checked pass failure、cost support matching，以及对缺少 evidence 的 changed transition 的拒绝。CI 在 base test job 中运行 runtime compiler，在独立 optional-typing job 中运行 mypy，并执行完整 pytest/Ruff Gate、双语 strict docs 与要求 `py.typed` 的 wheel check。

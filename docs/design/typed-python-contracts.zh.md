# 渐进式 Typed Python Contract

Blueprinting 使用 Python 的 typed/algebraic 子集实现形式化建模与 verified derivation。这不是第二套 IR hierarchy，也不是试图把 Python 变成 Haskell；目标是在保留普通 Python value 与工具链的同时，让合法 constructor、显式 effect、封闭 alternative 与 checked failure path 直接出现在源码中。

实现包含两个彼此独立的 Gate，并共享同一份声明来源：

```text
Python annotation + frozen record + ADT declaration + pass contract
             ├── runtime ContractCompiler（始终可用）
             └── standard mypy analysis（可选依赖）
```

Runtime declaration 是事实源。安装 mypy 后会增加 function body、flow-sensitive 与 exhaustiveness analysis，但不会改变 runtime semantic，也不会建立第二套类型模型。Blueprinting 不依赖自定义 mypy plugin。

## 第一层：Typed Runtime Value

`@record` 派生 frozen/slotted canonical product type，并执行 annotation-driven structural check；`@enum` 注册 closed canonical enumeration，同时不暴露 codec registry。`@adt` 与 `@variant` 定义 semantic constructor family；显式 union alias 与 `seal_adt` 共同封闭该 family。`@record` 与 `@adt` 自己负责 dataclass derivation，因此若再叠加 `@dataclass` 会直接拒绝，避免悄然绕过 structural check：

```python
@adt(wire="blueprinting.binding.workload-mode")
class WorkloadMode:
    pass


@variant("training")
class TrainingWorkload(WorkloadMode):
    pass


@variant("inference")
class InferenceWorkload(WorkloadMode):
    phase: InferencePhase


WorkloadModeVariant = TrainingWorkload | InferenceWorkload
seal_adt(WorkloadMode, WorkloadModeVariant)
```

因此“不带 phase 的 `InferenceWorkload`”无法被表示。同一模式现在也用于 portable task body 和 cost-provider support result。`PlanTask.kind` 等 derived property 只是只读 projection，不是 serialized discriminator，也不是平行 canonical schema。

Open-world semantic 保持开放。Target plugin 与 evidence provider 继续使用 `Protocol` 和显式 registry，因为 core package 无法封闭它们的 constructor 集合。

## 第二层：Checked Effect 与 Contract Compilation

预期失败使用 `Result[T, E]`；Blueprinting 的常用特化是 `Checked[T] = Result[T, DiagnosticSet]`。`Ok(value, warnings)` 与 `Err(diagnostics)` 使控制流和 diagnostic accumulation 显式化。Programmer error 仍使用异常；`require_from_json`、`require_run` 与 `CostResolver.require` 等具名 adapter 是明确的异常边界。

Canonical boundary 直接采用 checked form：

```python
decoded: Checked[ModelIR] = ModelIR.from_json(payload)
verified: Checked[ModelIR] = model.verify()
derived: Checked[PipelineResult[PortablePlanIR]] = manager.run(
    pipeline,
    model,
    session=session,
)
```

`ContractCompiler` 显式导入 built-in declaration owner，把 canonical codec entry、完整 record field shape/default、enum member、sealed ADT closure 与 pass contract 编译成一份 immutable manifest/digest。它会拒绝未封闭或不完整 family、重复 pass identity 或同时加载的 revision、缺少 lineage rule/独立 relation invariant/完整 normalizer 的跨层 pass，以及可能在不验证 output 的情况下 commit 的 pass。因此 field annotation、顺序、constructor mode、default、enum member 或 ADT constructor 的改变都会改变 contract digest。

每个发生变化且成功的跨层 pass transition 都有 executable semantic evidence。Canonical construction equality 与 relation verification 分开报告：`canonical_conformant` 表示同层结果符合声明的 normalizer；`relation_verified` 表示每个已产生 relation 都通过了独立 invariant 或 preservation claim。同层 no-op 可以是 `structural_only`；发生变化但没有 law 的 transition 直接失败，不能发布 analysis、observer 或 checkpoint。系统不存在成功的 `unverified` 状态。

只安装普通项目依赖即可运行 runtime Gate：

```bash
uv sync --locked
uv run python scripts/check_type_contracts.py
```

该 Gate 检查已经加载的 declaration 与 runtime contract；它不会解析任意 Python module，也不会证明 function body 内部的 branch reachability。

## 第三层：可选 Static Analysis

可选 `typing` dependency group 安装标准 mypy：

```bash
uv sync --locked --group typing
uv run mypy src/blueprinting
```

Mypy 检查跨 module annotation、泛型 `Result` composition、union narrowing 与 closed-match exhaustiveness。Strict mode 先覆盖 schema/contract foundation 和选定 canonical module，再渐进扩大。Base test job 明确不安装 `typing` group；独立 CI job 运行 mypy。Wheel 包含 `py.typed`，因此 downstream type checker 可以读取 Blueprinting 的 inline annotation，而 mypy 不会成为 runtime dependency。

当前没有理由引入自定义 plugin。Plugin 会把 correctness 耦合到单一 checker API，并重复 runtime declaration logic。只有在 runtime contract 已经明确、且标准 annotation 仍无法表达某个经过实测的具体 invariant 时，才应重新评估。

## Derivation Kernel 与领域迁移

Transformation semantic 继续保持为 source snapshot 与显式 `SynthesisSession` 的纯函数。`@derivation` 记录精确 source/output schema、revision、required binding/analysis、rule、determinism policy 与 canonical normalizer。Pass manager 会重新求值 normalizer，在 commit 前要求完整 snapshot equality，然后执行独立声明的 relation invariant。Normal-form equality 证明 implementation conformance，不计作 relation evidence。

第一批迁移消除了三类非法状态：

| 旧表示 | Canonical algebraic 表示 |
|---|---|
| workload enum + optional inference phase | `TrainingWorkload | InferenceWorkload(phase)` |
| 扁平 TP/PP/DP strategy field | 包含 typed axis 与 schedule ADT 的 canonical `Transformer*Parallelism` product |
| portable task kind field | `ComputeTask | CollectiveTask | TransferTask | BarrierTask | HostTask` body |
| support status + 条件性字段 | `CostAvailable | CostUnavailable(missing_fields) | InvalidCostSupport` |
| collective kind + optional reduction/root | `AllReduce(reduction) | ReduceScatter(reduction) | AllGather | AllToAll | Broadcast(root)` |

现有五个 IR root 继续处于 `0.0.0` schema epoch。这些修改定义尚未发布的初始 epoch，不会虚构 migration edge 或 compatibility history。

## 编写规则

新增或修改形式化类型时：

1. 所有字段同时存在时使用 frozen typed product；不同 alternative 拥有不同数据时使用 sealed ADT。
2. Semantic tag 保持显式；nested type 不独立版本化，由所属 root schema 控制版本。
3. Closed consumer 直接对显式 union alias 做 pattern match，并以 `typing_extensions.assert_never` 封闭；runtime 不注册无法证明函数体穷尽性的 consumer metadata。
4. Validation、decode、resolution 或 derivation 的预期失败返回 `Checked`；只在具名 process/application boundary unwrap。
5. Target/provider extension point 使用 `Protocol` 或 registry 保持开放，不伪装成 closed world。
6. 运行 runtime contract compilation、相关 positive/negative/round-trip test、Ruff 与可选 static Gate。

元编程保持克制：decorator 只派生机械结构与 manifest，不包装执行、不检查 stack frame、不生成隐藏业务规则，也不替代显式 normalizer/verifier。

Decorator surface 按受众分层：普通 `blueprinting.schema` 与 `blueprinting.synthesizer.passes` 分别提供值/codec 和 transaction runner，不导出 authoring decorator；core schema 或 trusted dialect author 从 `blueprinting.schema.authoring` 导入 `record/enum/adt/variant`，pass/target extension author 从 `blueprinting.synthesizer.passes.authoring` 导入 `derivation/relation/claim`。Codec registry、manifest compiler 与 pass registry 属于内部实现。

## 当前限制

这套机制增强了 Python 表达 compiler-like invariant 的能力，但不是 proof assistant。Runtime closure check 确认精确的 registered family membership；mypy 对显式 union 与 `assert_never` 检查它实际分析到的 closed match。二者都不能证明科学公式正确。Semantic confidence 仍来自独立 executable invariant、canonical construction check、negative mutation test、lineage、evidence provenance 与可复现实验。

Runtime compiler 当前覆盖 Blueprinting 加载的 built-in declaration。未来第三方 target plugin 需要明确 registration/conformance contract 后才能贡献 compiled manifest；仓库目前不声称已经实现通用 target-plugin registry。

# Python 代数化 IR 编写约定

Blueprinting 使用 Python 的类型注解、frozen/slotted dataclass、结构化模式匹配和少量 deriving decorator 表达 canonical IR。目标不是模拟 Haskell 语法，而是让源码主要呈现 constructor、数据关系和 invariant，机械的 codec/immutability/registration 由统一基础设施推导。

## 源码布局

五层 canonical representation 使用一致目录：

```text
src/blueprinting/synthesizer/stages/
├── model/
│   ├── ir.py
│   └── passes.py
├── distributed/
│   ├── ir.py
│   └── passes.py
├── portable_plan/
│   ├── ir.py
│   └── passes.py
├── concrete_plan/
│   ├── ir.py
│   └── passes.py
└── machine/
    ├── ir.py
    └── passes.py
```

`ir.py` 是该层 canonical 类型的唯一真实定义位置。`passes.py` 直接定义产生该层 snapshot 的 public pass，不允许只转发到另一个 `lowering` 模块。Transformer-specific 纯推导代数位于 `synthesizer/dialects/transformer/*_derivation.py`。

已删除的 `blueprinting.synthesizer.ir` 与 `blueprinting.synthesizer.lowering` 路径不构成 compatibility surface。内部 adapter、example 与 test 都直接导入所属 stage，因此仓库只有一套可发现的定义与变换层级。

## Record 与 ADT

普通 canonical product type 使用 `@record`：

```python
@record("blueprinting.example.axis")
class Axis:
    name: str
    size: int
```

Decorator 推导 frozen/slotted dataclass、canonical codec 注册和 annotation 驱动的基础结构检查。跨字段 semantic invariant 仍需显式函数或 verifier，不能隐藏在 decorator 中。

封闭 sum type 将完整 semantic wire namespace 声明在 family，只在 constructor 上保留短 stable tag：

```python
@adt(wire="blueprinting.ir.distributed-task.task")
class TaskBody:
    pass


@variant("local-compute")
class LocalCompute(TaskBody):
    pass


@variant("collective")
class Collective(TaskBody):
    spec: CollectiveSpecVariant
```

`collective` 自动展开为 `blueprinting.ir.distributed-task.task.collective`。短 tag 必须显式，因为 Python class 重命名不能隐式改变 wire identity 或 canonical digest。组件 tag 不维护独立版本号；schema compatibility 由所属 IR root 统一控制。`seal_adt(TaskBody, TaskBodyVariant)` 会冻结精确的 registered constructor set：family root 不可构造、late variant 会被拒绝，runtime field check 也会拒绝未注册 subclass。`adt_manifest(TaskBody)` 返回 deterministic variant manifest，供 schema 文档、检查器和工具读取。

## Envelope 加 sum payload

图节点的 ID、输入输出、依赖、lineage 等共同结构只定义一次；互斥语义进入 ADT body：

```python
@record("blueprinting.ir.distributed-task.task-envelope")
class DistributedTask:
    id: NodeId
    body: LocalCompute | Collective | PointToPoint | Reshard | Control
    operation: OperationName
    ranks: tuple[int, ...]
    inputs: tuple[ValueId, ...]
    outputs: tuple[ValueId, ...]
    dependencies: tuple[NodeId, ...]
    lineage: Lineage
```

这消除了 `kind + Optional payload` 的非法组合。`CollectiveSpec` 采用同一原则：`AllReduce`/`ReduceScatter` 拥有必需的 reduction，`Broadcast` 拥有必需的 root，而 `AllGather`/`AllToAll` 不拥有这两个字段。Interpreter、verifier 和 derived view 对显式 union alias 使用 `match`。Deriving layer 在 runtime 强制精确 closure；安装可选 mypy 后，当 subject 是显式 closed union 且函数由 `assert_never` 封闭时，还会检查 match 穷尽性。

## 可以推导与必须显式的边界

可以推导：immutability、slots、constructor、equality/hash、codec registration、ADT membership、基础字段类型、match args 和 variant manifest。

必须显式：wire family identity、constructor local tag、schema migration、lineage、pass/rule identity、cross-field invariant、preservation law、target legality 和 evidence validity。

五层 IR 当前都属于初始 `0.0.0` schema epoch。默认 migration registry 有意保持为空：未发布过的中间表示不构成 compatibility history。Migration 机制仍然保留，等第一个完成 graduation 的 schema boundary 再注册真实迁移。

## Pass、relation 与 canonical construction

Pass generic 已经声明输入输出类型。普通跨层推导需要声明 entity relation、每条 relation 的独立 invariant，以及一个纯 `normalize(source, session) -> target` 函数：

```python
def verify_decompose(
    source: ModelOperation,
    target: DistributedTask,
    context: RelationCheckContext,
) -> None:
    if source.id not in target.lineage.sources:
        raise ValueError("task does not retain its semantic source")
    if target.ranks != tuple(range(context.target_ir.mesh.size)):
        raise ValueError("task does not cover the derived logical mesh")


decompose = relation(
    "transformer-decompose",
    "Expand one semantic operation into distributed tasks",
    source=ModelOperation,
    target=DistributedTask,
    verifier=verify_decompose,
    introduces=("logical ranks", "collective tasks"),
)


def normalize(source: ModelIR, session: SynthesisSession) -> DistributedTaskIR:
    return DistributedTaskIR(...)


@derivation(
    "transformer-distribute",
    revision="1",
    bindings=(BindingAxis.WORKLOAD, BindingAxis.STRATEGY),
    rules=(decompose,),
    normalizer=normalize,
)
class DistributeTransformerTrainingPass(
    DerivationPass[ModelIR, DistributedTaskIR]
):
    ...
```

Commit gate 先解析完整 lineage graph，再重新执行 normalizer，并要求结果与待提交 snapshot 完全相等。这属于 canonical implementation conformance：它能在完整 schema shape 上发现遗漏与偏差，并与 relation evidence 分开记录。随后每条 relation verifier 独立检查 workload conservation、合法 role mapping、dependency correspondence 或 target ABI compatibility 等 semantic invariant。Normalizer 不能充当自身的语义证明。`relation(..., preserves=(claim(...),))` 把 relation 与具名子 obligation 都声明为普通 immutable value，不再提供平行的 `@rule` decorator 语法。

`@derivation` 从 generic base 推导 `ModelIR -> DistributedTaskIR` 及精确 schema version。Decorator 只生成静态 contract，不包装 `run()`、不读取调用栈，也不改变执行语义。

Pass 对封闭 ADT 和 typed strategy 使用 `match`。开放式 target plugin/obligation interpreter 才使用 registry 或 `singledispatch`；canonical IR 不用 runtime monkey patch 扩展 constructor。

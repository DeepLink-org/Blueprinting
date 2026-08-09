# 形式化表示体系

当前五层 canonical representation 是有明确 ownership 的 refinement boundary，不是五个任意的 Python model。具体类型使用 `*IR` 后缀，因为 Blueprinting 借用了编译工程中的 typed intermediate representation 技术。新增或合并稳定边界需要 ADR 与 migration evidence；当前层数不是不可修订的自然定律。

![五层 canonical IR contract](../../assets/architecture/ir-contract-stack.svg)

## 表示分类

### Canonical 形式化表示

Canonical representation 拥有某一推导阶段的权威语义。它具有 deterministic serialization、stable identity/lineage、versioned schema 和 verifier。当前代码与表格仍把这些表示称为 IR。

### Derived View

Derived view 由 canonical IR、binding、evidence 和 policy 可复现计算。它可以失效和重建，但不能修改 source。`CostedTaskView`、`TimingProjection` 和 `SimulationTraceIR` 都是 derived view。

### Artifact

Artifact 封装 search、reporting、interchange 或 execution output。`PlanSet`、`TimelineBundle`、`EvaluationReport`、`ObservationSet` 和 `ProgramArtifact` 都是 artifact；可序列化不意味着它们是 canonical IR dialect。

## 通用 Envelope

每份 canonical snapshot 包含：

```text
schema_name
schema_version
producer_version
feature_set
content_digest
payload
```

Entity 使用稳定 typed ID。Decomposition 记录 one-to-many lineage；fusion 记录 many-to-one lineage。Snapshot 必须 immutable 或 transactionally isolated。Type、effect、dependency、memory semantic 和影响兼容性的 extension 使用 typed field，而不是 free-form dictionary。

## 所有权摘要

| 层 | 拥有 | 不得拥有 |
|---|---|---|
| `ModelIR` | Value、operation、type、dataflow、effect、model state | Distribution、physical resource、target cost |
| `DistributedTaskIR` | Logical mesh、shard、collective、rank dependency | Physical device、route、target implementation |
| `PortablePlanIR` | Task DAG、exact work、abstract resource/buffer、strategy choice | Kernel ID、queue、address、predicted time |
| `ConcretePlanIR` | Common coordination core + typed target schedule extension、implementation、placement、ordering、sync、buffer plan | 把 predicted time 当 correctness、machine encoding |
| `MachineIR` | Target command/instruction、section、entry point、ABI | 重新解释 portable semantic |

## 信息所有权矩阵

| 信息 | 首次拥有者 | 下游规则 |
|---|---|---|
| Model operation、value、effect | `ModelIR` | 通过 lineage 引用，不重新解释 |
| Rank、shard、logical collective | `DistributedTaskIR` | Target stage 只选择实现 |
| Exact FLOPs、bytes、message | `PortablePlanIR` | Provider 不得覆盖 workload fact |
| Physical device、queue、buffer offset | `ConcretePlanIR` | Simulation 与 MachineIR 消费 |
| Target-only dataflow、route、issue/slot constraint | `ConcretePlanIR` typed extension | Target verifier、simulation 与 MachineIR 必须消费同一版本 |
| Implementation ID | `ConcretePlanIR` | MachineIR 进一步编码 |
| Target instruction 与 ABI section | `MachineIR` | Artifact 引用 digest |
| Predicted latency 与 timestamp | Derived view | 不定义 correctness |
| Actual runtime duration | `ObservationSet` | 产生新的 calibration revision |

## 通用验证规则

成熟 contract 的每层 canonical representation 都必须验证 schema identity、ID uniqueness、reference integrity、lineage validity、deterministic extension encoding 以及本层 invariant。Transformation 同时验证 input 与 output，并记录 parent digest。当前前三层有 production derivation slice；`ConcretePlanIR` 与 `MachineIR` 只有 experimental schema/structural verifier，不能据此声称完整 target legality。

## Schema 演进

Backward-compatible addition 提升 minor schema version 并由 feature 保护。Breaking change 提升 major version 或提供显式 upgrader。Serialized package path 是实现细节，不是 schema identity。内部 `1.0.0` 只标识 serialization schema；public stability 还需要 producer、独立 consumer、negative test、migration 与 cross-target conformance Gate。

## 参考页面

- [模型与分布式 IR](model-distributed.md)定义 target-neutral program 和 logical distribution semantic。
- [规划与执行 IR](planning-execution.md)定义 portable planning、target-binding gate、concrete command、MachineIR 和 derived product。

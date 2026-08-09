# Cost Provider 与性能数据导入

Blueprinting 现在已经具备一条可运行的 task-cost 接缝，位于 portable workload facts 与 plan-level composition 之间。当前实现刻意小于完整性能服务：它解析单个 operator 或 communication task 的 latency，保留 source identity，并拒绝歧义 evidence；它还不会对任意 shape 做插值、建模 contention，也不替代 schedule simulation。

## 为什么要有这条边界

`PortablePlanIR` 只拥有 work——operations、bytes、message、dependency 与 semantic shape。它不能拥有某个 runtime 或某张 GPU 上测得的 duration。Cost query 把这些 immutable work 与迟绑定的 architecture、runtime、implementation 和 deployment context 组合起来：

```text
PortablePlanIR task + late-bound CostQueryContext
  -> CostQuery
  -> CostResolver(ordered providers)
       1. PerformanceDatabaseProvider
       2. 其他 measured/simulated provider
       3. RooflineCostProvider
  -> CostEstimate + provider attempts + revisions
  -> inference cost view / future schedule simulation
```

Resolver 只选择一个 provider，不会把 correction 相乘，也不会平均无关 source。普通 miss 可以继续尝试下一个 provider；歧义或内部冲突 evidence 是错误，不能被 fallback 掩盖。

## Normalized Contract

`CostQuery` 的 first-class field 包括：

- subject（`operator` 或 `communication`）、operation、hardware 与 datatype；
- operations、read/write bytes、message bytes、participants、network tier 与 engine；
- hardware、implementation、runtime、topology 与 power-mode revision；
- `m/n/k`、batch、token、context、head 或 backend 等 operation-specific dimension。

所有字段都参与 canonical query digest。可选 identity 的空字符串表示“未知”；它不会命中一条明确要求某个 runtime 或 kernel 的 record。

`CostEstimate` 除 latency 外，还返回 method、match kind、provider/source revision、raw record ID、validity selector、uncertainty、component 与 assumption。当前 method 包括 measured、simulated、analytical、calibrated 与 vendor-model；当前 database provider 只产生 exact-selector 结果。

## Roofline Provider

`RooflineCostProvider` 包装版本化 `HardwareProfile`。对于 local operator，它计算：

```text
compute_time = operations / effective_engine_throughput
memory_time  = (read_bytes + write_bytes) / effective_memory_bandwidth
roofline     = max(compute_time, memory_time)
```

Peak-only 与 system-evidence efficiency mode 是显式选项。`processing_mode="roofline"` 使用 max bound；`"no_overlap"` 使用显式串行求和；`"profile"` 采用 legacy profile 的设置。Estimate 会暴露两个 component、arithmetic intensity 与最终 bottleneck。

Communication query 使用选定的 `NetworkProfile`：collective volume rule、participant count、bandwidth、efficiency 与 launch latency 都是可见输入。这是 analytical collective model，并不声称 cycle-accurate network simulation。

## Immutable 性能数据库

每个 `PerformanceRecord` 包含一个 latency sample、声明过的 selector 与 `EvidenceProvenance`：

```text
source + source revision
importer revision + source-file digest
method + raw record ID
selector + optional source metadata
```

`PerformanceDatabase` 支持 canonical serialization，并且 content-addressed。`PerformanceDatabaseProvider` 先寻找 selector 对 query context 做精确子集匹配的 record，再选择 specificity 最高的一组。同一 provenance group 的重复 sample 用 median 聚合，同时报告 population deviation 与 range。

Provider 构造时会按 core identity、selector schema 与 typed selector value 建立 immutable index，因此 query resolution 不需要线性扫描完整 imported corpus；database digest 只计算一次，并缓存为 provider identity。

如果两个不同 provenance group 具有相同 specificity，它们就是歧义 evidence。Provider 会拒绝，而不是选择最快 row 或静默混合 revision。第一版刻意不实现 interpolation；后续应把它做成具名 provider，并显式声明 validity domain 与 extrapolation distance。

## 导入 Simulator 表

`SimulatorPerformanceImporter` 支持 CSV、JSON/JSONL，以及安装 `performance-data` extra 后的 Parquet。`TabularImportSpec` 必须声明所有 mapping 与 latency unit：

```python
from blueprinting.compiler.analysis import (
    CostSubject,
    EstimateMethod,
    LatencyUnit,
    SimulatorPerformanceImporter,
    TabularImportSpec,
)
from blueprinting.compiler.frozen import FrozenDict

spec = TabularImportSpec(
    name="noc-sim-r7",
    subject=CostSubject.COMMUNICATION,
    source="our-network-simulator",
    source_revision="git:4e5c...",
    method=EstimateMethod.SIMULATED,
    latency_column="latency_us",
    latency_unit=LatencyUnit.MICROSECONDS,
    operation_column="collective",
    hardware="lpu-candidate-17",
    datatype_column="dtype",
    selector_columns=FrozenDict({
        "message_bytes": "bytes",
        "participants": "ranks",
    }),
    selector_types=FrozenDict({
        "message_bytes": "int",
        "participants": "int",
    }),
    record_id_column="run_id",
)
database = SimulatorPerformanceImporter.from_file("collectives.csv", spec)
```

Importer 不会猜测 unit，也不会启发式转换 selector column。无效、空、非有限值或类型错误的输入都会使 ingestion 失败。

## 导入 Vidur Profile

`VidurProfileImporter.from_csv()` 理解 Vidur attention 与 MLP profile schema，并把支持的 timing column 转换为 normalized record。Attention record 保留 phase、batch、context/cache semantic、TP degree、backend、block size 与模型 shape；compute record 保留 token count 和 model/parallel dimension。

这**不会**改变 oracle boundary：

- `VidurProfileBaseline.lookup()` 仍是 post-hoc comparison oracle，不是 `CostProvider`；
- `VidurProfileImporter` 是显式 user action，会创建一份新的 Blueprinting-owned evidence database；
- 只有被用户明确安装到 resolver 中的 `PerformanceDatabaseProvider` 才能影响 costing。

这样既能防止 validation baseline 泄漏到 lowering，又允许经过独立审查的 profile data 成为 admissible evidence。

## 导入 NVIDIA AIConfigurator 数据

官方 [AIConfigurator 仓库](https://github.com/ai-dynamo/aiconfigurator)把 component performance 存成按 operation family 区分的异构 Parquet 表。`AIConfiguratorPerformanceImporter` 当前为以下四类表提供 strict adapter：

| Upstream table | Normalized subject/operation | 必需 shape context |
|---|---|---|
| `gemm_perf` | operator / `gemm` | `m`、`n`、`k`、dtype |
| `context_attention_perf` | operator / `attention_core` prefill | batch、input length、local heads、head size、KV dtype |
| `generation_attention_perf` | operator / `attention_core` decode | batch、`isl + step`、local heads、head size、KV dtype |
| `custom_allreduce_perf` | communication / `all_reduce` | message bytes、GPU count、backend |

AIConfigurator latency 从毫秒规范化为秒。Framework、framework version 与 kernel source 会成为必须匹配的 runtime/implementation selector；device 与 upstream operation identity 留在 provenance metadata。因此，runtime/kernel 未知的 hardware-only query 不会命中 AIConfigurator row。

```python
database = AIConfiguratorPerformanceImporter.from_file(
    "gemm_perf.parquet",
    hardware_name="h100-sxm",
    source_revision="8fc57cf...",
)
```

AIConfigurator 的最终 `best_config_topn.csv` 与 Pareto output 描述的是 serving configuration，不是 atomic operator evidence，所以不会被导入成 task cost。未来可以增加 plan-level comparison adapter，但不能把 TTFT/TPOT 压扁成 kernel latency。

## 在 Inference Costing 中使用 Resolver

Static inference 已可使用新 resolver，同时保留 legacy `InferenceCostProvider` seam：

```python
resolver = CostResolver((
    PerformanceDatabaseProvider(database),
    RooflineCostProvider(hardware),
))

context = CostQueryContext(
    runtime="vllm",
    runtime_revision="0.24.0",
    implementations=FrozenDict({
        "attention_pre_projection": "torch.nn.functional.linear",
        "all_reduce": "vLLM_custom_graph",
    }),
    operation_dimensions=FrozenDict({
        "all_reduce": FrozenDict({"backend": "vllm_graph"}),
    }),
)

estimate = estimate_inference_phase(
    plan,
    hardware,
    cost_resolver=resolver,
    cost_context=context,
)
```

Inference task query 直接从 canonical `PlanTask.workload` 推导；GEMM dimension 与 local attention-head dimension 来自 model 和 TP facts。Tensor-parallel collective 与 pipeline P2P 使用同一个 resolver。所有 task 都必须被已安装的 provider 覆盖——通常是 exact database 后接 roofline——unknown task 不会被静默变成零。

## 已实现边界与后续工作

当前已经实现：

- normalized immutable query/estimate/support/resolution contract；
- 带 auditable attempt trace 的 deterministic ordered resolution；
- analytical roofline 与 collective fallback；
- immutable exact-selector database 与 repeated-sample aggregation；
- 通用 simulator/profiler table ingestion；
- 显式 Vidur 与四类 AIConfigurator ingestion；
- inference task/pipeline 接入与回归测试。

仍未实现：

- calibrated interpolation/extrapolation provider 与 confidence policy；
- portable snapshot 之外的 append-only raw-sample/environment-manifest store；
- 通过 resolver equivalence test 迁移 training path；
- target binding 中 first-class KV-head/GQA 与 runtime legalization context；
- contention、overlap、queueing 与 plan-level discrete-event simulation；
- 作为 normalized planner objective 的 energy/power metric；
- observation ingestion 与 calibration revision。

这些边界非常重要：task latency resolution 是 evidence layer，不是已经完成的 hardware simulator 或 serving simulator。

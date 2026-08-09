# 性能数据库

性能数据库是 normalized query protocol 背后的版本化 evidence store。它回答一个精确问题——某个 architecture component 或合法 implementation 在明确 context 中预计如何表现——但不会把 architecture choice 或 calibration knob 隐藏在 lookup table 中。

!!! note "设计状态"
    当前仓库仍直接加载 `HardwareProfile`。本页定义的 request/result/store 是已接受的迁移目标。

## Request Contract

`EstimateRequest` 标识所有可能实质影响结果的维度：

```text
subject identity
  portable task + legal implementation revision
workload context
  operation, shape, dtype, layout, operations, bytes, message volume
architecture context
  blueprint, component/engine, memory space, capability and implementation revisions
deployment context
  device class, topology/link class, participant count, placement, runtime/library revision
execution context
  concurrency class, queue/resource occupancy, power mode
evidence policy
  admissible providers, freshness, uncertainty and fallback limits
```

Optional field 是显式 unknown，而不是从 cache key 中省略的维度。Provider 要声明自己需要哪些字段，以及答案在哪个 domain 内有效。

## Result Contract

`EstimateResult` 不只是一个标量：

```text
metrics             latency, energy, bandwidth, utilization, counters
uncertainty         interval/distribution/confidence and sample count
validity            exact domain, interpolation, extrapolation distance
provenance          provider, raw record IDs, source and calibration revisions
method              measured, simulated, analytical, calibrated, fallback
diagnostics         missing context, assumptions, rejected alternatives
```

Normalization 不会抹掉 provider detail。Provider-specific payload 可以作为 typed extension 附加，而 planner 与 audit 所需的字段保持 portable。

## 存储模型

数据库把 immutable raw evidence 与 derived index/calibrated model 分开：

| Collection | 内容 | 更新规则 |
|---|---|---|
| Raw records | benchmark sample、simulator run、profiler event | append-only |
| Environment manifests | device、driver、firmware、runtime、clock、topology、protocol | content-addressed |
| Aggregates | cleaned sample、distribution、confidence interval | 可重建 |
| Calibration revisions | 拟合 curve/model 与 training-set digest | 新的 immutable revision |
| Query indexes | normalized request dimension 的检索加速 | 可重建 |

每个 raw sample 都应尽可能记录 unit、warm-up、repetition count、synchronization method、clock/power state 与 measurement error。单独一个 `(op_name, latency)` 不能构成充分 evidence。

## Provider Protocol

Provider 在概念上暴露四个操作：

```python
class EstimateProvider(Protocol):
    @property
    def revision(self) -> str: ...

    def supports(self, request: EstimateRequest) -> Support: ...
    def estimate(self, request: EstimateRequest) -> EstimateResult: ...
    def explain(self, result: EstimateResult) -> EvidenceTrace: ...
```

`supports()` 在昂贵求值前报告 domain coverage 与缺失的 required field。对于同一 request/provider revision，`estimate()` 必须确定；若使用随机协议，则结果要显式记录 seed 与 stochastic protocol。

## Resolution Policy

`CostResolver` 负责选择 evidence，而不是堆叠 correction factor。一种典型 policy 可以依次偏好：

1. compatible direct measurement；
2. 已在目标 revision 验证的 compatible simulator result；
3. measured domain 内的 calibrated interpolation；
4. analytical estimation；
5. 显式 conservative fallback。

冲突 source 仍然可检查。Resolver 记录某个 source 胜出的原因和其他 source 被拒绝的原因。只有具名、版本化且声明输入的 model 才能 blend evidence，不能使用匿名 coefficient product。

## Cache Identity 与可复现性

Result cache key 包含 canonical request digest、provider revision、calibration revision、resolver-policy revision，以及相关时的 deterministic seed。因此 target、deployment、runtime、topology 和 concurrency context 都是 identity 的一部分，而不是 ambient state。

发布的 plan 携带 evidence snapshot fingerprint。在该 snapshot 下重建时，系统要么复现相同 result，要么报告所需 evidence 已不可用；绝不能静默升级到数据库最新状态。

## 不依赖 Case Fitting 的校准

Calibration 从 observation 学习 target-wide 或 implementation-family response behavior，例如 operation-size efficiency、transfer-size efficiency、network efficiency、launch overhead 或 contention model。Training data 与 validation split 都要版本化。

禁止输入 benchmark case ID、comparison-oracle total time，或唯一作用是匹配某张表的 per-model correction factor。这些变量不能解释因果 target behavior，也无法泛化到新 plan。

## 从 HardwareProfile 迁移

现有 `HardwareProfile` 已经提供 matrix/vector throughput、memory transfer 与 collective 的有用版本化 curve。迁移应通过 provider 保持现有行为：

1. 把 portable task 转换为 normalized request；
2. 将当前 profile 包装为 analytical/system-evidence provider；
3. 通过 resolver 复现当前 Calculon experiment；
4. 增加 raw measurement provider 与 evidence manifest；
5. 只有 equivalence test 通过后，才替换 estimator/profile 的直接耦合。

这种分阶段 adapter 能保留已验证的 workload analysis，同时让 provenance、uncertainty 与未来 hardware simulator 成为 first-class capability。

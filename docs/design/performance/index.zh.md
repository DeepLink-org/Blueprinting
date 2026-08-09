# 性能证据与 Cost Model

Performance evidence 把 hardware blueprint 变成可证伪 prediction。Blueprinting 把 evidence 与 workload/architecture semantic 分开：exact work 是 mapped workload 的属性；latency、energy、utilization、area 与 cost 则是关于 mapping、architecture、deployment、method、context 与 revision 的陈述。

![Evidence resolution、simulation、observation 与 calibration 闭环](../../assets/architecture/evidence-simulation-loop.svg)

## 四类事实

系统严格区分四个类别：

| 类别 | 示例 | 所有者 |
|---|---|---|
| Workload facts | operations、read/write bytes、message bytes、dependency | canonical IR |
| Architecture facts | engine hierarchy、支持的数据类型、memory capacity、topology | architecture blueprint 与 deployment profile |
| Estimates | latency、energy、utilization、uncertainty | evidence provider 与 cost view |
| Observations | 实测 event interval、counter、memory use、communication | immutable observation set |

Observation 可以校准未来的 estimate revision，但不能改写历史 evidence 或 workload facts。Estimate 可以指导 scheduling，但不能向 command DAG 添加 correctness edge。

## Evidence 流程

标准流程是：

```text
canonical task + candidate architecture + legal implementation + execution context
  -> EstimateRequest
  -> CostResolver(policy, evidence snapshot)
  -> one or more EstimateProviders
  -> normalized EstimateResult
  -> CostedTaskView
  -> scheduling / timing / simulation
  -> ObservationSet
  -> CalibrationRevision
  -> new immutable evidence snapshot
```

每条边都是强类型且 content-addressed。Resolver 会记录实际响应的 provider、raw source revision、calibration revision、validity domain、fallback status 与 uncertainty。

## Provider 类型

Provider 可以来自：

- 实测 kernel/collective database；
- analytical roofline 或通信模型；
- hardware simulator 与 vendor performance model；
- network simulator 或 topology-aware collective model；
- 从 immutable observation 构建的 calibrated surrogate model。

这些 provider 实现统一的 normalized protocol。Blueprinting 不会把 simulator-specific schema 导入 canonical workload/architecture object，也不会把所有可用答案相加。Policy 选择最合适且被允许的 source；如果用户要求比较，则保留其他 alternative。

## 两级评估

Task-level estimation 在明确 context 中预测一个合法 implementation：shape、dtype、resource、placement class、concurrency state 与 environment。Plan-level evaluation 则通过 concrete dependency graph 和 resource model 组合 task。

这一区分避免了一个常见错误：把互相独立的 task latency 直接当成端到端 schedule。Overlap、serialization、contention、bubble 与 buffer pressure 是 plan/deployment 的属性，不是某条 kernel record 的标量属性。

## 不确定性与有效域

每个 estimate 都要声明 validity domain 和 uncertainty representation。至少需要记录结果属于 measured、simulated、analytical、calibrated、extrapolated 还是 fallback。Extrapolation distance 与缺失 context 必须对 Pareto/risk policy 可见。

Planner 可以优化 expected latency、conservative bound 或 risk-adjusted objective，但不能静默把 unknown cost 变成零，也不能把低置信 extrapolation 当成精确实测值。

## 分层 Profiler 关联

稳定 lineage 允许把 observation 聚合到多个语义层：

| 关联层 | 典型问题 |
|---|---|
| Model operation | 高层 decomposition 是否完整？ |
| Distributed task | Sharding 或 collective volume 是否正确？ |
| Portable task | 精确 work 与 capability requirement 是否正确？ |
| Concrete command | Queueing、contention 或 overlap 是否符合计划？ |
| Machine instruction | 误差是否来自 implementation 或 ABI behavior？ |

这并不表示每一层都要拥有 wall-clock duration。早期层通过 semantic/conservation facts 检查；concrete stage 才能额外使用 event time 检查。

## 当前实现边界

仓库当前提供强类型 `HardwareProfile`、peak-only 与 system-evidence efficiency curve、block/iteration estimate，以及可审计的 Calculon experiment。这些组成一条已实现 validation adapter，但尚不是通用 architecture-exploration evidence service。

Normalized request/result protocol、provider registry、evidence store、uncertainty model、discrete-event simulator、observation ingestion 与 calibration service 仍属于目标架构。它们应先包裹、再替代对 `HardwareProfile` 的直接耦合，而不改变 `PortablePlanIR`。

## 设计不变量

1. Canonical IR 永远不拥有 measured/predicted duration。
2. Raw evidence 与 observation 不可变；calibration 创建新 revision。
3. 每个 result 都能归因到 request、provider、revision 与 policy。
4. Workload derivation 不能读取 comparison oracle 的结果。
5. Plan-level overlap 从 dependency/resource 推导，不能使用 universal ratio。
6. Simulation 与 runtime 使用同一 concrete command identity。
7. 只有完整 semantic context 一致时 cache hit 才合法。

[性能数据库](database.md)定义 evidence storage 与 resolution；[仿真与校准](simulation.md)定义 plan-level composition 和反馈闭环。

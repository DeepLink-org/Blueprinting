# 仿真与校准

Simulation 是 candidate hardware blueprint 变成 executable hypothesis 的阶段。它在明确 cost/contention model 下评估 verified architecture-bound resource plan，而不是发明第二套 schedule。Calibration 比较 predicted/observed event 并创建新 evidence revision，但不会修改产生这些 event 的 blueprint 或 plan。

!!! warning "设计状态"
    Block/iteration analytical estimate 已实现；下面描述的 discrete-event simulator、observation normalizer 与 closed-loop calibration service 均为 **Planned**。

## Simulation 输入

Simulator 消费：

- 已验证的 `ConcretePlanIR` envelope（common coordination core + typed target extension）；
- deployment 中的 physical resource、queue、memory space 与 topology；
- 为已选 implementation 完整覆盖的 `CostedTaskView`；
- simulator model 与 evidence revision identity；
- 对显式 stochastic resource 可选的 deterministic seed。

它不会消费 `ModelIR` 并自行重建 placement 或 overlap。Portable/model ID 只通过 lineage 用于聚合和诊断。

## Discrete-Event 状态

最小状态包括：

```text
logical clock
ready/running/completed command sets
per-queue order and availability
per-engine and per-link capacity
signal/event/barrier state
buffer allocation and lifetime state
resource contention state
ordered event queue
```

只有显式 dependency、ordering predecessor、synchronization condition 和 required resource 全部满足时，command 才进入 ready。预测 start timestamp 永远不能使 command ready；target-enforced cycle/slot 则由 typed target extension 的规则解释。

## Event 算法

一个 deterministic baseline algorithm 是：

1. 将满足 semantic readiness condition 的 command 入队；
2. 按稳定顺序选择 ready command 并预留 resource；
3. 从选定 cost view 获取 duration/resource demand；
4. 安排 completion 与 intermediate transfer event；
5. 推进到下一个 event，释放 resource，更新 signal 与 buffer；
6. 重复，直到全部 command 完成或到达可诊断 deadlock。

每个 event 都携带 concrete command ID、resource ID、estimate provenance 与 upstream lineage。Result 包含 interval、critical path、resource utilization、memory high-water mark、queue delay 与 uncertainty propagation。

## 网络仿真

Communication 从 logical collective semantic、concrete participant、route、algorithm 与 link resource 建模。简单 provider 可以使用 latency-bandwidth curve 和 collective volume model；详细 network simulator 可以模拟 topology、routing、arbitration、congestion 与 failure。

两条路径都为同一 request 返回 normalized result。Network time 不能在上游直接写成 `message_bytes / nominal_bandwidth`，也不能使用 global overlap ratio 近似 command concurrency。

## Timing Projection 与 Simulation Trace

`TimingProjection` 是 concrete command 上的可重建 analysis：predicted interval、slack、critical path 与 uncertainty。`SimulationTraceIR` 是包含 resource event 和 source correlation 的 interchange trace。两者都不是 canonical execution program。

`TimelineBundle` 引用 concrete plan、projection/trace、evidence/policy fingerprint 和 diagnostic，作为可发布的分析/replay 产品包。它不能复制并修改 command DAG。

Replay/runtime engine 遵循 `ConcretePlanIR` 或其 `MachineIR` lowering 中的 dependency、ordering 与 synchronization。Simulator 可以附加 predicted timestamp；runtime profiler 则附加 observed timestamp。两者关联到同一 command identity，但 runtime 可以执行 target contract 允许的 bounded backpressure 和 failure handling。

## Observation Normalization

Profiler adapter 把 target-specific counter/event 转换为 immutable `ObservationSet`：

```text
environment manifest and run identity
artifact, MachineIR, ConcretePlanIR and evidence fingerprints
command/instruction correlation
observed intervals and resource counters
missing, duplicated, or unmatched events
measurement protocol and uncertainty
```

Raw trace 继续被附带或引用。Normalization 绝不为 unmatched event 伪造 command ID；缺失 correlation 本身就是 diagnostic 和 quality metric。

## Calibration Revision

Calibration 把 estimate request/result 与 compatible observation 对比，按因果维度分层 residual，并产生新的 immutable model revision。例如按 operation size 学习 efficiency curve、按 runtime revision 学习 launch overhead，或按 topology/participant count 学习 collective behavior。

旧 evidence snapshot 以及所有基于它推导的 plan 仍然可以复现。采用新 revision 是显式重新分析与 replanning 选择。如果 estimate 变化导致 placement 或 scheduling 改变，结果必须具有新的 `ConcretePlanIR` digest。

## 有界 Refinement Loop

部分相互作用需要迭代：

```text
candidate implementation/placement
  -> preliminary costs
  -> schedule and contention context
  -> context-aware re-estimation
  -> reschedule if material
```

Refinement 必须有 deterministic convergence rule、iteration budget 与 oscillation diagnostic。最终 plan 记录每轮迭代和选定 evidence。无界 feedback loop 不是合法 analysis 或 transformation transaction。

## 对应关系义务

只有满足以下条件时，simulation 与 execution 才能声明达到某一级 correspondence：

1. 两者来自同一个 verified concrete command DAG；
2. 每个 emitted/simulated operation 都映射到 concrete command 或声明过的 runtime support action；
3. dependency、queue order、synchronization 与 buffer contract 被保持；
4. 从 trace 去掉 timing 后，execution semantic 仍完整；
5. observed deviation 作为 evidence 表示，而不是追溯修改 IR。
6. observation policy 声明 exact-event、bounded-divergence 或 partial-observation level。

这些义务比 total latency 匹配更强：它们检查 simulation 与 optional emission 是否实现同一个 verified architecture-bound plan；它们不证明 predicted timestamp 与 observed timestamp 相等。

## 验证阶梯

Simulation system 应按逐步提高的 fidelity 演进：

1. 使用可手算 DAG 的 deterministic virtual-target unit test；
2. 与现有 block/iteration estimator 做 analytical consistency；
3. Calculon workload 与 schedule 对比；
4. replayable command-engine 对比；
5. 真实 GPU profiler correlation；
6. LPU architecture/simulator contract 可用后做 simulation correspondence；ABI/runtime 可用后再做 executable correspondence。

每一级都先验证 semantic conservation 与 command correlation，再验证 aggregate latency accuracy。

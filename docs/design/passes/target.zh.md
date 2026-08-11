# 架构绑定与计划构造

Architecture binding 是从 portable workload mapping 到 architecture-bound simulation plan 的形式化桥梁。它针对 candidate hardware blueprint、deployment 与 evidence policy 检查已验证的 `PortablePlanIR`，构造唯一权威的 `ConcretePlanIR` envelope，再从中派生 timing、simulation 与 optional target program。

!!! warning "设计状态"
    迟绑定边界、experimental schema、queue/slot 两个 deterministic reference binder 已存在；target plugin registry、production producer 和后续 analysis/transformation chain 仍为 **Planned**。

## 为什么必须做成一条纵向切片

Legalization、implementation selection、cost resolution、placement、scheduling 和 memory planning 不能作为互不相关的 utility 分别实现。每项决策都会约束下一项，而 scheduling 也可能反过来使 allocation 或 implementation choice 失效。因此，第一个 target-complete milestone 必须是一条端到端 virtual target 路径，而不是若干断开的 half-pass。

已接受的顺序是：

```text
PortablePlanIR
  -> target legalization
  -> implementation candidates
  -> cost resolution analysis
  -> placement + target scheduling + memory planning
  -> ConcretePlanIR
  -> timing/simulation projection -> TimelineBundle
  -> optional replayable MachineIR artifact
```

## Binding Gate

这个 gate 要求三个显式输入：

- `TargetBinding`：architecture、runtime、capability、library 与 ABI identity；
- `DeploymentBinding`：device instance、topology、capacity 与 reservation；
- evidence policy/revision：允许的 provider、uncertainty budget、fallback rule 与 freshness。

输入 portable digest 保持不变；所有被选择的 identity 与 planning-evidence fingerprint 进入新的 concrete plan construction provenance，后续 evaluation evidence 进入 derived analysis identity。缺失 binding 必须产生 diagnostic，不能通过读取 process-global default 补齐。

## Legalization 与 Implementation Selection

Legalization 把每项 capability requirement 映射为合法 target implementation，并检查 dtype、shape、layout、alignment、memory space、collective、runtime、library 与 ABI constraint。Target plugin 可以给出 rewrite alternative，但不能改变精确 workload operations、bytes 或 logical collective semantic。

Implementation selection 可以为搜索保留多个 candidate。每个 candidate 记录 source portable task、target capability rule、plugin revision、required resource 与 rejection reason。不支持的工作必须通过带 source lineage 的 diagnostic 失败，不能静默选择一个 generic zero-cost implementation。

## Cost Resolution Analysis

Cost resolution 为合法 implementation candidate 构造 normalized estimate request，并在 experiment policy 下解析 evidence。它产生 `CostedTaskView`，而不是新的 semantic representation。

因此，同一个 portable/legal plan 可以使用实测 kernel、analytical roofline、hardware simulator 或 network simulator 评估，而 digest 不会改变。超出 evidence domain 的结果必须带有显式 confidence/fallback status，不能伪装为 direct measurement。

## 联合 Scheduling 与 Memory Planning

Scheduler 拥有 physical placement、implementation instance、ordering、synchronization、routing requirement、resource occupancy 与 buffer allocation。跨 target 稳定的内容进入 common coordination core；dataflow、route、issue slot 或 micro-protocol 的 correctness semantic 进入 typed target extension。Memory planning 必须和 scheduling 联合完成，因为 buffer interference 取决于可能发生的 overlap。

构造过程至少必须保证：

1. command dependency 闭合且无环；
2. queue order 与 signal/wait edge 不会形成 deadlock；
3. 每个 command 都有合法 implementation 与 placement；
4. buffer lifetime 覆盖全部 producer 和 consumer；
5. 被复用 allocation 在任何合法执行中都不重叠；
6. device、memory space、engine 与 link capacity 均满足；
7. 用于优化的 cost 已经覆盖，或由显式 risk policy 处理。

算法可以从 deterministic list scheduling 与 interval allocation 起步。后续可加入 constraint programming 或 bounded refinement，但当前 `ConcretePlanIR` contract 仍是 experimental；只有 common core 经 queue-centric 与 non-queue-centric target 验证后，才能讨论冻结兼容性。

## Timing 与 Simulation Projection

Command DAG 验证完成后，timing projection 计算预测 interval、contention、critical path 与 uncertainty。Simulator lowering 把同一组 command/resource 映射为 discrete event。

预测 timestamp 是 annotation，不是 readiness semantic。移除它们后仍必须保留可 replay 的 dependency/queue program。正因为如此，新 evidence revision 可以改变预期 duration，而不会改变执行正确性。

如果某个 target 把 cycle/slot 作为 correctness constraint，它在 binding 后进入 typed target extension，而不是 `TimingProjection`。Projection 与 trace 连同 concrete/evidence/policy digest 发布为 `TimelineBundle`；详细语义见 [Timeline 阶段路径](../timeline-path.md)。

## 已实现的 Reference Binder Pass

`stages/concrete_plan/passes.py` 直接定义两个用于验证 contract 的 deterministic pass：

- `BindReferenceQueueTargetPass`：产生 queue-centric `QueueScheduleExtension`；
- `BindReferenceSlotTargetPass`：产生 non-queue `SlotDataflowExtension`。

两者都保持 `PlanTask -> ConcreteCommand` 的 1:1 identity、dependency topology 和 buffer use，不声称是 production scheduler。Buffer 使用稳定顺序与 alignment 做线性分配：

```text
offset_0 = 0
bound_i = align_up(offset_i, alignment_i)
offset_(i+1) = bound_i + size_i
```

每个 command 的 implementation 与 placement 都来自显式 target/deployment binding；commit gate 使用同一个纯 normalizer 重建完整 concrete plan，并要求 source buffer identity、exact size、task lineage、command mapping 与 typed extension 全部相等。独立 relation invariant 另外检查 portable-to-concrete buffer identity/capacity/alignment、dependency correspondence、buffer access、operation identity 与 target ABI。两个 pass 使用相同 portable input 产生不同 typed extension，用于证明 common envelope 不依赖 queue-only 假设。

这是 contract reference implementation，不引用性能论文，也不以 heuristic quality 为设计声明。源码在 `src/blueprinting/synthesizer/stages/concrete_plan/passes.py`，positive/negative、lineage 与 deterministic replay 测试在 `tests/synthesizer/test_reference_targets.py`。

## Machine Lowering 与 Artifact Emission

Target plugin 把 verified concrete command lowering 到自己的 `MachineIR` dialect。Virtual target 应先输出 deterministic replay package；CUDA、LPU 与其他硬件 plugin 再逐步增加 ABI-specific instruction 和 executable artifact。

每个 artifact manifest 都记录 target/deployment identity、portable/concrete/machine digest、analysis-engine/plugin revision、runtime requirement、evidence fingerprint 与可分离 lineage map。Emission 不得引入 concrete plan 中不存在的高层 scheduling decision。

## Checkpoint 与诊断

Observer 分别挂接在 legalization、concrete construction、timing、simulation 与 machine verification 边界，检查 capability coverage、command correlation、cost coverage、resource conflict、trace conservation 与 ABI legality。

Diagnostic 必须保留完整 candidate/source path：

```text
model operation -> distributed task -> portable task
  -> implementation candidate -> concrete command -> machine instruction
```

这条路径使硬件或 simulator mismatch 能落到真正负责的层，而不是只剩一个 total-latency error。

## 第一个实现里程碑

下一条纵向切片应主动压低 target complexity：

1. 实现一个 queue-centric 和一个 non-queue-centric `VirtualTargetPlugin`，共同验证 coordination core；
2. 把现有 `SystemProfile` 适配到 normalized estimate provider；
3. legalize 当前 Transformer `PortablePlanIR`；
4. 构造具有显式 ordering/resource/buffer 与 typed extension 的单设备或简单 TP `ConcretePlanIR`；
5. 派生 timing projection、discrete-event result 与 `TimelineBundle`；
6. 输出 replayable `MachineIR` package；
7. 建立 simulation 与 replay 的 command/event correspondence level。

只有这条切片通过后，CUDA 或 LPU-specific emission 才应扩展 plugin protocol。LPU 可以先达到 architecture/simulation maturity，无需等待硬件；executable maturity 则必须等待稳定 ABI、runtime 与 profiler contract。

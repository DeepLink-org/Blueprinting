# 工作负载与映射模型

硬件结论的可信度取决于映射到 candidate 上的 workload。Blueprinting 把 workload 建模为结构化 computation、data movement、communication、dependency、phase 与 buffer lifetime，而不是一个 FLOP count 或 model-name lookup。

## Workload 语义

Workload specification 包含 model operator、tensor shape/dtype、training/inference semantic、state、control/effect、batch/sequence regime、optimizer 或 KV-cache behavior，以及 correctness-relevant dependency。

这些 fact 与 architecture 无关，描述必须发生什么，而不是由哪个 engine 执行或需要多长时间。

## Scenario Suite

Architecture exploration 评估 suite，而不是 isolated case。Suite 可以包括：

- 多种 global/micro-batch 与 context size 的 training；
- 跨 latency/throughput regime 的 inference prefill/decode；
- dense、MoE、multimodal 或 communication-heavy variant；
- precision 与 sparsity mode；
- representative、stress 与 projected future scenario；
- scenario weight 或 service-level constraint。

一个只在 headline benchmark 获胜、却无法覆盖预期 workload distribution 的 candidate 不应主导探索结论。

## 精确 Workload Facts

Progressive decomposition 推导可审计 quantity：

```text
operations by primitive and phase
read/write bytes by logical tensor
logical message bytes and collective semantics
dependencies and critical-path structure
persistent, checkpoint, temporary, and communication buffers
recomputation and recommunication work
```

这些 quantity 在 architecture candidate 之间保持守恒。Target-specific implementation 只有通过显式、带 source lineage 的 transformation，才能改变 physical traffic 或 executed operation。

## Mapping 决策

Mapping 把 architecture-independent work 连接到 candidate blueprint，并选择：

- tensor/data/pipeline/expert parallel decomposition；
- tiling、fusion、recomputation 与 layout；
- implementation 与 engine assignment；
- physical placement 与 communication route；
- queue order、synchronization 与 overlap opportunity；
- buffer memory level、allocation、reuse 与 movement。

Mapping 是 candidate evaluation 的一部分，因为强 architecture 在差 mapping 下可能表现很弱。Blueprinting 必须联合比较 architecture/mapping，同时保留每个决策来自哪一层。

## Workload Work 与 Implementation Work

模型区分 semantic work 与 implementation overhead。Logical all-reduce 具有 source message volume 与 reduction semantic；ring/tree algorithm 会增加 physical transfer 与 local reduction。Logical matrix multiply 具有 semantic operation；padding、packing 或 target-specific decomposition 会增加 implementation work。

这种分离可以呈现 hardware feature 的真实收益，并避免 target overhead 污染共同 workload baseline。

## 时间与内存结构

Peak memory 与 exposed communication 不能只从总量推断。因此 mapping model 会保留 dependency、buffer producer/consumer、lifetime、queue 与 synchronization，直到 architecture-bound simulation。

早期 analytical stage 可以使用 safe bound。有关 overlap、contention 或 memory reuse 的最终 claim 需要 concrete resource plan，或明确声明且带 uncertainty 的 approximation。

## Profiler 与 Simulator 关联

Stable lineage 把 model operation 连接到 distributed task、portable work、concrete command、simulator event 与 runtime instruction。不同层适合不同检查：

| 层 | 检查 |
|---|---|
| Workload semantic | shape、type、effect 与 numerical/reference behavior |
| Distributed work | shard reconstruction、rank/mesh consistency、communication volume |
| Portable mapping | operation/byte conservation、capability、abstract lifetime |
| Concrete mapping | placement、queue、synchronization、buffer 与 capacity legality |
| Simulation/runtime | event correlation、utilization、duration、counter、missing work |

Profiler observation 用于校准 evidence，不会重新定义 workload semantic。

## 当前 Transformer 覆盖

当前实现切片导入 typed decoder-only Transformer training specification，并把一个 local tensor-parallel block 分解为 forward、recomputation、activation-gradient、weight-gradient、optimizer 与 collective invocation。系统推导精确 operation/byte 并产生 `PortablePlanIR`。

Full-model PP/DP graph、完整 intermediate-buffer lifetime、inference prefill/decode、MoE 与通用 framework importer 仍为 planned。[Transformer 工作负载推导 reference](../design/passes/transformer.md)记录当前算法与限制。

## 与 Hardware Model 的 Contract

Workload side 提供 exact work、logical dependency、abstract resource need 与 capability requirement；hardware side 提供 resource、capability implementation、topology、physical constraint 与 evidence；architecture-binding step 在二者之间构造合法 mapped plan。

双方都不得夹带对方的事实。这个边界使同一 workload suite 能比较多个 blueprint，也使同一 blueprint 能比较多个 workload，而无需复制语义。

## 验证策略

验证从 exact fact 逐步推进到 system behavior：

1. unit-test primitive operation/byte algebra；
2. 验证 distribution/mapping 过程中的 conservation；
3. 与独立 reference 对比 aggregate work/memory；
4. 与 analytical tool 对比 schedule composition/bottleneck；
5. 用相同 command identity 关联 simulator/runtime event；
6. 只有 workload discrepancy 解决后，才校准 target behavior。

[Calculon 实验](../experiments/calculon-calibration.md)当前针对选定 Transformer training case 覆盖步骤 1–4。

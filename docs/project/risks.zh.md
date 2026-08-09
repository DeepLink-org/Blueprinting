# 架构风险登记表

本页记录会破坏硬件 candidate 可比较性、计划可执行性或结论可信度的系统性风险。它不是会议问题清单；无论合作对象是编译器、芯片、仿真、网络还是 ML 系统团队，都使用同一组 Gate。

**审计日期：** 2026-08-09

## 风险等级

- **P0**：不关闭就不能开始 architecture-bound scheduling 或宣称端到端能力；
- **P1**：可以原型验证，但不能发布稳定 contract 或性能结论；
- **P2**：不会立即破坏正确性，但会造成扩展、复现或解释成本。

## 当前风险

| ID | 等级 | 风险 | 当前证据 | 控制措施与关闭 Gate | 状态 |
|---|---|---|---|---|---|
| R-01 | P0 | Timeline 同时被理解为 execution truth 与 predicted trace | 已删除的 legacy stack 使用 `TimelineIR`；当前代码使用 `ConcretePlanIR` + derived timing | 统一术语；预测时间不参与 correctness；target-enforced time 只能作为 typed target semantic | 文档已纠正，代码 Gate 待实现 |
| R-02 | P0 | 通用 concrete schema 过拟合 GPU queue/stream 模型 | 当前 v1 只有 device、queue、buffer、command；没有 typed target schedule extension | 用 non-queue-centric virtual target 验证 common coordination core；禁止 correctness-critical free-form metadata | 开放 |
| R-03 | P0 | 文档把 schema/verifier 骨架写成 resource-complete plan | 当前没有 portable-to-concrete producer；route、occupancy 和 target-specific schedule semantic 不完整 | 状态页明确降级为 experimental contract；producer、consumer 与 end-to-end verifier 通过后再升级 | 已降级措辞，能力未实现 |
| R-04 | P1 | Planning evidence 与 evaluation evidence identity 混淆 | `ConcretePlanIR.evidence_revision` 记录构造输入，derived cost view 也有独立 evidence | Contract 中区分 construction provenance 与 re-evaluation revision；验证 re-cost 不会静默改 plan | 开放 |
| R-05 | P0 | Simulator/runtime “等价”被误读为时间行为相同 | 没有真实 backend 或 conformance evidence | 只承诺 command/event correspondence；定义 strict、bounded-divergence 和 partial-observation 等级 | 文档已纠正，测试待实现 |
| R-06 | P1 | `1.0.0` schema version 被误认为 public stability | 五个 IR schema 有版本，但 target producer/consumer 未贯通 | 明确 internal serialization version 不等于 compatibility promise；建立 graduation checklist | 文档已纠正，policy 待实现 |
| R-07 | P0 | “Runtime 零决策”忽略 backpressure、failure 与动态 duration | 当前没有 runtime contract | Runtime 禁止无界 global replanning，但保留 bounded safety/mechanism decisions；在 artifact 中声明 policy | 文档已纠正，runtime 待实现 |
| R-08 | P0 | LPU 特性提前污染 portable semantic | LPU ABI、capability 和 resource contract 尚未存在 | 按 Architecture/Simulation/Replay/Executable maturity 分层；physical detail 只在 target gate 后出现 | 受控 |
| R-09 | P1 | Simulator 与 emitter 各自补全缺失 schedule | 两条 production path 均未实现，缺少 cross-consumer conformance test | 两者消费同一 concrete digest + typed target extension；emitter decision delta 必须为空 | 开放 |
| R-10 | P1 | “形式化验证”被误读为 theorem-proved correctness | 当前是 typed schema、executable verifier、property test 与 external experiment | 对 claim 使用 assurance level；没有 mechanized proof 时不得使用 certified/proven 等表述 | 文档已纠正 |
| R-11 | P1 | Calibration 通过 case-specific 参数追平 reference | Calculon 路径已经禁止 per-case timing coefficient，但通用 evidence service 未实现 | 版本化 training/validation split、validity domain、uncertainty 与 held-out gate | 局部受控 |
| R-12 | P2 | Dynamic workload 被静态 plan 隐式排除 | 当前 slice 是显式静态 Transformer training configuration | 声明 shape/workload envelope；超出 envelope 时 diagnostic 或显式 replan，不能静默复用 | 开放 |

## 发布前的兼容性 Gate

任何 canonical schema、target plugin 或 timeline bundle 在声明 stable 前，至少需要：

1. 一个 production producer 和两个独立 consumer；
2. canonical round-trip 与 unknown-extension behavior；
3. negative verifier suite，而不只是 happy-path fixture；
4. 至少一个 queue-centric 和一个 non-queue-centric virtual target；
5. schema migration/compatibility policy；
6. provenance、evidence invalidation 和 replay test；
7. 文档中的 `Implemented` 状态与可运行端到端测试一致。

## 风险关闭原则

文档改名只能消除歧义，不能关闭实现风险。只有对应 producer、consumer、verifier、negative test 和可复现实验到位后，风险才能从“文档已纠正”变成“已关闭”。状态变化必须与[实现状态](status.md)和[路线图](roadmap.md)同步。

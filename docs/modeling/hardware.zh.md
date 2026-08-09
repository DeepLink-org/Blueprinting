# 硬件架构模型

Hardware model 是 candidate architecture 的语义描述，定义存在哪些 resource/capability，以及它们如何连接。Performance evidence 估算这些 resource 如何表现；deployment 标识具体 instance。只有保持三者独立，design-space exploration 才有意义。

!!! note "设计状态"
    当前 `SystemProfile` 只实现有限的 compute、memory 与 network evidence profile。这里描述的 hierarchical architecture schema 是目标设计，尚未端到端连接。

## Architecture、Deployment 与 Evidence

| 对象 | 拥有 | 不得拥有 |
|---|---|---|
| `ArchitectureBlueprint` | resource hierarchy、capability、topology、design variable、physical constraint | benchmark observation 或 concrete device allocation |
| `DeploymentProfile` | concrete device、link、reservation、environment 与 runtime revision | portable workload semantic 或 architecture search variable |
| `EvidenceSnapshot` | measurement、simulator result、calibrated model、uncertainty 与 provenance | architectural truth 或 workload meaning |

同一 blueprint 可以在多个 deployment/evidence snapshot 中测试；同一 evidence database 可以服务多个兼容 blueprint。改变 efficiency curve 不得改变 architecture digest。

## 分层组合

模型组合可复用的 typed component：

```text
cluster
  -> nodes and scale-out links
  -> packages and scale-up links
  -> dies/chiplets and die-to-die fabric
  -> tiles
  -> compute engines + local memories + on-chip network endpoints
```

每个 component 都有 stable identity、multiplicity、clock/power domain、parent scope、port 与 typed connection。Template 定义 architecture family；完成 parameter binding 后才形成一个 candidate blueprint。

## Compute 模型

Compute engine 声明 supported operation family、datatype/accumulation rule、shape/layout limit、concurrency、local-storage access、issue semantic 与 synchronization capability。例如 matrix array、vector unit、scalar/control core、reduction engine、DMA engine 和 collective accelerator。

Architecture model 只记录 capacity 与 legality。Throughput/latency curve 仍然是按 operation、shape、implementation、engine revision 与 context 索引的 evidence，避免在 engine definition 中嵌入一个乐观 universal utilization factor。

## Memory 模型

Memory space 声明：

- scope 与 visibility；
- capacity 与 allocatable reservation；
- bank、port、channel 与 address granularity；
- supported transfer、multicast、reduction 与 coherence semantic；
- 到 engine/其他 memory level 的 connectivity；
- alignment、layout 与 allocation constraint。

Bandwidth/latency evidence 可以随 transfer size、access pattern、bank mapping、occupancy 与 contention 变化。Buffer placement/lifetime 是 mapped plan 的决策，不是 memory component 的固有 field。

## Interconnect 模型

Interconnect 由 endpoint、link、router/switch、topology、routing capability、arbitration、buffer 与 supported communication operation 组成。同一抽象覆盖 NoC、chiplet link、scale-up fabric 与 scale-out network，同时允许 scope-specific extension。

简单 analytical provider 可以提供 latency/bandwidth 与 collective-volume curve；详细 simulator 消费相同 topology identity，并返回 route/link event、congestion 与 uncertainty。Logical workload message 继续是上游 fact。

## Capability 与 Legality 模型

Capability 回答 workload primitive 能否执行，以及如何执行，包括 operation semantic、precision、layout、shape range、memory accessibility、queue/event behavior、collective support、runtime requirement 与 ABI constraint。

对同一 blueprint/mapping request，legality 必须确定。缺失 support 时产生包含 source task、violated rule 与 considered alternative 的 diagnostic，绝不能在 estimate 中暗中转换为 infinite 或 zero cost。

## 物理指标

Area、energy、power、thermal、yield 与 cost 通过关联 component/system identity 的 provider 派生。早期 provider 可以是 analytical；对精选 candidate 可以接入 floorplan、RTL、power、thermal 或 packaging tool。

模型区分 additive、peak、averaged 与 state-dependent quantity。例如，除非 execution plan 确实能同时激活所有 component，否则 system power envelope 不能简单相加无关 peak component number。

## Fidelity 层级

同一 architecture identity 支持多个 evaluation level：

1. structural feasibility 与 exact capacity bound；
2. analytical throughput/bandwidth/energy model；
3. empirical 或 surrogate component model；
4. event-level resource simulation；
5. detailed network、hardware、power 或 thermal simulation；
6. prototype 或 silicon measurement。

Result 记录 fidelity 与 validity domain。更高 fidelity 细化 evidence，而不会静默改变 component semantic。

## Extension 与 Target Plugin

Common component 覆盖共享概念；target plugin 提供 architecture-specific capability、legality、implementation catalog、simulator adapter 与可选 MachineIR emission。LPU plugin 可以暴露专用 dataflow/memory operation，而不迫使 GPU candidate 使用同一 instruction model。

核心要求是 experiment semantic 可比较：共享 workload fact、显式 architecture binding、normalized evidence result、resource-correlated trace 与稳定 provenance。

## 版本化与验证

Architecture blueprint 验证 component reference、topology connectivity、scope、multiplicity、clock/power-domain rule、conditional variable、unit consistency 与 constraint satisfiability。Canonical digest 包含全部 semantic field，但排除 derived estimate。

Schema migration 必须显式。已发布 experiment 保留原始 blueprint/provider revision，使 architecture conclusion 在模型演进后仍可复现。

## 当前实现差距

`SystemProfile` 当前提供 matrix/vector throughput curve、memory capacity/bandwidth curve、network tier 与 collective model，用于 Calculon calibration。它尚未建模 component hierarchy、NoC、queue、power/area/cost、architecture variable 或通用 target capability graph。

第一步迁移应把 `SystemProfile` 包装为 minimal virtual `ArchitectureBlueprint` 的 evidence，在保持现有结果的同时引入上述分离。参见[路线图](../project/roadmap.md)。

# 探索工作空间

探索工作空间是 Blueprinting workbench 的交互面：一个由 framework-neutral `BlueprintingService` 支撑的单页 NiceGUI 应用。它是分析栈上的 presentation layer——不引入新的 semantic layer，也绝不成为第二份 workload truth。

!!! note "术语澄清"
    “工作空间”在文档中有两个含义，不要混淆：本页描述的是 workbench 的**探索工作空间**（四种模式的交互视图）；`PortablePlanIR` 中另有 `WORKSPACE` buffer role，表达抽象工作内存的容量约束，见[规划与执行 IR](../design/ir/planning-execution.md)。

## 工作空间是什么

工作空间按模式渲染四种视图，全部共享同一配置面。Analysis 在 UI event loop 之外执行；结果跨视图共享，配置变更后标记 stale；失败的 candidate 保留为结构化 diagnostic 而不是静默消失。它消费 `application/` 的 framework-neutral service，与 CLI 走同一套 typed contract。

## 共享配置面

单点与批量模式共享同一配置对话框：model/execution preset（来自 `data/` 的保留 JSON preset）、TP/PP/DP 拓扑、calibration mode（系统证据曲线 vs 理论峰值基线）与 candidate 范围。快速控件与完整配置双向同步。

## 工作空间模式

### 单点剖析（POINT LENS）

聚焦一个 case：执行 `ModelIR -> DistributedTaskIR -> PortablePlanIR` 推导，解析 evidence estimate，展示任务级贡献、资源约束与证据边界。推导失败时保留带 lineage 的 diagnostic，而不是给出虚构的 performance number。

### 批量探索（BATCH LENS）

把一组 TP/PP/DP case 作为整体：每个 case 独立推导，观察分布、上下界与可行边界，逐步收缩候选空间。失败项保留状态与诊断，并参与分布统计。

### 性能证据（EVIDENCE）

只读 evidence catalog：固定的 Vidur Phi-2/A100 exact-selector 记录。GEMM primitive 以相同 workload facts 对比 measured 与 analytical roofline；measured series 只显示精确 sample point，不声明插值；attention 与其他 operation 只进入 coverage catalog。

### 浮点分析（NUMERIC）

承载[浮点数数值分析](../design/numerical-analysis.md)面板：格式、编码、动态范围与运算边界。它独立于 workload analysis 可用，不要求先运行推导。

## 工作空间不变量

- 载入时不执行 eager analysis；
- analysis 在 UI event loop 外执行（`io_bound`），界面保持响应；
- 单个结果跨视图共享，配置变更后标记 stale；
- sweep candidate 保留 per-case 状态与失败诊断；
- evidence 只读：lab 从不回写 evidence 或 canonical IR；
- 可选的 portable dependency Chrome Trace 导出明确标记 `executable=false`，不是 `TimelineBundle`。

## 架构关系

Workbench 消费 `application/` service 并展示 canonical derivation audit、cost resolution 与 evidence catalog。它本身不构造 canonical plan；所有分析入口与 CLI 共享同一 typed contract 与 session binding。

## 当前边界

工作空间是分析/审计面，不是 simulator。它没有 first-class `ArchitectureBlueprint` 编辑、design-space search 或 discrete-event simulation；这些仍是 planned product slice（见[实现状态](../project/status.md)与[探索工作流](workflow.md)）。

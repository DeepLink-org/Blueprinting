# 浮点数数值分析

Blueprinting 的交互式浮点分析是工作空间中的一块**数值事实探索面**。它解码 IEEE 风格二进制浮点格式，暴露动态范围与可表示值，并量化候选 datatype 的量化误差与溢出/下溢风险。它是 presentation analysis view——既不预测执行时间，也不描述硬件行为，更不定义 workload semantic。

## 范围与动机

Datatype 是 first-class workload fact：`WorkloadFacts` 与 plan buffer 携带精确 datatype 与 bytes-per-element，Transformer 推导把 precision 当作 semantic 输入（详见 [workload 模型](../modeling/workload.md)）。在把 blueprint 或 mapping 承诺给某个 datatype（bf16 vs fp16 vs fp8）之前，架构师需要回答数值问题：

- 该格式实际能表示什么（动态范围、subnormal 区域、Inf/NaN）；
- 在 workload 关心的 magnitude 上，nearest-value 量化误差有多大；
- 加减乘除哪些运算在格式边界处会下溢或溢出。

浮点面板对标准格式与任意自定义布局交互式回答这些问题。

## 分析视图

面板围绕一个选定格式暴露五块相互关联的视图。

### 格式位宽对比

标准预设——`fp32`、`tf32`、`bf16`、`fp16`、`fp8(E5M2)`、`fp8(E4M3)`、`fp4(E2M1)`——以及由符号位（可选）、指数位（`2..8`）与尾数位（`0..23`）定义的自定义格式。位宽图跨格式比较 sign/exponent/mantissa 宽度。

### 位级解码

位编辑器按 IEEE zero/subnormal/special 规则解码一个具体 bit pattern：指数全 0 且 fraction 为 0 是 zero；指数全 0 且 fraction 非 0 是 subnormal；指数全 1 且 fraction 为 0 是 infinity；指数全 1 且 fraction 非 0 是 NaN。解码显示 category、raw exponent、significand 与 value。

### 动态范围与可表示值

面板报告选定格式的 bias、min normal、min subnormal 与 max finite。可表示值视图在观察窗口内枚举有限值，并标出 normal（蓝）与 subnormal（红）区域。

### 量化误差

Nearest-value 量化误差曲线对密集输入范围采样，报告到最近可表示值的绝对误差，暴露该格式在关注 magnitude 上的精度。

### 四则运算范围影响

对最多 64 个降采样的可表示值做笛卡尔组合，执行 `A + B`、`A − B`、`A × B`、`A ÷ B`，统计结果停留在 normal、落入 subnormal、下溢为零或溢出为 Inf/NaN 的计数。这是数值范围分析，不是硬件执行时间。

## 在分析架构中的位置

- 分析函数是纯函数且确定：位于 `src/blueprinting/workbench/float_analysis.py`，可表示值枚举位于 `src/blueprinting/fp/`。
- 面板不消费 `ModelIR`、`PortablePlanIR` 或任何 canonical IR。它是独立交互面，不是 canonical plan 的 derived view。
- 它不产生 cost estimate，因此不是 cost provider，也从不进入 evidence resolution 路径。
- 交互式枚举被刻意限制在 exponent + mantissa ≤ 12 位的格式，以保持 UI 响应。

## 当前边界与后续

**已实现**：纯分析函数、workbench 面板，以及覆盖动态范围、位解码、枚举限制与运算影响完整性的单元测试。

**未实现**：与 workload datatype binding 联动、为 training/inference plan 推导 per-operation 量化/溢出 guardrail，以及 fp8 scaling-policy 分析。未来的 slice 可以把数值事实暴露为按 plan 的 datatype digest 寻址的 analysis，而不改变 workload semantic。

承载该面板的探索工作空间见[探索工作空间](../exploration/workspace.md)；全项目实现状态以[实现状态](../project/status.md)为准。

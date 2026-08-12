# Pass 公式与 API

本页渲染全部公开 canonical Pass 的源码文档。核心公式直接写在 class docstring 中，因此实现推导发生变化时，代码参考也必须在同一个 review 面中变化。

## 覆盖与来源

| Pass | 边界 | 公式来源 | Commit 证据 |
|---|---|---|---|
| `DistributeTransformerTrainingPass` | Model → distributed task | Megatron-LM、selective recomputation | executable lineage rule + deterministic replay |
| `DistributeTransformerInferencePass` | Model → distributed task | Transformer、Megatron-LM、FlashAttention implementation boundary | executable lineage rule + exact-work test |
| `PlanTransformerTrainingPass` | Distributed task → portable plan | 已推导 Transformer work vector 的守恒映射 | executable work-equality rule |
| `PlanTransformerInferencePass` | Distributed task → portable plan | work conservation、保守 unfused workspace | executable work-equality rule |
| `BindReferenceQueueTargetPass` | Portable → concrete | 内部 deterministic contract，不声明论文算法 | structural verifier + cross-boundary predicate |
| `BindReferenceSlotTargetPass` | Portable → concrete | 内部 deterministic contract，不声明论文算法 | extension verifier + dependency-order proof |

!!! note "公式与性能证据的边界"

    下列 FLOPs、bytes、payload、shape 和 capacity 是 canonical workload fact。Latency、efficiency、overlap 与 uncertainty 属于 evidence/cost view，不能混入这些 Pass 公式。

## 训练分布推导

该 Pass 从一个语义 Transformer block 推导 local TP compute、collective、recompute、反向与 optimizer invocation。PP/DP 保留在 typed strategy 中，不伪造尚未物化的跨 stage DAG。

::: blueprinting.synthesizer.stages.distributed.passes.DistributeTransformerTrainingPass
    options:
      members:
        - run
      show_root_heading: false
      show_root_toc_entry: false

## 推理分布推导

该 Pass 分别处理 prefill 的 `q=c` 与 decode 的 `q=1`，显式推导 attention、projection、MLP、KV-cache 和 collective work。

::: blueprinting.synthesizer.stages.distributed.passes.DistributeTransformerInferencePass
    options:
      members:
        - run
      show_root_heading: false
      show_root_toc_entry: false

## 训练 PortablePlan 推导

该 Pass 不重新估算 work，而是证明 distributed invocation 的 operations/read/write/message 向量被无损写入 `WorkloadFacts`。

::: blueprinting.synthesizer.stages.portable_plan.passes.PlanTransformerTrainingPass
    options:
      members:
        - run
      show_root_heading: false
      show_root_toc_entry: false

## 推理 PortablePlan 推导

该 Pass 同时形成 persistent weight、KV state、boundary 和 conservative workspace obligation；target binding 之后才能用已选实现收紧 workspace。

::: blueprinting.synthesizer.stages.portable_plan.passes.PlanTransformerInferencePass
    options:
      members:
        - run
      show_root_heading: false
      show_root_toc_entry: false

## Reference queue binding

这是用于验证 ConcretePlan contract 的确定性构造，不是性能最优 scheduler。公式给出 buffer alignment recurrence 与 stable queue subsequence。

::: blueprinting.synthesizer.stages.concrete_plan.passes.BindReferenceQueueTargetPass
    options:
      members:
        - run
      show_root_heading: false
      show_root_toc_entry: false

## Reference slot/dataflow binding

这是用于验证 typed target extension 的确定性 cycle/slot 构造，并通过 PortablePlan 的拓扑顺序证明 dependency legality。

::: blueprinting.synthesizer.stages.concrete_plan.passes.BindReferenceSlotTargetPass
    options:
      members:
        - run
      show_root_heading: false
      show_root_toc_entry: false

## 研究来源

- Shoeybi et al., [Megatron-LM](https://arxiv.org/abs/1909.08053)。
- Narayanan et al., [Efficient Large-Scale Language Model Training Using Megatron-LM](https://arxiv.org/abs/2104.04473)。
- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762)。
- Korthikanti et al., [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198)。
- Dao et al., [FlashAttention](https://arxiv.org/abs/2205.14135)。
- Rajbhandari et al., [ZeRO](https://arxiv.org/abs/1910.02054)。

论文用于说明算法来源，不自动证明实现正确；实际 preservation 由 verifier、negative test、deterministic replay 和 baseline gate 证明。


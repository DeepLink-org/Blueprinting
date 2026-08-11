# Pass formulae and API

This page renders the source documentation for every public canonical pass.
The equations are part of the class docstrings, so changing an implemented
derivation and changing its code reference happen in the same review surface.

## Coverage and provenance

| Pass | Boundary | Formula provenance | Commit evidence |
|---|---|---|---|
| `DistributeTransformerTrainingPass` | Model → distributed task | Megatron-LM; selective recomputation | executable lineage rules + deterministic replay |
| `DistributeTransformerInferencePass` | Model → distributed task | Transformer; Megatron-LM; FlashAttention boundary | executable lineage rules + exact work tests |
| `PlanTransformerTrainingPass` | Distributed task → portable plan | conservation of the derived Transformer work vector | executable work-equality rules |
| `PlanTransformerInferencePass` | Distributed task → portable plan | work conservation; conservative unfused workspace | executable work-equality rules |
| `BindReferenceQueueTargetPass` | Portable → concrete | internal deterministic contract; no paper claim | structural verifier + cross-boundary predicates |
| `BindReferenceSlotTargetPass` | Portable → concrete | internal deterministic contract; no paper claim | extension verifier + dependency-order proof |

!!! note "Formula versus performance evidence"

    FLOPs, bytes, payloads, shapes, and capacities below are canonical workload
    facts. Latency, efficiency, overlap, and uncertainty belong to evidence/cost
    views and are deliberately absent from these pass equations.

## Training distribution

::: blueprinting.synthesizer.stages.distributed.passes.DistributeTransformerTrainingPass
    options:
      members:
        - run
      show_root_heading: false
      show_root_toc_entry: false

## Inference distribution

::: blueprinting.synthesizer.stages.distributed.passes.DistributeTransformerInferencePass
    options:
      members:
        - run
      show_root_heading: false
      show_root_toc_entry: false

## Training portable planning

::: blueprinting.synthesizer.stages.portable_plan.passes.PlanTransformerTrainingPass
    options:
      members:
        - run
      show_root_heading: false
      show_root_toc_entry: false

## Inference portable planning

::: blueprinting.synthesizer.stages.portable_plan.passes.PlanTransformerInferencePass
    options:
      members:
        - run
      show_root_heading: false
      show_root_toc_entry: false

## Reference queue binding

::: blueprinting.synthesizer.stages.concrete_plan.passes.BindReferenceQueueTargetPass
    options:
      members:
        - run
      show_root_heading: false
      show_root_toc_entry: false

## Reference slot/dataflow binding

::: blueprinting.synthesizer.stages.concrete_plan.passes.BindReferenceSlotTargetPass
    options:
      members:
        - run
      show_root_heading: false
      show_root_toc_entry: false

## Research sources

- Shoeybi et al., [Megatron-LM](https://arxiv.org/abs/1909.08053).
- Narayanan et al., [Efficient Large-Scale Language Model Training Using
  Megatron-LM](https://arxiv.org/abs/2104.04473).
- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
- Korthikanti et al., [Reducing Activation Recomputation in Large Transformer
  Models](https://arxiv.org/abs/2205.05198).
- Dao et al., [FlashAttention](https://arxiv.org/abs/2205.14135).
- Rajbhandari et al., [ZeRO](https://arxiv.org/abs/1910.02054).

These papers establish algorithmic provenance; repository verifiers and tests,
not citation alone, establish what this implementation actually preserves.


"""Verified derivations whose committed result is :class:`PortablePlanIR`.

These passes materialize exact target-neutral work. They cannot select kernels,
physical devices, queues, empirical durations, or wall-clock timestamps.
"""

from __future__ import annotations

from ...axes import BindingAxis
from ...dialects.transformer.inference_derivation import (
    INFERENCE_PLANNING_RULES,
    normalize_inference_plan,
)
from ...dialects.transformer.training_derivation import (
    TRAINING_PLANNING_RULES,
    normalize_training_plan,
)
from ...passes.authoring import DerivationPass, PassContext, derivation
from ..distributed.ir import DistributedTaskIR
from .ir import PortablePlanIR


@derivation(
    "transformer-plan-work",
    revision="1",
    bindings=(BindingAxis.STRATEGY,),
    rules=TRAINING_PLANNING_RULES,
    normalizer=normalize_training_plan,
)
class PlanTransformerTrainingPass(DerivationPass[DistributedTaskIR, PortablePlanIR]):
    r"""Materialize exact training work without choosing a hardware target.

    This pass is a semantics-preserving reification, not a second estimator.
    For every distributed invocation ``i`` it copies the exact work vector

    $$
    \mathbf{w}_i=(F_i,R_i,W_i,M_i)
    $$

    into ``WorkloadFacts(operations, read_bytes, write_bytes, message_bytes)``.
    Resource requirements are non-lossy projections of that vector:

    $$
    Q_{\mathrm{compute}}=F_i,\qquad
    Q_{\mathrm{memory}}=R_i+W_i,\qquad
    Q_{\mathrm{network}}=M_i.
    $$

    Exact block memory is reduced from the structurally derived layer facts;
    for example, stored activations are

    $$
    C_{\mathrm{act}}=\sum_{\ell}
      \left(A_\ell-O_\ell\,[\neg\mathrm{storeOutput}_\ell]
      -A_\ell\,[\neg\mathrm{storeActivation}_\ell]\right).
    $$

    The executable pass rules verify work conservation and lineage before the
    snapshot is committed.  The scientific provenance is inherited from the
    Transformer decomposition rather than introducing a new performance model.

    References:
        - Shoeybi et al., [Megatron-LM](https://arxiv.org/abs/1909.08053).
        - Rajbhandari et al., [ZeRO](https://arxiv.org/abs/1910.02054).
        - Korthikanti et al., [Selective activation
          recomputation](https://arxiv.org/abs/2205.05198).
    """

    def run(self, ir: DistributedTaskIR, context: PassContext) -> PortablePlanIR:
        return normalize_training_plan(ir, context.session)


@derivation(
    "transformer-inference-plan-work",
    revision="1",
    bindings=(BindingAxis.STRATEGY,),
    rules=INFERENCE_PLANNING_RULES,
    normalizer=normalize_inference_plan,
)
class PlanTransformerInferencePass(DerivationPass[DistributedTaskIR, PortablePlanIR]):
    r"""Materialize exact inference work without target placement or timing.

    Each inference invocation is mapped homomorphically into a portable task:

    $$
    (F_i,R_i,W_i,M_i)_{\mathrm{distributed}}
      =(F_i,R_i,W_i,M_i)_{\mathrm{portable}}.
    $$

    The conservative workspace bound intentionally assumes an unfused score
    materialization until target binding selects an implementation.  With
    boundary ``D=bqHe`` and local intermediate element counts
    ``3bqH/t``, ``b(h/t)qc``, and ``bqF/t``, it is

    $$
    C_{\mathrm{workspace}}
      =D+e\max\left(3bq\frac{H}{t},
                     b\frac{h}{t}qc,
                     bq\frac{F}{t}\right).
    $$

    This bound is a portable capacity obligation.  A target implementation such
    as tiled exact attention may replace its workspace only during verified
    target binding; it may not rewrite the canonical operation count.

    References:
        - Vaswani et al., [Attention Is All You
          Need](https://arxiv.org/abs/1706.03762).
        - Dao et al., [FlashAttention](https://arxiv.org/abs/2205.14135).
    """

    def run(self, ir: DistributedTaskIR, context: PassContext) -> PortablePlanIR:
        return normalize_inference_plan(ir, context.session)


__all__ = ["PlanTransformerInferencePass", "PlanTransformerTrainingPass"]

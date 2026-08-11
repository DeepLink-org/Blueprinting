"""Verified derivations whose committed result is :class:`DistributedTaskIR`.

The pass contracts live beside the output IR. Transformer-specific algebra is
kept in its dialect module and is called as a pure derivation.
"""

from __future__ import annotations

from ...axes import BindingAxis
from ...dialects.transformer.inference_derivation import (
    INFERENCE_DISTRIBUTION_RULES,
    normalize_inference_distribution,
)
from ...dialects.transformer.training_derivation import (
    TRAINING_DISTRIBUTION_RULES,
    normalize_training_distribution,
)
from ...passes.authoring import DerivationPass, PassContext, derivation
from ..model.ir import ModelIR
from .ir import DistributedTaskIR


@derivation(
    "transformer-distribute",
    revision="1",
    bindings=(BindingAxis.WORKLOAD, BindingAxis.STRATEGY),
    rules=TRAINING_DISTRIBUTION_RULES,
    normalizer=normalize_training_distribution,
)
class DistributeTransformerTrainingPass(DerivationPass[ModelIR, DistributedTaskIR]):
    r"""Expand one training block into logical TP-local and collective tasks.

    The pass interprets a typed ``TP × PP × DP`` strategy, but this snapshot
    materializes one local tensor-parallel block only.  Let ``B`` be the
    microbatch, ``S`` the sequence length, ``H`` the hidden width, ``F`` the
    feed-forward width, ``t`` the TP degree, and ``e`` bytes per element.  A
    matrix product with shapes ``[m,n] × [n,k]`` contributes

    $$
    W_{\mathrm{gemm}} = 2mnk.
    $$

    Therefore the local attention projections contribute

    $$
    W_{\mathrm{QKV}}=\frac{6BSH^2}{t},\qquad
    W_{\mathrm{attn\,matmul}}=\frac{4BS^2H}{t},
    $$

    and the two MLP projections contribute

    $$
    W_{\mathrm{MLP}}=\frac{4BSHF}{t}.
    $$

    At a TP semantic boundary, the logical payload and local reduction work are

    $$
    M_{\mathrm{TP}}=BSH\,e,\qquad
    W_{\mathrm{reduce}}=BSH\frac{t-1}{t}.
    $$

    The derivation emits explicit forward, recompute, activation-gradient,
    weight-gradient, optimizer, and collective invocations.  It does not attach
    latency or choose a physical collective algorithm.

    References:
        - Shoeybi et al., [Megatron-LM](https://arxiv.org/abs/1909.08053).
        - Narayanan et al., [Efficient Large-Scale Language Model
          Training](https://arxiv.org/abs/2104.04473).
        - Korthikanti et al., [Reducing Activation
          Recomputation](https://arxiv.org/abs/2205.05198).
    """

    def run(self, ir: ModelIR, context: PassContext) -> DistributedTaskIR:
        return normalize_training_distribution(ir, context.session)


@derivation(
    "transformer-inference-distribute",
    revision="1",
    bindings=(BindingAxis.WORKLOAD, BindingAxis.STRATEGY),
    rules=INFERENCE_DISTRIBUTION_RULES,
    normalizer=normalize_inference_distribution,
)
class DistributeTransformerInferencePass(DerivationPass[ModelIR, DistributedTaskIR]):
    r"""Expand one inference phase into logical TP-local and collective tasks.

    Let ``b`` be batch size, ``q`` the query-token count, ``c`` the visible KV
    context, ``H`` hidden width, ``F`` feed-forward width, ``h`` attention-head
    count, ``t`` TP degree, and ``e`` bytes per element.  Prefill uses ``q=c``;
    decode uses ``q=1``.  The exact local attention-core work is

    $$
    W_{\mathrm{attention}}
      = \frac{4bqcH}{t}
      + 5b\frac{h}{t}qc,
    $$

    where the first term is ``QKᵀ`` plus ``PV`` and the second is the explicit
    softmax model.  Projection and MLP work are

    $$
    W_{\mathrm{QKV}}=\frac{6bqH^2}{t},\quad
    W_{\mathrm{out}}=\frac{2bqH^2}{t},\quad
    W_{\mathrm{MLP}}=\frac{4bqHF}{t}.
    $$

    The per-rank KV state retained by the derived analysis is

    $$
    C_{\mathrm{KV}}=2bc\frac{H}{t}e.
    $$

    The pass marks KV reads/writes as effects and leaves fused/paged attention
    as target-neutral implementation alternatives rather than assuming them.

    References:
        - Vaswani et al., [Attention Is All You
          Need](https://arxiv.org/abs/1706.03762).
        - Shoeybi et al., [Megatron-LM](https://arxiv.org/abs/1909.08053).
        - Dao et al., [FlashAttention](https://arxiv.org/abs/2205.14135).
    """

    def run(self, ir: ModelIR, context: PassContext) -> DistributedTaskIR:
        return normalize_inference_distribution(ir, context.session)


__all__ = ["DistributeTransformerInferencePass", "DistributeTransformerTrainingPass"]

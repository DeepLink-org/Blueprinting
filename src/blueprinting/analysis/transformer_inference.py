"""Exact per-block work derivation for decoder-only Transformer inference.

The task boundaries intentionally follow observable implementation families
(projection, RoPE, KV save, fused attention, MLP and collectives).  This keeps
the analytical model inspectable and permits post-hoc comparison with external
baselines without allowing those baselines to change workload semantics.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..compiler.bindings import InferencePhase
from ..compiler.codec import record_type
from ..compiler.ir import CollectiveKind
from ..compiler.models.transformer import TransformerModelSpec
from ..compiler.models.transformer_inference import TransformerInferenceExecutionSpec
from .transformer_workload import EngineKind, PhaseWork

# Keep the legacy codec namespace as a stable serialized identity.


@record_type("compiler.analysis.inference_invocation.v1")
@dataclass(frozen=True)
class InferenceInvocation:
    """One target-neutral component invocation for a single decoder block."""

    name: str
    source_layer: str
    primitive: str
    phase: InferencePhase
    engine: EngineKind
    work: PhaseWork
    collective: CollectiveKind | None = None
    network_tier: int | None = None

    def __post_init__(self) -> None:
        for field_name in ("name", "source_layer", "primitive"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field_name} must not be empty")
        if not isinstance(self.phase, InferencePhase):
            raise TypeError("phase must be InferencePhase")
        if not isinstance(self.engine, EngineKind):
            raise TypeError("engine must be EngineKind")
        if not isinstance(self.work, PhaseWork):
            raise TypeError("work must be PhaseWork")
        if self.engine is EngineKind.COLLECTIVE:
            if self.collective is None or self.network_tier is None:
                raise ValueError("collective invocations require kind and network tier")
        elif self.collective is not None or self.network_tier is not None:
            raise ValueError("local invocations cannot carry collective metadata")


@record_type("compiler.analysis.inference_block_memory_facts.v1")
@dataclass(frozen=True)
class InferenceBlockMemoryFacts:
    """Per-rank storage for one tensor-parallel block shard and phase."""

    weights: int
    kv_cache: int
    working_upper_bound: int
    boundary: int

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")


def _work(*, operations: int = 0, read: int = 0, write: int = 0, message: int = 0) -> PhaseWork:
    return PhaseWork(operations=operations, read_bytes=read, write_bytes=write, message_bytes=message)


def compile_transformer_inference_block(
    model: TransformerModelSpec,
    execution: TransformerInferenceExecutionSpec,
    *,
    phase: InferencePhase,
    batch_size: int,
    context_tokens: int,
) -> tuple[tuple[InferenceInvocation, ...], InferenceBlockMemoryFacts]:
    """Derive exact work for one local block at one inference phase point.

    ``context_tokens`` is the number of keys visible to attention.  Prefill has
    ``query_tokens == context_tokens``; decode has one query token and includes
    the newly appended token in the context.
    """

    execution.validate_model(model)
    if not isinstance(phase, InferencePhase):
        raise TypeError("phase must be InferencePhase")
    for name, value in (("batch_size", batch_size), ("context_tokens", context_tokens)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    b = batch_size
    q = context_tokens if phase is InferencePhase.PREFILL else 1
    c = context_tokens
    h = model.hidden_size
    f = model.feedforward_size
    tp = execution.tensor_parallel
    heads = model.attention_heads // tp
    local_h = h // tp
    local_f = f // tp
    element_bytes = execution.bytes_per_element
    token_elements = b * q * h
    local_token_elements = b * q * local_h
    score_elements = b * heads * q * c

    invocations: list[InferenceInvocation] = []

    def local(
        name: str,
        source_layer: str,
        primitive: str,
        engine: EngineKind,
        work: PhaseWork,
    ) -> None:
        invocations.append(
            InferenceInvocation(
                name=f"{phase.value}.{name}",
                source_layer=source_layer,
                primitive=primitive,
                phase=phase,
                engine=engine,
                work=work,
            )
        )

    def all_reduce(name: str, source_layer: str) -> None:
        if tp == 1:
            return
        invocations.append(
            InferenceInvocation(
                name=f"{phase.value}.{name}",
                source_layer=source_layer,
                primitive="all_reduce",
                phase=phase,
                engine=EngineKind.COLLECTIVE,
                work=_work(message=token_elements * element_bytes),
                collective=CollectiveKind.ALL_REDUCE,
                network_tier=execution.tensor_parallel_network,
            )
        )

    norm_parameter_elements = 2 * h
    norm_work = _work(
        operations=5 * token_elements,
        read=(token_elements + norm_parameter_elements) * element_bytes,
        write=token_elements * element_bytes,
    )
    local("input_norm", "attention.input_norm", "input_layernorm", EngineKind.VECTOR, norm_work)

    qkv_weight_elements = 3 * h * h // tp
    local(
        "attention_pre_projection",
        "attention.qkv",
        "attention_pre_projection",
        EngineKind.MATRIX,
        _work(
            operations=6 * b * q * h * h // tp,
            read=(token_elements + qkv_weight_elements) * element_bytes,
            write=3 * local_token_elements * element_bytes,
        ),
    )
    local(
        "attention_rope",
        "attention.rope",
        "attention_rope",
        EngineKind.VECTOR,
        _work(
            operations=12 * local_token_elements,
            read=2 * local_token_elements * element_bytes,
            write=2 * local_token_elements * element_bytes,
        ),
    )
    local(
        "attention_kv_cache_save",
        "attention.kv_cache",
        "attention_kv_cache_save",
        EngineKind.VECTOR,
        _work(
            read=2 * local_token_elements * element_bytes,
            write=2 * local_token_elements * element_bytes,
        ),
    )
    local(
        "attention_core",
        "attention.core",
        "attention_core",
        EngineKind.MATRIX,
        _work(
            operations=4 * b * q * c * h // tp + 5 * score_elements,
            read=(local_token_elements + 2 * b * c * local_h) * element_bytes,
            write=local_token_elements * element_bytes,
        ),
    )
    output_weight_elements = h * h // tp
    local(
        "attention_post_projection",
        "attention.output",
        "attention_post_projection",
        EngineKind.MATRIX,
        _work(
            operations=2 * b * q * h * h // tp,
            read=(local_token_elements + output_weight_elements) * element_bytes,
            write=token_elements * element_bytes,
        ),
    )
    all_reduce("attention_all_reduce", "attention.output")
    local(
        "attention_residual",
        "attention.residual",
        "residual_add",
        EngineKind.VECTOR,
        _work(
            operations=token_elements,
            read=2 * token_elements * element_bytes,
            write=token_elements * element_bytes,
        ),
    )
    local("post_attention_norm", "mlp.input_norm", "post_attention_layernorm", EngineKind.VECTOR, norm_work)

    mlp_weight_elements = h * f // tp
    local(
        "mlp_up_projection",
        "mlp.up",
        "mlp_up_projection",
        EngineKind.MATRIX,
        _work(
            operations=2 * b * q * h * f // tp,
            read=(token_elements + mlp_weight_elements) * element_bytes,
            write=b * q * local_f * element_bytes,
        ),
    )
    local(
        "mlp_activation",
        "mlp.activation",
        "mlp_activation",
        EngineKind.VECTOR,
        _work(
            operations=8 * b * q * local_f,
            read=b * q * local_f * element_bytes,
            write=b * q * local_f * element_bytes,
        ),
    )
    local(
        "mlp_down_projection",
        "mlp.down",
        "mlp_down_projection",
        EngineKind.MATRIX,
        _work(
            operations=2 * b * q * h * f // tp,
            read=(b * q * local_f + mlp_weight_elements) * element_bytes,
            write=token_elements * element_bytes,
        ),
    )
    all_reduce("mlp_all_reduce", "mlp.down")
    local(
        "mlp_residual",
        "mlp.residual",
        "residual_add",
        EngineKind.VECTOR,
        _work(
            operations=token_elements,
            read=2 * token_elements * element_bytes,
            write=token_elements * element_bytes,
        ),
    )

    matrix_weight_elements = (4 * h * h + 2 * h * f) // tp
    replicated_norm_elements = 4 * h
    block_weights = (matrix_weight_elements + replicated_norm_elements) * element_bytes
    block_kv_cache = 2 * b * c * local_h * element_bytes
    boundary = token_elements * element_bytes
    qkv_working = 3 * local_token_elements
    attention_working = score_elements
    mlp_working = b * q * local_f
    # PortablePlanIR has not selected a fused attention implementation yet.
    # Retain the unfused score materialization as a conservative capacity
    # bound; target binding may replace it with implementation workspace.
    working_upper_bound = boundary + max(qkv_working, attention_working, mlp_working) * element_bytes
    return tuple(invocations), InferenceBlockMemoryFacts(
        weights=block_weights,
        kv_cache=block_kv_cache,
        working_upper_bound=working_upper_bound,
        boundary=boundary,
    )

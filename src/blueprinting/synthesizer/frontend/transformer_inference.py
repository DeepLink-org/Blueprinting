"""Transformer inference adapters for formal synthesis."""

from __future__ import annotations

from typing import Any

from blueprinting.workload import TransformerInferenceExecutionSpec, TransformerModelSpec

from ..axes import BindingAxis
from ..bindings import BindingSet, InferencePhase, StrategyBinding, WorkloadBinding, WorkloadMode
from ..expr import Symbol
from ..frozen import FrozenDict
from ..ids import Lineage, NodeId, ValueId
from ..ir import Effect, EffectKind, ModelIR, ModelOperation, ModelValue, OperationName, TensorType, ValueRole
from ..session import SynthesisSession

_SUPPORTED_DATATYPES = frozenset({"float8", "float16", "bfloat16", "float32"})


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def build_transformer_inference_model_ir(
    model: TransformerModelSpec,
    *,
    datatype: str = "float16",
) -> ModelIR:
    """Import a phase-neutral decoder inference operation.

    KV cache is an explicit mutable semantic resource. Its concrete context
    extent is supplied by a phase workload binding, not embedded in the model.
    """

    if datatype not in _SUPPORTED_DATATYPES:
        raise ValueError(f"unsupported datatype: {datatype!r}")
    batch = Symbol("batch_size", BindingAxis.WORKLOAD, positive=True)
    context = Symbol("sequence_length", BindingAxis.WORKLOAD, positive=True)
    query = Symbol("query_tokens", BindingAxis.WORKLOAD, positive=True)
    hidden_type = TensorType((batch, query, model.hidden_size), datatype)
    cache_type = TensorType((model.block_count, 2, batch, context, model.hidden_size), datatype)
    input_id = ValueId.derive("transformer-inference", model, "hidden-input")
    cache_id = ValueId.derive("transformer-inference", model, "kv-cache")
    output_id = ValueId.derive("transformer-inference", model, "hidden-output")
    operation_id = NodeId.derive("transformer-inference", model, "decoder")
    return ModelIR(
        name=f"{model.name}-inference",
        values=(
            ModelValue(
                input_id, hidden_type, ValueRole.INPUT, Lineage.root("transformer-inference-import"), "hidden_input"
            ),
            ModelValue(
                cache_id, cache_type, ValueRole.KV_CACHE, Lineage.root("transformer-inference-import"), "kv_cache"
            ),
            ModelValue(
                output_id,
                hidden_type,
                ValueRole.OUTPUT,
                Lineage.lowered("transformer-inference-semantic-op", (input_id,)),
                "hidden_output",
            ),
        ),
        operations=(
            ModelOperation(
                operation_id,
                OperationName("transformer", "decoder_inference"),
                (input_id, cache_id),
                (output_id,),
                Lineage.root("transformer-inference-import"),
                effects=(Effect(EffectKind.STATE, "kv_cache"),),
                attributes=FrozenDict({"model_spec": model}),
            ),
        ),
        inputs=(input_id,),
        outputs=(output_id,),
        attributes=FrozenDict(
            {
                "model_family": "decoder-only-transformer",
                "workload_mode": WorkloadMode.INFERENCE.value,
            }
        ),
    )


def inference_synthesis_session_for(
    model: TransformerModelSpec,
    execution: TransformerInferenceExecutionSpec,
    *,
    phase: InferencePhase,
    batch_size: int,
    context_tokens: int,
) -> SynthesisSession:
    """Create an explicit phase binding for static inference specialization."""

    execution.validate_model(model)
    _positive_integer(batch_size, "batch_size")
    _positive_integer(context_tokens, "context_tokens")
    if not isinstance(phase, InferencePhase):
        raise TypeError("phase must be InferencePhase")
    query_tokens = context_tokens if phase is InferencePhase.PREFILL else 1
    workload = WorkloadBinding(
        WorkloadMode.INFERENCE,
        batch_size=batch_size,
        sequence_length=context_tokens,
        inference_phase=phase,
        attributes=FrozenDict(
            {
                "query_tokens": query_tokens,
                "context_tokens": context_tokens,
            }
        ),
    )
    strategy = StrategyBinding(
        tensor_parallel=execution.tensor_parallel,
        pipeline_parallel=execution.pipeline_parallel,
        data_parallel=execution.replicas,
        recompute_policy="none",
        pipeline_policy="static-inference",
        attributes=FrozenDict({"inference_execution_spec": execution}),
    )
    return SynthesisSession(
        bindings=BindingSet(workload=workload, strategy=strategy),
        features=frozenset({"transformer-inference-analysis-v1", f"inference-{phase.value}"}),
    )

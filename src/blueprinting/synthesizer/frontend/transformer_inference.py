"""Transformer inference adapters for formal synthesis."""

from __future__ import annotations

from typing import Any

from blueprinting.mapping import TransformerInferenceMappingSpec
from blueprinting.schema.frozen import FrozenDict
from blueprinting.workload import TransformerDataType, TransformerModelSpec, require_transformer_data_type

from ..axes import BindingAxis
from ..bindings import BindingSet, InferencePhase, InferenceWorkload, StrategyBinding, WorkloadBinding
from ..dialects.transformer import (
    TransformerInferenceStrategySemantic,
    TransformerInferenceWorkloadSemantic,
    TransformerModelOperationSemantic,
)
from ..expr import Symbol
from ..ids import Lineage, NodeId, ValueId
from ..session import SynthesisSession
from ..stages.common import Effect, EffectKind, OperationName, TensorType
from ..stages.model.ir import ModelIR, ModelOperation, ModelValue, ValueRole


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def build_transformer_inference_model_ir(
    model: TransformerModelSpec,
    *,
    datatype: TransformerDataType = "float16",
) -> ModelIR:
    """Import a phase-neutral decoder inference operation.

    KV cache is an explicit mutable semantic resource. Its concrete context
    extent is supplied by a phase workload binding, not embedded in the model.
    """

    datatype = require_transformer_data_type(datatype)
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
                semantic=TransformerModelOperationSemantic(model),
            ),
        ),
        inputs=(input_id,),
        outputs=(output_id,),
        attributes=FrozenDict(
            {
                "model_family": "decoder-only-transformer",
                "workload_mode": "inference",
            }
        ),
    )


def inference_synthesis_session_for(
    model: TransformerModelSpec,
    mapping: TransformerInferenceMappingSpec,
    *,
    phase: InferencePhase,
    batch_size: int,
    context_tokens: int,
    datatype: TransformerDataType = "float16",
) -> SynthesisSession:
    """Create an explicit phase binding for static inference specialization."""

    mapping.validate_model(model)
    _positive_integer(batch_size, "batch_size")
    _positive_integer(context_tokens, "context_tokens")
    datatype = require_transformer_data_type(datatype)
    if not isinstance(phase, InferencePhase):
        raise TypeError("phase must be InferencePhase")
    workload = WorkloadBinding(
        InferenceWorkload(phase),
        semantic=TransformerInferenceWorkloadSemantic(batch_size, context_tokens, datatype),
    )
    strategy = StrategyBinding(
        semantic=TransformerInferenceStrategySemantic(mapping),
    )
    return SynthesisSession(
        bindings=BindingSet(workload=workload, strategy=strategy),
        features=frozenset({"transformer-inference-analysis"}),
    )

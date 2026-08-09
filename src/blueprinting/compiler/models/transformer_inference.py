"""Typed frontend facts for decoder-only Transformer inference.

Inference keeps three concerns separate:

* :class:`TransformerModelSpec` describes model structure;
* :class:`TransformerInferenceExecutionSpec` describes a logical mapping;
* :class:`TransformerInferenceRequestSpec` describes one request cohort.

The frontend emits a phase-neutral semantic operation.  ``PREFILL`` and
``DECODE`` become explicit workload bindings, so the same model snapshot can
be specialized independently for request-level simulation later on.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..axes import BindingAxis
from ..bindings import (
    BindingSet,
    InferencePhase,
    StrategyBinding,
    WorkloadBinding,
    WorkloadMode,
)
from ..codec import record_type
from ..expr import Symbol
from ..frozen import FrozenDict
from ..ids import Lineage, NodeId, ValueId
from ..ir import Effect, EffectKind, ModelIR, ModelOperation, ModelValue, OperationName, TensorType, ValueRole
from ..session import CompilationSession
from .transformer import TransformerModelSpec

_SUPPORTED_DATATYPES = frozenset({"float8", "float16", "bfloat16", "float32"})


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@record_type("compiler.transformer.inference_execution_spec.v1")
@dataclass(frozen=True)
class TransformerInferenceExecutionSpec:
    """Target-neutral logical mapping for an inference replica."""

    world_size: int
    tensor_parallel: int
    pipeline_parallel: int
    replicas: int
    datatype: str
    tensor_parallel_network: int
    pipeline_parallel_network: int

    def __post_init__(self) -> None:
        for field_name in ("world_size", "tensor_parallel", "pipeline_parallel", "replicas"):
            _positive_integer(getattr(self, field_name), field_name)
        for field_name in ("tensor_parallel_network", "pipeline_parallel_network"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        if self.world_size != self.tensor_parallel * self.pipeline_parallel * self.replicas:
            raise ValueError("world_size must equal tensor_parallel * pipeline_parallel * replicas")
        if self.datatype not in _SUPPORTED_DATATYPES:
            raise ValueError(f"unsupported datatype: {self.datatype!r}")

    @property
    def bytes_per_element(self) -> int:
        return {"float8": 1, "float16": 2, "bfloat16": 2, "float32": 4}[self.datatype]

    def validate_model(self, model: TransformerModelSpec) -> None:
        """Reject mappings whose local tensor shapes are not integral."""

        divisibility = {
            "hidden_size": model.hidden_size,
            "feedforward_size": model.feedforward_size,
            "attention_heads": model.attention_heads,
        }
        for name, value in divisibility.items():
            if value % self.tensor_parallel:
                raise ValueError(f"{name} must be divisible by tensor_parallel")
        if self.pipeline_parallel > model.block_count:
            raise ValueError("pipeline_parallel cannot exceed block_count")
        if model.block_count % self.pipeline_parallel:
            raise ValueError("pipeline_parallel must divide block_count for static inference planning")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> TransformerInferenceExecutionSpec:
        replicas = data.get("replicas", data.get("data_par", 1))
        tensor_parallel = data["tensor_par"]
        pipeline_parallel = data["pipeline_par"]
        return cls(
            world_size=data.get("num_procs", tensor_parallel * pipeline_parallel * replicas),
            tensor_parallel=tensor_parallel,
            pipeline_parallel=pipeline_parallel,
            replicas=replicas,
            datatype=data["datatype"],
            tensor_parallel_network=data.get("tensor_par_net", 0),
            pipeline_parallel_network=data.get("pipeline_par_net", 0),
        )


@record_type("compiler.transformer.inference_request_spec.v1")
@dataclass(frozen=True)
class TransformerInferenceRequestSpec:
    """A homogeneous request cohort before online scheduling is applied."""

    batch_size: int
    prompt_tokens: int
    generated_tokens: int

    def __post_init__(self) -> None:
        for field_name in ("batch_size", "prompt_tokens", "generated_tokens"):
            _positive_integer(getattr(self, field_name), field_name)

    @property
    def decode_iterations(self) -> int:
        """Iterations after prefill; prefill itself emits the first token."""

        return self.generated_tokens - 1

    @property
    def final_context_tokens(self) -> int:
        """Largest KV context consumed while producing this cohort."""

        return self.prompt_tokens + self.decode_iterations

    def decode_contexts(self) -> range:
        return range(self.prompt_tokens + 1, self.final_context_tokens + 1)

    def validate_model(self, model: TransformerModelSpec) -> None:
        if self.final_context_tokens > model.sequence_length:
            raise ValueError("prompt_tokens + generated_tokens - 1 cannot exceed the model sequence_length")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> TransformerInferenceRequestSpec:
        return cls(
            batch_size=data["batch_size"],
            prompt_tokens=data["prompt_tokens"],
            generated_tokens=data["generated_tokens"],
        )


def build_transformer_inference_model_ir(
    model: TransformerModelSpec,
    *,
    datatype: str = "float16",
) -> ModelIR:
    """Import a phase-neutral decoder inference operation.

    KV cache is an explicit mutable semantic resource.  Its concrete context
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


def inference_compilation_session_for(
    model: TransformerModelSpec,
    execution: TransformerInferenceExecutionSpec,
    *,
    phase: InferencePhase,
    batch_size: int,
    context_tokens: int,
) -> CompilationSession:
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
    return CompilationSession(
        bindings=BindingSet(workload=workload, strategy=strategy),
        features=frozenset({"transformer-inference-analysis-v1", f"inference-{phase.value}"}),
    )

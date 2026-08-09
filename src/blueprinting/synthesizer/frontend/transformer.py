"""Transformer training adapters for formal synthesis.

This module is the explicit dependency seam between target-neutral workload
contracts and Blueprinting's canonical representation/binding machinery.
"""

from __future__ import annotations

from blueprinting.mapping import TransformerTrainingMappingSpec
from blueprinting.schema.frozen import FrozenDict
from blueprinting.workload import TransformerModelSpec, TransformerTrainingWorkloadSpec

from ..axes import BindingAxis
from ..bindings import BindingSet, StrategyBinding, WorkloadBinding, WorkloadMode
from ..expr import Symbol
from ..ids import Lineage, NodeId, ValueId
from ..ir import ModelIR, ModelOperation, ModelValue, OperationName, TensorType, ValueRole
from ..session import SynthesisSession

_SUPPORTED_DATATYPES = frozenset({"float8", "float16", "bfloat16", "float32"})


def build_transformer_model_ir(model: TransformerModelSpec, *, datatype: str = "float16") -> ModelIR:
    """Import a model as one semantic operation before structural lowering."""

    if datatype not in _SUPPORTED_DATATYPES:
        raise ValueError(f"unsupported datatype: {datatype!r}")
    batch = Symbol("microbatch_size", BindingAxis.WORKLOAD, positive=True)
    sequence = Symbol("sequence_length", BindingAxis.WORKLOAD, positive=True)
    tensor_type = TensorType((batch, sequence, model.hidden_size), datatype)
    input_id = ValueId.derive("transformer", model, "hidden-input")
    output_id = ValueId.derive("transformer", model, "hidden-output")
    operation_id = NodeId.derive("transformer", model, "decoder-training")
    return ModelIR(
        name=model.name,
        values=(
            ModelValue(input_id, tensor_type, ValueRole.INPUT, Lineage.root("transformer-import"), "hidden_input"),
            ModelValue(
                output_id,
                tensor_type,
                ValueRole.OUTPUT,
                Lineage.lowered("transformer-semantic-op", (input_id,)),
                "hidden_output",
            ),
        ),
        operations=(
            ModelOperation(
                operation_id,
                OperationName("transformer", "decoder_training"),
                (input_id,),
                (output_id,),
                Lineage.root("transformer-import"),
                attributes=FrozenDict({"model_spec": model}),
            ),
        ),
        inputs=(input_id,),
        outputs=(output_id,),
        attributes=FrozenDict({"model_family": "decoder-only-transformer"}),
    )


def synthesis_session_for(
    model: TransformerModelSpec,
    workload_spec: TransformerTrainingWorkloadSpec,
    mapping: TransformerTrainingMappingSpec,
) -> SynthesisSession:
    """Create the explicit session consumed by Transformer lowering passes."""

    mapping.validate_workload(workload_spec)
    workload = WorkloadBinding(
        WorkloadMode.TRAINING,
        batch_size=workload_spec.microbatch_size,
        sequence_length=model.sequence_length,
        micro_batches=mapping.microbatch_count(workload_spec),
        attributes=FrozenDict({"workload_spec": workload_spec}),
    )
    strategy = StrategyBinding(
        tensor_parallel=mapping.tensor_parallel,
        pipeline_parallel=mapping.pipeline_parallel,
        data_parallel=mapping.data_parallel,
        recompute_policy=mapping.recompute.value,
        pipeline_policy=f"1f1b-interleaved-{mapping.pipeline_interleaving}",
        attributes=FrozenDict({"mapping_spec": mapping}),
    )
    return SynthesisSession(
        bindings=BindingSet(workload=workload, strategy=strategy),
        features=frozenset({"transformer-training-analysis-v2"}),
    )

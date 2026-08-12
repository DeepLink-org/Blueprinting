"""Transformer training adapters for formal synthesis.

This module is the explicit dependency seam between target-neutral workload
contracts and Blueprinting's canonical representation/binding machinery.
"""

from __future__ import annotations

from blueprinting.mapping import TransformerTrainingMappingSpec
from blueprinting.schema.frozen import FrozenDict
from blueprinting.workload import (
    TransformerDataType,
    TransformerModelSpec,
    TransformerTrainingWorkloadSpec,
    require_transformer_data_type,
)

from ..axes import BindingAxis
from ..bindings import BindingSet, StrategyBinding, TrainingWorkload, WorkloadBinding
from ..dialects.transformer import (
    TransformerModelOperationSemantic,
    TransformerTrainingStrategySemantic,
    TransformerTrainingWorkloadSemantic,
)
from ..expr import Symbol
from ..ids import Lineage, NodeId, ValueId
from ..session import SynthesisSession
from ..stages.common import OperationName, TensorType
from ..stages.model.ir import ModelIR, ModelOperation, ModelValue, ValueRole


def build_transformer_model_ir(model: TransformerModelSpec, *, datatype: TransformerDataType = "float16") -> ModelIR:
    """Import a model as one semantic operation before structural lowering."""

    datatype = require_transformer_data_type(datatype)
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
                semantic=TransformerModelOperationSemantic(model),
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

    mapping.validate_model(model)
    mapping.validate_workload(workload_spec)
    workload = WorkloadBinding(
        TrainingWorkload(),
        semantic=TransformerTrainingWorkloadSemantic(workload_spec),
    )
    strategy = StrategyBinding(
        semantic=TransformerTrainingStrategySemantic(mapping),
    )
    return SynthesisSession(
        bindings=BindingSet(workload=workload, strategy=strategy),
        features=frozenset({"transformer-training-analysis"}),
    )

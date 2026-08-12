"""Registered typed semantic payloads for the Transformer dialect."""

from __future__ import annotations

from blueprinting.mapping import TransformerInferenceMappingSpec, TransformerTrainingMappingSpec
from blueprinting.schema.authoring import NonEmptyText, PositiveInt, record
from blueprinting.workload import TransformerDataType, TransformerModelSpec, TransformerTrainingWorkloadSpec

from ...bindings import InferencePhase
from ...semantics import (
    BindingSemantic,
    BufferSemantic,
    DistributedTaskSemantic,
    ModelOperationSemantic,
    PlanTaskSemantic,
    ProgramSemantic,
)
from ...stages.distributed.ir import CollectiveKind
from .common import EngineKind
from .inference import InferenceBlockMemoryFacts, InferenceInvocation
from .training import BlockMemoryFacts, PrimitiveInvocation, TrainingPhase


@record("blueprinting.ir.semantic.transformer.model-operation-semantic")
class TransformerModelOperationSemantic(ModelOperationSemantic):
    model: TransformerModelSpec


@record("blueprinting.ir.semantic.transformer.training-workload-binding-semantic")
class TransformerTrainingWorkloadSemantic(BindingSemantic):
    workload: TransformerTrainingWorkloadSpec


@record("blueprinting.ir.semantic.transformer.training-strategy-binding-semantic")
class TransformerTrainingStrategySemantic(BindingSemantic):
    mapping: TransformerTrainingMappingSpec


@record("blueprinting.ir.semantic.transformer.inference-workload-binding-semantic")
class TransformerInferenceWorkloadSemantic(BindingSemantic):
    batch_size: PositiveInt
    context_tokens: PositiveInt
    datatype: TransformerDataType


@record("blueprinting.ir.semantic.transformer.inference-strategy-binding-semantic")
class TransformerInferenceStrategySemantic(BindingSemantic):
    mapping: TransformerInferenceMappingSpec


@record("blueprinting.ir.semantic.transformer.training-distributed-task-semantic")
class TransformerTrainingDistributedTaskSemantic(DistributedTaskSemantic):
    invocation: PrimitiveInvocation


@record("blueprinting.ir.semantic.transformer.inference-distributed-task-semantic")
class TransformerInferenceDistributedTaskSemantic(DistributedTaskSemantic):
    invocation: InferenceInvocation


@record("blueprinting.ir.semantic.transformer.training-plan-task-semantic")
class TransformerTrainingPlanTaskSemantic(PlanTaskSemantic):
    name: str
    source_layer: str
    primitive: str
    phase: TrainingPhase
    engine: EngineKind
    collective: CollectiveKind | None = None


@record("blueprinting.ir.semantic.transformer.inference-plan-task-semantic")
class TransformerInferencePlanTaskSemantic(PlanTaskSemantic):
    name: str
    source_layer: str
    primitive: str
    phase: InferencePhase
    engine: EngineKind
    query_tokens: int
    context_tokens: int
    collective: CollectiveKind | None = None


@record("blueprinting.ir.semantic.transformer.training-program-semantic")
class TransformerTrainingProgramSemantic(ProgramSemantic):
    model: TransformerModelSpec
    workload: TransformerTrainingWorkloadSpec
    mapping: TransformerTrainingMappingSpec
    scope: str
    block_memory: BlockMemoryFacts


@record("blueprinting.ir.semantic.transformer.training-plan-semantic")
class TransformerTrainingPlanSemantic(TransformerTrainingProgramSemantic):
    pass


@record("blueprinting.ir.semantic.transformer.inference-program-semantic")
class TransformerInferenceProgramSemantic(ProgramSemantic):
    model: TransformerModelSpec
    mapping: TransformerInferenceMappingSpec
    phase: InferencePhase
    batch_size: int
    query_tokens: int
    context_tokens: int
    datatype: TransformerDataType
    scope: str
    block_memory: InferenceBlockMemoryFacts


@record("blueprinting.ir.semantic.transformer.inference-plan-semantic")
class TransformerInferencePlanSemantic(TransformerInferenceProgramSemantic):
    pass


@record("blueprinting.ir.semantic.transformer.buffer-semantic")
class TransformerBufferSemantic(BufferSemantic):
    role: NonEmptyText
    phase: InferencePhase | None = None
    bound: NonEmptyText | None = None


def training_task_semantic(invocation: PrimitiveInvocation) -> TransformerTrainingPlanTaskSemantic:
    return TransformerTrainingPlanTaskSemantic(
        invocation.name,
        invocation.source_layer,
        invocation.primitive,
        invocation.phase,
        invocation.engine,
        invocation.collective,
    )


def inference_task_semantic(
    invocation: InferenceInvocation,
    *,
    query_tokens: int,
    context_tokens: int,
) -> TransformerInferencePlanTaskSemantic:
    return TransformerInferencePlanTaskSemantic(
        invocation.name,
        invocation.source_layer,
        invocation.primitive,
        invocation.phase,
        invocation.engine,
        query_tokens,
        context_tokens,
        invocation.collective,
    )

"""Exact Transformer work facts and their target-neutral derivation."""

from .common import EngineKind, PhaseWork
from .inference import InferenceBlockMemoryFacts, InferenceInvocation, derive_transformer_inference_block
from .semantics import (
    TransformerBufferSemantic,
    TransformerInferenceDistributedTaskSemantic,
    TransformerInferencePlanSemantic,
    TransformerInferencePlanTaskSemantic,
    TransformerInferenceProgramSemantic,
    TransformerInferenceStrategySemantic,
    TransformerInferenceWorkloadSemantic,
    TransformerModelOperationSemantic,
    TransformerTrainingDistributedTaskSemantic,
    TransformerTrainingPlanSemantic,
    TransformerTrainingPlanTaskSemantic,
    TransformerTrainingProgramSemantic,
    TransformerTrainingStrategySemantic,
    TransformerTrainingWorkloadSemantic,
    inference_task_semantic,
    training_task_semantic,
)
from .training import (
    BlockMemoryFacts,
    PrimitiveInvocation,
    TrainingPhase,
    derive_transformer_block,
)

__all__ = [
    "BlockMemoryFacts",
    "EngineKind",
    "InferenceBlockMemoryFacts",
    "InferenceInvocation",
    "PhaseWork",
    "PrimitiveInvocation",
    "TrainingPhase",
    "TransformerBufferSemantic",
    "TransformerInferenceDistributedTaskSemantic",
    "TransformerInferenceProgramSemantic",
    "TransformerInferencePlanSemantic",
    "TransformerInferencePlanTaskSemantic",
    "TransformerInferenceStrategySemantic",
    "TransformerInferenceWorkloadSemantic",
    "TransformerModelOperationSemantic",
    "TransformerTrainingDistributedTaskSemantic",
    "TransformerTrainingProgramSemantic",
    "TransformerTrainingPlanSemantic",
    "TransformerTrainingPlanTaskSemantic",
    "TransformerTrainingStrategySemantic",
    "TransformerTrainingWorkloadSemantic",
    "derive_transformer_block",
    "derive_transformer_inference_block",
    "inference_task_semantic",
    "training_task_semantic",
]

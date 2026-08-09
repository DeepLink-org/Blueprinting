"""Exact Transformer work facts and their target-neutral derivation."""

from .common import EngineKind, PhaseWork
from .inference import InferenceBlockMemoryFacts, InferenceInvocation, derive_transformer_inference_block
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
    "derive_transformer_block",
    "derive_transformer_inference_block",
]

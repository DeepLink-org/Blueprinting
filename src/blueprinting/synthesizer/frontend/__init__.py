"""Adapters from workload contracts into canonical synthesis state."""

from .transformer import build_transformer_model_ir, synthesis_session_for
from .transformer_inference import (
    build_transformer_inference_model_ir,
    inference_synthesis_session_for,
)

__all__ = [
    "build_transformer_inference_model_ir",
    "build_transformer_model_ir",
    "inference_synthesis_session_for",
    "synthesis_session_for",
]

"""Target-neutral model and workload contracts."""

from .transformer import TransformerModelSpec, TransformerTrainingWorkloadSpec
from .transformer_inference import TransformerInferenceRequestSpec

__all__ = [
    "TransformerInferenceRequestSpec",
    "TransformerModelSpec",
    "TransformerTrainingWorkloadSpec",
]

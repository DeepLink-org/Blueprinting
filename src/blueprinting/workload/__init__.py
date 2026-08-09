"""Target-neutral workload and logical-mapping contracts."""

from .transformer import (
    RecomputePolicy,
    TensorParallelCommunication,
    TransformerExecutionSpec,
    TransformerModelSpec,
)
from .transformer_inference import (
    TransformerInferenceExecutionSpec,
    TransformerInferenceRequestSpec,
)

__all__ = [
    "RecomputePolicy",
    "TensorParallelCommunication",
    "TransformerExecutionSpec",
    "TransformerInferenceExecutionSpec",
    "TransformerInferenceRequestSpec",
    "TransformerModelSpec",
]

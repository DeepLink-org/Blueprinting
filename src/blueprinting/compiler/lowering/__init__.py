"""Production lowering passes for canonical IR dialects."""

from .transformer import DistributeTransformerTrainingPass, PlanTransformerTrainingPass

__all__ = ["DistributeTransformerTrainingPass", "PlanTransformerTrainingPass"]

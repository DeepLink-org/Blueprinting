"""Stable authoring surface for verified derivation passes.

Application code should execute passes through :class:`PassManager`.  This
module is for trusted pass and target-extension authors and deliberately omits
transaction-runner and registry implementation details.
"""

from .base import (
    AnalysisKey,
    AnalysisProduct,
    DerivationPass,
    MutationModel,
    PassContext,
    PassResult,
    PassRule,
    RelationCheckContext,
    RuleClaim,
    VerificationPolicy,
)
from .deriving import claim, derivation, equal_claim, relation

__all__ = [
    "AnalysisKey",
    "AnalysisProduct",
    "DerivationPass",
    "MutationModel",
    "PassContext",
    "PassResult",
    "PassRule",
    "RelationCheckContext",
    "RuleClaim",
    "VerificationPolicy",
    "claim",
    "derivation",
    "equal_claim",
    "relation",
]

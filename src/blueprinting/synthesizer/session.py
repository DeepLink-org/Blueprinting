"""Explicit immutable formal-synthesis session."""

from __future__ import annotations

from dataclasses import field, replace
from typing import Any

from blueprinting.schema.authoring import NonEmptyText, NonNegativeInt, record
from blueprinting.schema.codec import content_digest
from blueprinting.schema.frozen import FrozenDict

from .axes import BindingAxis
from .bindings import BindingSet, BindingValue, TargetRequirements
from .errors import BindingError


@record("blueprinting.synthesis.session")
class SynthesisSession:
    bindings: BindingSet = field(default_factory=BindingSet)
    target_requirements: TargetRequirements = field(default_factory=TargetRequirements)
    evidence_snapshot: NonEmptyText = "none"
    seed: NonNegativeInt = 0
    features: frozenset[NonEmptyText] = frozenset()
    options: FrozenDict = field(default_factory=FrozenDict)

    def with_binding(self, binding: BindingValue) -> SynthesisSession:
        return replace(self, bindings=self.bindings.with_binding(binding))

    def with_options(self, **changes: Any) -> SynthesisSession:
        return replace(self, options=self.options.evolve(**changes))

    def require(self, *axes: BindingAxis) -> None:
        """Validate binding presence and target/deployment constraints together."""

        self.bindings.require(*axes)
        if BindingAxis.TARGET in axes:
            target = self.bindings.target
            if target is not None and not target.satisfies(self.target_requirements):
                raise BindingError(f"target {target.name!r} does not satisfy synthesis requirements")
        if BindingAxis.DEPLOYMENT in axes:
            deployment = self.bindings.deployment
            if deployment is not None and not deployment.satisfies(self.target_requirements):
                raise BindingError(f"deployment {deployment.name!r} does not satisfy synthesis requirements")

    @property
    def fingerprint(self) -> str:
        # The digest domain is a stable wire identity retained across the
        # Python package and public class rename.
        return content_digest(self, "synthesis-session")

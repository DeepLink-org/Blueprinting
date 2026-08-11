"""Explicit immutable formal-synthesis session."""

from __future__ import annotations

from dataclasses import field, replace
from typing import Any

from blueprinting.schema.authoring import record
from blueprinting.schema.codec import content_digest
from blueprinting.schema.frozen import FrozenDict, freeze

from .axes import BindingAxis
from .bindings import BindingSet, BindingValue, TargetRequirements
from .errors import BindingError


@record("blueprinting.synthesis.session")
class SynthesisSession:
    bindings: BindingSet = field(default_factory=BindingSet)
    target_requirements: TargetRequirements = field(default_factory=TargetRequirements)
    evidence_snapshot: str = "none"
    seed: int = 0
    features: frozenset[str] = frozenset()
    options: FrozenDict = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        if not isinstance(self.bindings, BindingSet):
            raise TypeError("bindings must be a BindingSet")
        if not isinstance(self.target_requirements, TargetRequirements):
            raise TypeError("target_requirements must be TargetRequirements")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("synthesis seed must be a non-negative integer")
        if not isinstance(self.evidence_snapshot, str) or not self.evidence_snapshot:
            raise ValueError("evidence_snapshot must not be empty")
        features = frozenset(self.features)
        if any(not isinstance(feature, str) or not feature for feature in features):
            raise ValueError("synthesis features must be non-empty strings")
        object.__setattr__(self, "features", features)
        options = freeze(self.options)
        if not isinstance(options, FrozenDict):
            raise TypeError("session options must be a mapping")
        object.__setattr__(self, "options", options)

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

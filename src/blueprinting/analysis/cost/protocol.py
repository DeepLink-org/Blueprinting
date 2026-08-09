"""Normalized task-cost contracts and deterministic evidence resolution.

The protocol keeps workload facts, evidence selection, and the resulting
estimate separate.  Providers answer one immutable :class:`CostQuery`; the
resolver selects exactly one answer and records every attempted provider.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

from ...compiler.codec import content_digest, enum_type, record_type
from ...compiler.frozen import FrozenDict

# Keep the legacy codec namespace as a stable serialized identity.


class CostModelError(RuntimeError):
    """Base class for cost-model protocol failures."""


class CostNotAvailableError(CostModelError):
    """Raised when no provider can answer a query."""


class InvalidCostEvidenceError(CostModelError):
    """Raised when available evidence is ambiguous or internally invalid."""


@enum_type("compiler.analysis.cost.subject.v1")
class CostSubject(Enum):
    OPERATOR = "operator"
    COMMUNICATION = "communication"


@enum_type("compiler.analysis.cost.method.v1")
class EstimateMethod(Enum):
    MEASURED = "measured"
    SIMULATED = "simulated"
    ANALYTICAL = "analytical"
    CALIBRATED = "calibrated"
    VENDOR_MODEL = "vendor_model"


@enum_type("compiler.analysis.cost.match.v1")
class EstimateMatch(Enum):
    EXACT_SELECTOR = "exact_selector"
    INTERPOLATED = "interpolated"
    EXTRAPOLATED = "extrapolated"
    ANALYTICAL = "analytical"
    FALLBACK = "fallback"


@enum_type("compiler.analysis.cost.support_status.v1")
class SupportStatus(Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    INVALID = "invalid"


_RESERVED_DIMENSIONS = frozenset(
    {
        "subject",
        "operation",
        "hardware",
        "datatype",
        "operations",
        "read_bytes",
        "write_bytes",
        "message_bytes",
        "participants",
        "network_tier",
        "engine",
        "hardware_revision",
        "implementation",
        "implementation_revision",
        "runtime",
        "runtime_revision",
        "topology",
        "power_mode",
    }
)


def _required_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _optional_text(value: str, name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")


def _non_negative_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


@record_type("compiler.analysis.cost.query.v1")
@dataclass(frozen=True)
class CostQuery:
    """One fully identified task-cost question.

    ``dimensions`` carries shape or domain-specific selectors.  Fields that
    affect every provider (work, deployment, and implementation identity) are
    first-class so they cannot disappear from cache identity accidentally.
    Empty strings mean explicitly unknown, not a wildcard claim by a provider.
    """

    subject: CostSubject
    operation: str
    hardware: str
    datatype: str
    operations: int = 0
    read_bytes: int = 0
    write_bytes: int = 0
    message_bytes: int = 0
    participants: int = 1
    network_tier: int = 0
    engine: str = ""
    hardware_revision: str = ""
    implementation: str = ""
    implementation_revision: str = ""
    runtime: str = ""
    runtime_revision: str = ""
    topology: str = ""
    power_mode: str = ""
    dimensions: FrozenDict = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        if not isinstance(self.subject, CostSubject):
            raise TypeError("subject must be CostSubject")
        for name in ("operation", "hardware", "datatype"):
            _required_text(getattr(self, name), name)
        for name in (
            "engine",
            "hardware_revision",
            "implementation",
            "implementation_revision",
            "runtime",
            "runtime_revision",
            "topology",
            "power_mode",
        ):
            _optional_text(getattr(self, name), name)
        for name in ("operations", "read_bytes", "write_bytes", "message_bytes", "network_tier"):
            _non_negative_integer(getattr(self, name), name)
        if isinstance(self.participants, bool) or not isinstance(self.participants, int) or self.participants <= 0:
            raise ValueError("participants must be a positive integer")
        object.__setattr__(self, "dimensions", FrozenDict(self.dimensions))
        collisions = _RESERVED_DIMENSIONS.intersection(self.dimensions)
        if collisions:
            raise ValueError(f"dimensions use reserved names: {', '.join(sorted(collisions))}")

    @property
    def digest(self) -> str:
        return content_digest(self, "cost-query")

    @property
    def match_context(self) -> FrozenDict:
        """Return the flat context against which evidence selectors match."""

        context = self.dimensions.to_dict()
        context.update(
            {
                "operations": self.operations,
                "read_bytes": self.read_bytes,
                "write_bytes": self.write_bytes,
                "message_bytes": self.message_bytes,
                "participants": self.participants,
                "network_tier": self.network_tier,
            }
        )
        for name in (
            "engine",
            "hardware_revision",
            "implementation",
            "implementation_revision",
            "runtime",
            "runtime_revision",
            "topology",
            "power_mode",
        ):
            value = getattr(self, name)
            if value:
                context[name] = value
        return FrozenDict(context)


@record_type("compiler.analysis.cost.query_context.v1")
@dataclass(frozen=True)
class CostQueryContext:
    """Deployment/implementation facts supplied after portable planning.

    Maps are keyed by semantic operation name.  A generic operation key (for
    example ``gemm``) is used as a fallback when no semantic key is present.
    """

    runtime: str = ""
    runtime_revision: str = ""
    topology: str = ""
    power_mode: str = ""
    implementations: FrozenDict = field(default_factory=FrozenDict)
    implementation_revisions: FrozenDict = field(default_factory=FrozenDict)
    dimensions: FrozenDict = field(default_factory=FrozenDict)
    operation_dimensions: FrozenDict = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        for name in ("runtime", "runtime_revision", "topology", "power_mode"):
            _optional_text(getattr(self, name), name)
        if self.runtime:
            object.__setattr__(self, "runtime", self.runtime.strip().lower())
        for name in ("implementations", "implementation_revisions", "dimensions", "operation_dimensions"):
            object.__setattr__(self, name, FrozenDict(getattr(self, name)))
        for name, values in (
            ("implementations", self.implementations),
            ("implementation_revisions", self.implementation_revisions),
        ):
            if any(not isinstance(value, str) or not value for value in values.values()):
                raise ValueError(f"{name} values must be non-empty strings")
        for operation, dimensions in self.operation_dimensions.items():
            if not isinstance(dimensions, Mapping):
                raise TypeError(f"operation dimensions for {operation!r} must be a mapping")

    def implementation_for(self, semantic_operation: str, operation: str) -> str:
        return self.implementations.get(semantic_operation, self.implementations.get(operation, ""))

    def implementation_revision_for(self, semantic_operation: str, operation: str) -> str:
        return self.implementation_revisions.get(
            semantic_operation,
            self.implementation_revisions.get(operation, ""),
        )

    def dimensions_for(self, semantic_operation: str, operation: str) -> FrozenDict:
        result = self.dimensions.to_dict()
        specific = self.operation_dimensions.get(semantic_operation, self.operation_dimensions.get(operation, {}))
        overlap = set(result).intersection(specific)
        conflicts = tuple(key for key in overlap if result[key] != specific[key])
        if conflicts:
            raise ValueError(f"operation-specific dimensions conflict with global context: {sorted(conflicts)}")
        result.update(specific)
        return FrozenDict(result)


@record_type("compiler.analysis.cost.uncertainty.v1")
@dataclass(frozen=True)
class EstimateUncertainty:
    sample_count: int = 0
    standard_deviation_seconds: float | None = None
    lower_bound_seconds: float | None = None
    upper_bound_seconds: float | None = None
    confidence: float | None = None

    def __post_init__(self) -> None:
        _non_negative_integer(self.sample_count, "sample_count")
        for name in ("standard_deviation_seconds", "lower_bound_seconds", "upper_bound_seconds"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
            ):
                raise ValueError(f"{name} must be finite and non-negative when present")
        if self.confidence is not None and (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not math.isfinite(self.confidence)
            or not 0 <= self.confidence <= 1
        ):
            raise ValueError("confidence must be in [0, 1] when present")
        if (
            self.lower_bound_seconds is not None
            and self.upper_bound_seconds is not None
            and self.lower_bound_seconds > self.upper_bound_seconds
        ):
            raise ValueError("uncertainty lower bound cannot exceed upper bound")


@record_type("compiler.analysis.cost.estimate.v1")
@dataclass(frozen=True)
class CostEstimate:
    seconds: float
    provider: str
    provider_revision: str
    source_revision: str
    method: EstimateMethod
    match: EstimateMatch
    uncertainty: EstimateUncertainty = field(default_factory=EstimateUncertainty)
    raw_record_ids: tuple[str, ...] = ()
    validity_domain: FrozenDict = field(default_factory=FrozenDict)
    components: FrozenDict = field(default_factory=FrozenDict)
    assumptions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            isinstance(self.seconds, bool)
            or not isinstance(self.seconds, (int, float))
            or not math.isfinite(self.seconds)
            or self.seconds < 0
        ):
            raise ValueError("estimate seconds must be finite and non-negative")
        for name in ("provider", "provider_revision", "source_revision"):
            _required_text(getattr(self, name), name)
        if not isinstance(self.method, EstimateMethod):
            raise TypeError("method must be EstimateMethod")
        if not isinstance(self.match, EstimateMatch):
            raise TypeError("match must be EstimateMatch")
        if not isinstance(self.uncertainty, EstimateUncertainty):
            raise TypeError("uncertainty must be EstimateUncertainty")
        object.__setattr__(self, "raw_record_ids", tuple(self.raw_record_ids))
        object.__setattr__(self, "validity_domain", FrozenDict(self.validity_domain))
        object.__setattr__(self, "components", FrozenDict(self.components))
        object.__setattr__(self, "assumptions", tuple(self.assumptions))
        if any(not isinstance(item, str) or not item for item in self.raw_record_ids):
            raise ValueError("raw record IDs must be non-empty strings")
        if any(not isinstance(item, str) or not item for item in self.assumptions):
            raise ValueError("assumptions must be non-empty strings")


@dataclass(frozen=True)
class CostSupport:
    status: SupportStatus
    reason: str
    missing_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.status, SupportStatus):
            raise TypeError("status must be SupportStatus")
        _required_text(self.reason, "support reason")
        object.__setattr__(self, "missing_fields", tuple(self.missing_fields))
        if any(not isinstance(item, str) or not item for item in self.missing_fields):
            raise ValueError("missing fields must be non-empty strings")

    @classmethod
    def available(cls, reason: str = "query is covered") -> CostSupport:
        return cls(SupportStatus.AVAILABLE, reason)

    @classmethod
    def unavailable(cls, reason: str, *missing_fields: str) -> CostSupport:
        return cls(SupportStatus.UNAVAILABLE, reason, tuple(missing_fields))

    @classmethod
    def invalid(cls, reason: str) -> CostSupport:
        return cls(SupportStatus.INVALID, reason)


@dataclass(frozen=True)
class ProviderAttempt:
    provider: str
    revision: str
    support: CostSupport


@dataclass(frozen=True)
class CostResolution:
    query_digest: str
    estimate: CostEstimate
    attempts: tuple[ProviderAttempt, ...]
    resolver_revision: str


@runtime_checkable
class CostProvider(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def revision(self) -> str: ...

    def supports(self, query: CostQuery) -> CostSupport: ...

    def estimate(self, query: CostQuery) -> CostEstimate: ...


class CostResolver:
    """Ordered, deterministic provider selection without anonymous blending."""

    def __init__(self, providers: tuple[CostProvider, ...], *, policy_name: str = "ordered-first-supported-v1") -> None:
        self._providers = tuple(providers)
        _required_text(policy_name, "policy_name")
        identities = tuple((provider.name, provider.revision) for provider in self._providers)
        if len(set(identities)) != len(identities):
            raise ValueError("resolver providers must have unique name/revision identities")
        self._policy_name = policy_name
        self._revision = content_digest(
            FrozenDict({"policy": policy_name, "providers": identities}),
            "cost-resolver",
        )

    @property
    def providers(self) -> tuple[CostProvider, ...]:
        return self._providers

    @property
    def revision(self) -> str:
        return self._revision

    def try_resolve(self, query: CostQuery) -> CostResolution | None:
        if not isinstance(query, CostQuery):
            raise TypeError("query must be CostQuery")
        attempts: list[ProviderAttempt] = []
        for provider in self._providers:
            support = provider.supports(query)
            if not isinstance(support, CostSupport):
                raise TypeError(f"provider {provider.name!r} returned an invalid support result")
            attempts.append(ProviderAttempt(provider.name, provider.revision, support))
            if support.status is SupportStatus.INVALID:
                raise InvalidCostEvidenceError(
                    f"provider {provider.name!r} found invalid evidence for {query.digest}: {support.reason}"
                )
            if support.status is SupportStatus.UNAVAILABLE:
                continue
            estimate = provider.estimate(query)
            if not isinstance(estimate, CostEstimate):
                raise TypeError(f"provider {provider.name!r} returned an invalid estimate")
            if estimate.provider != provider.name or estimate.provider_revision != provider.revision:
                raise InvalidCostEvidenceError(f"provider {provider.name!r} returned inconsistent provenance identity")
            return CostResolution(query.digest, estimate, tuple(attempts), self.revision)
        return None

    def resolve(self, query: CostQuery) -> CostResolution:
        resolution = self.try_resolve(query)
        if resolution is None:
            attempted = ", ".join(provider.name for provider in self._providers) or "none"
            raise CostNotAvailableError(f"no cost provider covers query {query.digest}; attempted: {attempted}")
        return resolution

"""Normalized task-cost contracts and deterministic evidence resolution.

The protocol keeps workload facts, evidence selection, and the resulting
estimate separate.  Providers answer one immutable :class:`CostQuery`; the
resolver selects exactly one answer and records every attempted provider.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from blueprinting.schema.authoring import (
    CanonicalLowerText,
    NonBlankText,
    NonEmptyText,
    NonNegativeFiniteNumber,
    NonNegativeInt,
    PositiveInt,
    UnitIntervalNumber,
    adt,
    enum,
    is_adt_variant,
    record,
    seal_adt,
    variant,
)
from blueprinting.schema.codec import content_digest
from blueprinting.schema.diagnostics import Diagnostic, DiagnosticSet
from blueprinting.schema.frozen import FrozenDict
from blueprinting.schema.result import Checked, Err, Ok

# Keep the legacy codec namespace as a stable serialized identity.


class CostModelError(RuntimeError):
    """Base class for cost-model protocol failures."""


class CostNotAvailableError(CostModelError):
    """Raised when no provider can answer a query."""


class InvalidCostEvidenceError(CostModelError):
    """Raised when available evidence is ambiguous or internally invalid."""


@enum("blueprinting.analysis.cost.subject")
class CostSubject(Enum):
    OPERATOR = "operator"
    COMMUNICATION = "communication"


@enum("blueprinting.analysis.cost.method")
class EstimateMethod(Enum):
    MEASURED = "measured"
    SIMULATED = "simulated"
    ANALYTICAL = "analytical"
    CALIBRATED = "calibrated"
    VENDOR_MODEL = "vendor_model"


@enum("blueprinting.analysis.cost.match")
class EstimateMatch(Enum):
    EXACT_SELECTOR = "exact_selector"
    INTERPOLATED = "interpolated"
    EXTRAPOLATED = "extrapolated"
    ANALYTICAL = "analytical"
    FALLBACK = "fallback"


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


@record("blueprinting.analysis.cost.query")
class CostQuery:
    """One fully identified task-cost question.

    ``dimensions`` carries shape or domain-specific selectors.  Fields that
    affect every provider (work, deployment, and implementation identity) are
    first-class so they cannot disappear from cache identity accidentally.
    Empty strings mean explicitly unknown, not a wildcard claim by a provider.
    """

    subject: CostSubject
    operation: NonBlankText
    hardware: NonBlankText
    datatype: NonBlankText
    operations: NonNegativeInt = 0
    read_bytes: NonNegativeInt = 0
    write_bytes: NonNegativeInt = 0
    message_bytes: NonNegativeInt = 0
    participants: PositiveInt = 1
    network_tier: NonNegativeInt = 0
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


@record("blueprinting.analysis.cost.query-context")
class CostQueryContext:
    """Deployment/implementation facts supplied after portable planning.

    Maps are keyed by semantic operation name.  A generic operation key (for
    example ``gemm``) is used as a fallback when no semantic key is present.
    """

    runtime: CanonicalLowerText = ""
    runtime_revision: str = ""
    topology: str = ""
    power_mode: str = ""
    implementations: FrozenDict[NonEmptyText] = field(default_factory=FrozenDict)
    implementation_revisions: FrozenDict[NonEmptyText] = field(default_factory=FrozenDict)
    dimensions: FrozenDict = field(default_factory=FrozenDict)
    operation_dimensions: FrozenDict[Mapping[str, Any]] = field(default_factory=FrozenDict)

    def implementation_for(self, semantic_operation: str, operation: str) -> str:
        value = self.implementations.get(semantic_operation, self.implementations.get(operation, ""))
        if not isinstance(value, str):
            raise TypeError("implementation identity must be a string")
        return value

    def implementation_revision_for(self, semantic_operation: str, operation: str) -> str:
        value = self.implementation_revisions.get(
            semantic_operation,
            self.implementation_revisions.get(operation, ""),
        )
        if not isinstance(value, str):
            raise TypeError("implementation revision must be a string")
        return value

    def dimensions_for(self, semantic_operation: str, operation: str) -> FrozenDict:
        result = self.dimensions.to_dict()
        specific = self.operation_dimensions.get(semantic_operation, self.operation_dimensions.get(operation, {}))
        overlap = set(result).intersection(specific)
        conflicts = tuple(key for key in overlap if result[key] != specific[key])
        if conflicts:
            raise ValueError(f"operation-specific dimensions conflict with global context: {sorted(conflicts)}")
        result.update(specific)
        return FrozenDict(result)


@record("blueprinting.analysis.cost.uncertainty")
class EstimateUncertainty:
    sample_count: NonNegativeInt = 0
    standard_deviation_seconds: NonNegativeFiniteNumber | None = None
    lower_bound_seconds: NonNegativeFiniteNumber | None = None
    upper_bound_seconds: NonNegativeFiniteNumber | None = None
    confidence: UnitIntervalNumber | None = None

    def __post_init__(self) -> None:
        if (
            self.lower_bound_seconds is not None
            and self.upper_bound_seconds is not None
            and self.lower_bound_seconds > self.upper_bound_seconds
        ):
            raise ValueError("uncertainty lower bound cannot exceed upper bound")


@record("blueprinting.analysis.cost.estimate")
class CostEstimate:
    seconds: NonNegativeFiniteNumber
    provider: NonBlankText
    provider_revision: NonBlankText
    source_revision: NonBlankText
    method: EstimateMethod
    match: EstimateMatch
    uncertainty: EstimateUncertainty = field(default_factory=EstimateUncertainty)
    raw_record_ids: tuple[NonEmptyText, ...] = ()
    validity_domain: FrozenDict = field(default_factory=FrozenDict)
    components: FrozenDict = field(default_factory=FrozenDict)
    assumptions: tuple[NonEmptyText, ...] = ()


@adt(wire="blueprinting.analysis.cost.support")
class CostSupport:
    """Closed provider coverage result with no status/payload mismatch."""

    @classmethod
    def available(cls, reason: str = "query is covered") -> CostSupportVariant:
        return CostAvailable(reason)

    @classmethod
    def unavailable(cls, reason: str, *missing_fields: str) -> CostSupportVariant:
        return CostUnavailable(reason, tuple(missing_fields))

    @classmethod
    def invalid(cls, reason: str) -> CostSupportVariant:
        return InvalidCostSupport(reason)


@variant("available")
class CostAvailable(CostSupport):
    reason: NonBlankText


@variant("unavailable")
class CostUnavailable(CostSupport):
    reason: NonBlankText
    missing_fields: tuple[NonEmptyText, ...] = ()


@variant("invalid")
class InvalidCostSupport(CostSupport):
    reason: NonBlankText


CostSupportVariant = CostAvailable | CostUnavailable | InvalidCostSupport
seal_adt(CostSupport, CostSupportVariant)


@dataclass(frozen=True)
class ProviderAttempt:
    provider: str
    revision: str
    support: CostSupportVariant


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

    def supports(self, query: CostQuery) -> CostSupportVariant: ...

    def estimate(self, query: CostQuery) -> CostEstimate: ...


class CostResolver:
    """Ordered, deterministic provider selection without anonymous blending."""

    def __init__(self, providers: tuple[CostProvider, ...], *, policy_name: str = "ordered-first-supported-v0") -> None:
        self._providers = tuple(providers)
        if not isinstance(policy_name, str) or not policy_name.strip():
            raise ValueError("policy_name must be a non-empty string")
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

    def resolve(self, query: CostQuery) -> Checked[CostResolution]:
        """Resolve expected availability and evidence failures as diagnostics."""

        if not isinstance(query, CostQuery):
            raise TypeError("query must be CostQuery")
        attempts: list[ProviderAttempt] = []
        for provider in self._providers:
            support = provider.supports(query)
            if not is_adt_variant(support, CostSupport):
                raise TypeError(f"provider {provider.name!r} returned an invalid support result")
            attempts.append(ProviderAttempt(provider.name, provider.revision, support))
            match support:
                case InvalidCostSupport(reason):
                    return Err(
                        DiagnosticSet.of(
                            Diagnostic(
                                "cost.invalid_evidence",
                                f"provider {provider.name!r} found invalid evidence: {reason}",
                                ("provider", provider.name),
                            )
                        )
                    )
                case CostUnavailable():
                    continue
                case CostAvailable():
                    pass
            try:
                estimate = provider.estimate(query)
            except InvalidCostEvidenceError as error:
                return Err(
                    DiagnosticSet.of(
                        Diagnostic(
                            "cost.invalid_evidence",
                            f"provider {provider.name!r} rejected its selected evidence: {error}",
                            ("provider", provider.name),
                        )
                    )
                )
            if not isinstance(estimate, CostEstimate):
                raise TypeError(f"provider {provider.name!r} returned an invalid estimate")
            if estimate.provider != provider.name or estimate.provider_revision != provider.revision:
                return Err(
                    DiagnosticSet.of(
                        Diagnostic(
                            "cost.inconsistent_provenance",
                            f"provider {provider.name!r} returned inconsistent provenance identity",
                            ("provider", provider.name),
                        )
                    )
                )
            return Ok(CostResolution(query.digest, estimate, tuple(attempts), self.revision))
        attempted = ", ".join(provider.name for provider in self._providers) or "none"
        return Err(
            DiagnosticSet.of(
                Diagnostic(
                    "cost.unavailable",
                    f"no cost provider covers query {query.digest}; attempted: {attempted}",
                    ("query", query.digest),
                )
            )
        )

    def require(self, query: CostQuery) -> CostResolution:
        """Explicit exception adapter for application boundaries."""

        def exception(diagnostics: DiagnosticSet) -> CostModelError:
            rendered = "; ".join(item.render() for item in diagnostics.errors)
            if any(
                item.code.startswith("cost.invalid") or item.code == "cost.inconsistent_provenance"
                for item in diagnostics.errors
            ):
                return InvalidCostEvidenceError(rendered)
            return CostNotAvailableError(rendered)

        return self.resolve(query).or_raise(exception)

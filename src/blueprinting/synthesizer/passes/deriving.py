"""Low-noise deriving syntax for typed pass and lineage-rule contracts.

The decorators only derive metadata already present in Python annotations and
generic bases. They never wrap execution or hide pass semantics.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, TypeVar, cast, get_args, get_origin

from ..axes import BindingAxis
from ..stages.common import CanonicalIRMixin
from .base import (
    AnalysisKey,
    DerivationPass,
    MutationModel,
    PassContract,
    PassNormalizer,
    PassRule,
    RelationCheckContext,
    RuleClaim,
    RuleVerifier,
    VerificationPolicy,
)

PassType = TypeVar("PassType", bound=type[DerivationPass[Any, Any]])

_PASS_CONTRACTS: dict[type[DerivationPass[Any, Any]], PassContract] = {}


def claim(name: str, verifier: Callable[[Any, Any, RelationCheckContext], None]) -> RuleClaim:
    """Declare one preservation property and the predicate that proves it."""

    return RuleClaim(name, cast(RuleVerifier, verifier))


def equal_claim(
    name: str,
    source: Callable[[Any], Any],
    target: Callable[[Any], Any],
) -> RuleClaim:
    """Declare equality between source and target semantic projections."""

    def verify(source_value: Any, target_value: Any, _context: RelationCheckContext) -> None:
        expected = source(source_value)
        actual = target(target_value)
        if actual != expected:
            raise ValueError(f"claim {name!r} failed: {actual!r} != {expected!r}")

    return RuleClaim(name, cast(RuleVerifier, verify))


def relation(
    transform: str,
    rewrite: str,
    *,
    source: type[Any] | str,
    target: type[Any] | str,
    verifier: Callable[[Any, Any, RelationCheckContext], None],
    preserves: Iterable[RuleClaim] = (),
    introduces: Iterable[str] = (),
    forbids: Iterable[str] = (),
) -> PassRule:
    """Declare a lineage shape plus an independent executable relation invariant."""

    source_name = source if isinstance(source, str) else source.__name__
    target_name = target if isinstance(target, str) else target.__name__
    return PassRule(
        transform,
        source_name,
        target_name,
        rewrite,
        preserves=tuple(preserves),
        introduces=tuple(introduces),
        forbids=tuple(forbids),
        verifier=cast(RuleVerifier, verifier),
    )


def _pass_types(pass_type: type[DerivationPass[Any, Any]]) -> tuple[type[CanonicalIRMixin], type[CanonicalIRMixin]]:
    for base in getattr(pass_type, "__orig_bases__", ()):
        if get_origin(base) is not DerivationPass:
            continue
        source, target = get_args(base)
        if (
            isinstance(source, type)
            and isinstance(target, type)
            and issubclass(source, CanonicalIRMixin)
            and issubclass(target, CanonicalIRMixin)
        ):
            return source, target
    raise TypeError(f"{pass_type.__name__} must directly specialize DerivationPass[SourceIR, TargetIR]")


def derivation(
    name: str,
    *,
    revision: str,
    bindings: Iterable[BindingAxis] = (),
    requires: Iterable[AnalysisKey] = (),
    preserves: Iterable[AnalysisKey] = (),
    produces: Iterable[AnalysisKey] = (),
    rules: Iterable[PassRule] = (),
    mutation: MutationModel = MutationModel.IMMUTABLE,
    verification: VerificationPolicy = VerificationPolicy.BOTH,
    deterministic: bool = True,
    uses_session_seed: bool = False,
    normalizer: PassNormalizer | None = None,
) -> Callable[[PassType], PassType]:
    """Attach a complete exact-schema contract to a typed pass class."""

    def decorate(pass_type: PassType) -> PassType:
        source, target = _pass_types(pass_type)
        pass_type.contract = PassContract.create(
            name,
            source,
            target,
            revision=revision,
            required_bindings=frozenset(bindings),
            required_analyses=frozenset(requires),
            preserved_analyses=frozenset(preserves),
            produced_analyses=frozenset(produces),
            mutation_model=mutation,
            verification=verification,
            deterministic=deterministic,
            uses_session_seed=uses_session_seed,
            rules=tuple(rules),
            normalizer=normalizer,
        )
        previous = _PASS_CONTRACTS.get(pass_type)
        if previous is not None and previous != pass_type.contract:
            raise RuntimeError(f"pass class {pass_type.__name__} is already registered with another contract")
        _PASS_CONTRACTS[pass_type] = pass_type.contract
        return pass_type

    return decorate


def pass_contract_manifest() -> tuple[tuple[type[DerivationPass[Any, Any]], PassContract], ...]:
    """Return all decorator-declared pass contracts in deterministic order."""

    return tuple(
        sorted(
            _PASS_CONTRACTS.items(),
            key=lambda item: (item[1].name, item[1].revision, item[0].__module__, item[0].__qualname__),
        )
    )


__all__ = ["claim", "derivation", "equal_claim", "pass_contract_manifest", "relation"]

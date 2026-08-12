"""Declarative, immutable derivation-pass infrastructure.

The pass manager treats lowering as a sequence of typed snapshot transitions.
It validates schemas, bindings, analyses, mutation behavior, verification
policy, and provenance around every pass.  Analysis publication is committed
only after the produced IR has passed its contract, avoiding half-written
caches after a failed lowering.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, Protocol, TypeVar

from blueprinting.schema.codec import content_digest
from blueprinting.schema.diagnostics import Diagnostic, DiagnosticSet
from blueprinting.schema.errors import SerializationError
from blueprinting.schema.frozen import freeze
from blueprinting.schema.result import Checked, Err, Ok

from ..axes import BindingAxis
from ..errors import (
    BindingError,
    IRVerificationError,
    MissingAnalysisError,
    MissingBindingError,
    PassContractError,
    PassExecutionError,
    SynthesisError,
)
from ..ids import Lineage, LineageKind, StableId
from ..session import SynthesisSession
from ..stages.common import CanonicalIRMixin, SchemaVersion

InputIR = TypeVar("InputIR", bound=CanonicalIRMixin)
OutputIR = TypeVar("OutputIR", bound=CanonicalIRMixin)


def _diagnostic_code(error: SynthesisError) -> str:
    if isinstance(error, MissingBindingError):
        return "pass.missing_binding"
    if isinstance(error, BindingError):
        return "binding.invalid"
    if isinstance(error, MissingAnalysisError):
        return "pass.missing_analysis"
    if isinstance(error, PassContractError):
        return "pass.contract"
    if isinstance(error, IRVerificationError):
        return "ir.verification"
    return "synthesis.failure"


class RuleVerifier(Protocol):
    """Executable semantic predicate for one declared lineage rule."""

    def __call__(self, source: Any, target: Any, context: RelationCheckContext) -> None: ...


PassNormalizer = Callable[[Any, SynthesisSession], CanonicalIRMixin]


@dataclass(frozen=True)
class RelationCheckContext:
    """Read-only whole-boundary context available to executable rule claims."""

    source_ir: CanonicalIRMixin
    target_ir: CanonicalIRMixin
    source_to_targets: Mapping[StableId, tuple[StableId, ...]]
    target_to_sources: Mapping[StableId, tuple[StableId, ...]]
    target_entity_kinds: Mapping[StableId, str]
    session: SynthesisSession

    def targets_for(self, source_id: StableId) -> tuple[StableId, ...]:
        return self.source_to_targets.get(source_id, ())

    def only_target_for(self, source_id: StableId, target_entity: str | None = None) -> StableId:
        targets = self.targets_for(source_id)
        if target_entity is not None:
            targets = tuple(item for item in targets if self.target_entity_kinds.get(item) == target_entity)
        if len(targets) != 1:
            suffix = f" of kind {target_entity}" if target_entity is not None else ""
            raise ValueError(f"source {source_id} maps to {len(targets)} targets{suffix}, expected exactly one")
        return targets[0]


@dataclass(frozen=True)
class RuleClaim:
    """One named preservation property backed by an executable predicate."""

    name: str
    verifier: RuleVerifier = field(compare=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("rule claim name must be a non-empty string")
        if not callable(self.verifier):
            raise TypeError("rule claim verifier must be callable")


@dataclass(frozen=True, order=True)
class SchemaRange:
    """Inclusive schema compatibility interval."""

    minimum: SchemaVersion
    maximum: SchemaVersion

    def __post_init__(self) -> None:
        if self.maximum < self.minimum:
            raise ValueError("schema range maximum must not precede minimum")

    @classmethod
    def exact(cls, version: SchemaVersion) -> SchemaRange:
        return cls(version, version)

    def accepts(self, version: SchemaVersion) -> bool:
        return self.minimum <= version <= self.maximum


@dataclass(frozen=True, order=True)
class AnalysisKey:
    """Identity of one derived analysis kind in the current schema epoch."""

    namespace: str
    name: str

    def __post_init__(self) -> None:
        if not self.namespace or not self.name:
            raise ValueError("analysis namespace and name must not be empty")

    def __str__(self) -> str:
        return f"{self.namespace}.{self.name}"


@dataclass(frozen=True, order=True)
class AnalysisAddress:
    ir_digest: str
    key: AnalysisKey
    context_fingerprint: str


@dataclass(frozen=True)
class AnalysisEntry:
    address: AnalysisAddress
    value: Any
    value_digest: str
    producer: str


@dataclass(frozen=True)
class AnalysisProduct:
    key: AnalysisKey
    value: Any


class AnalysisStore:
    """Thread-safe, content-addressed cache for immutable derived views."""

    def __init__(self) -> None:
        self._entries: dict[AnalysisAddress, AnalysisEntry] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _address(ir_digest: str, key: AnalysisKey, context_fingerprint: str) -> AnalysisAddress:
        return AnalysisAddress(ir_digest=ir_digest, key=key, context_fingerprint=context_fingerprint)

    @staticmethod
    def _entry(address: AnalysisAddress, value: Any, producer: str) -> AnalysisEntry:
        frozen_value = freeze(value)
        try:
            digest = content_digest(frozen_value, f"analysis:{address.key}")
        except SerializationError as error:
            raise PassContractError(f"analysis {address.key} is not a canonical immutable value: {error}") from error
        return AnalysisEntry(address=address, value=frozen_value, value_digest=digest, producer=producer)

    def has(self, ir_digest: str, key: AnalysisKey, context_fingerprint: str) -> bool:
        address = self._address(ir_digest, key, context_fingerprint)
        with self._lock:
            return address in self._entries

    def get(self, ir_digest: str, key: AnalysisKey, context_fingerprint: str) -> Any:
        address = self._address(ir_digest, key, context_fingerprint)
        with self._lock:
            try:
                return self._entries[address].value
            except KeyError as error:
                raise MissingAnalysisError(f"analysis {key} is unavailable for IR {ir_digest}") from error

    def put(
        self,
        ir_digest: str,
        key: AnalysisKey,
        context_fingerprint: str,
        value: Any,
        *,
        producer: str,
    ) -> AnalysisEntry:
        address = self._address(ir_digest, key, context_fingerprint)
        entry = self._entry(address, value, producer)
        with self._lock:
            self._merge_entry(entry)
        return entry

    def _merge_entry(self, entry: AnalysisEntry) -> None:
        existing = self._entries.get(entry.address)
        if existing is not None and existing.value_digest != entry.value_digest:
            raise PassContractError(
                f"analysis cache conflict at {entry.address.key}: {existing.value_digest} != {entry.value_digest}"
            )
        self._entries[entry.address] = entry

    def commit_transition(
        self,
        *,
        source_digest: str,
        target_digest: str,
        context_fingerprint: str,
        preserved: frozenset[AnalysisKey],
        products: tuple[AnalysisProduct, ...],
        producer: str,
    ) -> None:
        """Atomically carry preserved analyses and publish new products."""

        pending = []
        with self._lock:
            for key in preserved:
                source = self._address(source_digest, key, context_fingerprint)
                existing = self._entries.get(source)
                if existing is not None:
                    target = self._address(target_digest, key, context_fingerprint)
                    pending.append(
                        AnalysisEntry(
                            address=target,
                            value=existing.value,
                            value_digest=existing.value_digest,
                            producer=producer,
                        )
                    )
            for product in products:
                address = self._address(target_digest, product.key, context_fingerprint)
                pending.append(self._entry(address, product.value, producer))
            for entry in pending:
                existing = self._entries.get(entry.address)
                if existing is not None and existing.value_digest != entry.value_digest:
                    raise PassContractError(f"analysis cache conflict at {entry.address.key}")
            for entry in pending:
                self._entries[entry.address] = entry

    def entries_for(self, ir_digest: str, context_fingerprint: str) -> tuple[AnalysisEntry, ...]:
        with self._lock:
            entries = tuple(
                item
                for address, item in self._entries.items()
                if address.ir_digest == ir_digest and address.context_fingerprint == context_fingerprint
            )
        return tuple(sorted(entries, key=lambda item: item.address.key))

    def clone(self) -> AnalysisStore:
        """Return an isolated snapshot for deterministic replay."""

        clone = AnalysisStore()
        with self._lock:
            clone._entries = dict(self._entries)
        return clone


class MutationModel(Enum):
    IMMUTABLE = "immutable"
    TRANSACTIONAL = "transactional"


class VerificationPolicy(Enum):
    BOTH = "both"
    INPUT_ONLY = "input_only"
    OUTPUT_ONLY = "output_only"
    NONE = "none"

    @property
    def verifies_input(self) -> bool:
        return self in {VerificationPolicy.BOTH, VerificationPolicy.INPUT_ONLY}

    @property
    def verifies_output(self) -> bool:
        return self in {VerificationPolicy.BOTH, VerificationPolicy.OUTPUT_ONLY}


class DeterminismPolicy(Enum):
    OFF = "off"
    VERIFY = "verify"


@dataclass(frozen=True, order=True)
class PassRule:
    """Declarative semantic contract for one named lineage transform."""

    transform: str
    source_entity: str
    target_entity: str
    rewrite: str
    preserves: tuple[RuleClaim, ...] = ()
    introduces: tuple[str, ...] = ()
    forbids: tuple[str, ...] = ()
    verifier: RuleVerifier | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        identity = (self.transform, self.source_entity, self.target_entity, self.rewrite)
        if any(not isinstance(item, str) or not item for item in identity):
            raise ValueError("pass rule identity and rewrite fields must be non-empty strings")
        claims = tuple(self.preserves)
        if any(not isinstance(item, RuleClaim) for item in claims):
            raise TypeError("pass rule preserves must contain executable RuleClaim values")
        if len({item.name for item in claims}) != len(claims):
            raise ValueError("pass rule preservation claim names must be unique")
        object.__setattr__(self, "preserves", claims)
        for field_name in ("introduces", "forbids"):
            values = tuple(getattr(self, field_name))
            if any(not isinstance(item, str) or not item for item in values):
                raise ValueError(f"pass rule {field_name} must contain non-empty strings")
            object.__setattr__(self, field_name, values)
        if self.verifier is not None and not callable(self.verifier):
            raise TypeError("pass rule verifier must be callable")

    @property
    def preservation_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.preserves)


@dataclass(frozen=True, order=True)
class ClaimEvidence:
    name: str
    verifier: str


@dataclass(frozen=True, order=True)
class TransitionRelation:
    rule_id: str
    transform: str
    source_entity: str
    target_entity: str
    source_ids: tuple[str, ...]
    target_id: str
    lineage_kind: LineageKind
    evidence: tuple[ClaimEvidence, ...] = ()


class TransitionVerificationStatus(Enum):
    STRUCTURAL_ONLY = "structural_only"
    CANONICAL_CONFORMANT = "canonical_conformant"
    RELATION_VERIFIED = "relation_verified"


@dataclass(frozen=True)
class TransitionReport:
    source_digest: str
    target_digest: str
    relations: tuple[TransitionRelation, ...] = ()
    status: TransitionVerificationStatus = TransitionVerificationStatus.STRUCTURAL_ONLY
    canonical_conformance: ClaimEvidence | None = None

    @property
    def verified_relations(self) -> int:
        return sum(bool(item.evidence) for item in self.relations)

    @property
    def verified_claims(self) -> int:
        return sum(len(item.evidence) for item in self.relations)

    @property
    def canonical_conformant(self) -> bool:
        return self.canonical_conformance is not None


@dataclass(frozen=True)
class _EntityDescriptor:
    kind: str
    identifier: StableId
    lineage: Lineage
    value: Any


def _lineage_entities(ir: CanonicalIRMixin) -> tuple[_EntityDescriptor, ...]:
    """Expose canonical entity identity without importing application views."""

    from ..stages.concrete_plan.ir import ConcretePlanIR
    from ..stages.distributed.ir import DistributedTaskIR
    from ..stages.machine.ir import MachineIR
    from ..stages.model.ir import ModelIR
    from ..stages.portable_plan.ir import PortablePlanIR

    entities: tuple[Any, ...]
    if isinstance(ir, ModelIR):
        entities = tuple(ir.values) + tuple(ir.operations)
    elif isinstance(ir, DistributedTaskIR):
        entities = tuple(ir.values) + tuple(ir.tasks)
    elif isinstance(ir, PortablePlanIR):
        entities = tuple(ir.buffers) + tuple(ir.tasks)
    elif isinstance(ir, ConcretePlanIR):
        entities = tuple(ir.buffers) + tuple(ir.commands)
    elif isinstance(ir, MachineIR):
        entities = ir.instructions
    else:
        return ()
    return tuple(_EntityDescriptor(type(item).__name__, item.id, item.lineage, item) for item in entities)


class TransitionVerifier:
    """Verify typed entity lineage and executable derivation laws."""

    @staticmethod
    def verify(
        source: CanonicalIRMixin,
        target: CanonicalIRMixin,
        contract: PassContract,
        session: SynthesisSession | None = None,
    ) -> TransitionReport:
        if contract.normalizer is not None and session is None:
            raise PassContractError(
                f"pass {contract.name!r} requires its synthesis session to evaluate the declared normal form"
            )
        relation_session = session if session is not None else SynthesisSession()

        def verify_normal_form() -> ClaimEvidence | None:
            if contract.normalizer is None:
                return None
            try:
                expected = contract.normalizer(source, relation_session)
            except Exception as error:
                raise PassContractError(
                    f"pass {contract.name!r} could not evaluate its declared normal form: {error}"
                ) from error
            if expected != target:
                raise PassContractError(
                    f"pass {contract.name!r} output differs from its declared canonical normal form"
                )
            return ClaimEvidence(
                "canonical normal form",
                contract._callable_identity(contract.normalizer),
            )

        source_entities = {item.identifier: item for item in _lineage_entities(source)}
        target_entities = _lineage_entities(target)
        if not contract.rules:
            if type(source) is not type(target):
                raise PassContractError(f"cross-stage pass {contract.name!r} must declare executable lineage rules")
            normal_form_evidence = verify_normal_form()
            if source.digest == target.digest:
                status = (
                    TransitionVerificationStatus.CANONICAL_CONFORMANT
                    if normal_form_evidence is not None
                    else TransitionVerificationStatus.STRUCTURAL_ONLY
                )
            elif normal_form_evidence is not None:
                status = TransitionVerificationStatus.CANONICAL_CONFORMANT
            else:
                raise PassContractError(
                    f"same-stage pass {contract.name!r} changed canonical IR without a declared normal form or rules"
                )
            return TransitionReport(
                source.digest,
                target.digest,
                status=status,
                canonical_conformance=normal_form_evidence,
            )
        rules = {item.transform: item for item in contract.rules}
        pending: list[tuple[_EntityDescriptor, tuple[_EntityDescriptor, ...], PassRule, str]] = []
        for target_entity in target_entities:
            lineage = target_entity.lineage
            if lineage.kind is LineageKind.ROOT:
                raise PassContractError(
                    f"pass {contract.name!r} produced root lineage for {target_entity.kind} {target_entity.identifier}"
                )
            rule = rules.get(lineage.transform)
            if rule is None:
                raise PassContractError(
                    f"pass {contract.name!r} produced undeclared lineage transform {lineage.transform!r}"
                )
            resolved = []
            for source_id in lineage.sources:
                entity = source_entities.get(source_id)
                if entity is None:
                    raise PassContractError(
                        f"pass {contract.name!r} lineage source {source_id} does not exist in its input snapshot"
                    )
                resolved.append(entity)
            if lineage.kind is LineageKind.GENERATED:
                if resolved or rule.source_entity not in {"none", "generated"}:
                    raise PassContractError("generated lineage requires zero sources and an explicit generated rule")
                actual_source_kind = "none"
            else:
                if not resolved:
                    raise PassContractError(f"{lineage.kind.value} lineage requires at least one source")
                if lineage.kind in {LineageKind.PRESERVED, LineageKind.CLONED} and len(resolved) != 1:
                    raise PassContractError(f"{lineage.kind.value} lineage requires exactly one source")
                if lineage.kind is LineageKind.FUSED and len(resolved) < 2:
                    raise PassContractError("fused lineage requires at least two sources")
                source_kinds = {item.kind for item in resolved}
                if len(source_kinds) != 1:
                    raise PassContractError("one lineage relation cannot mix source entity kinds")
                actual_source_kind = next(iter(source_kinds))
                if len(resolved) > 1:
                    actual_source_kind += " set"
            if actual_source_kind != rule.source_entity or target_entity.kind != rule.target_entity:
                raise PassContractError(
                    f"pass {contract.name!r} transform {rule.transform!r} expected "
                    f"{rule.source_entity} -> {rule.target_entity}, got "
                    f"{actual_source_kind} -> {target_entity.kind}"
                )
            pending.append((target_entity, tuple(resolved), rule, actual_source_kind))

        source_to_targets: dict[StableId, list[StableId]] = {}
        target_to_sources: dict[StableId, tuple[StableId, ...]] = {}
        for target_entity, resolved_entities, _rule, _kind in pending:
            target_to_sources[target_entity.identifier] = tuple(item.identifier for item in resolved_entities)
            for item in resolved_entities:
                source_to_targets.setdefault(item.identifier, []).append(target_entity.identifier)
        context = RelationCheckContext(
            source,
            target,
            {key: tuple(value) for key, value in source_to_targets.items()},
            target_to_sources,
            {item.identifier: item.kind for item in target_entities},
            relation_session,
        )
        normal_form_evidence = verify_normal_form()
        relations = []
        for target_entity, resolved_entities, rule, actual_source_kind in pending:
            source_value: Any = tuple(item.value for item in resolved_entities)
            if len(source_value) == 1:
                source_value = source_value[0]
            try:
                evidence = []
                if rule.verifier is not None:
                    rule.verifier(source_value, target_entity.value, context)
                    evidence.append(ClaimEvidence("relation invariant", contract._callable_identity(rule.verifier)))
                for claim in rule.preserves:
                    claim.verifier(source_value, target_entity.value, context)
                    evidence.append(ClaimEvidence(claim.name, contract._callable_identity(claim.verifier)))
            except PassContractError:
                raise
            except Exception as error:
                raise PassContractError(
                    f"pass {contract.name!r} rule {rule.transform!r} rejected its relation: {error}"
                ) from error
            relations.append(
                TransitionRelation(
                    rule_id=f"{contract.name}.{rule.transform}",
                    transform=rule.transform,
                    source_entity=actual_source_kind,
                    target_entity=target_entity.kind,
                    source_ids=tuple(str(item.identifier) for item in resolved_entities),
                    target_id=str(target_entity.identifier),
                    lineage_kind=target_entity.lineage.kind,
                    evidence=tuple(evidence),
                )
            )
        has_complete_evidence = bool(relations) and all(item.evidence for item in relations)
        if not has_complete_evidence and source.digest != target.digest:
            raise PassContractError(
                f"pass {contract.name!r} changed canonical IR without complete executable relation evidence"
            )
        status = (
            TransitionVerificationStatus.RELATION_VERIFIED
            if has_complete_evidence
            else TransitionVerificationStatus.STRUCTURAL_ONLY
        )
        if type(source) is not type(target) and status is not TransitionVerificationStatus.RELATION_VERIFIED:
            raise PassContractError(
                f"cross-stage pass {contract.name!r} did not produce executable claim evidence for every relation"
            )
        return TransitionReport(
            source.digest,
            target.digest,
            tuple(relations),
            status,
            normal_form_evidence,
        )


@dataclass(frozen=True)
class PassContract:
    """Complete static contract for one canonical IR transition."""

    name: str
    revision: str
    input_type: type[CanonicalIRMixin]
    input_schema: SchemaRange
    output_type: type[CanonicalIRMixin]
    output_schema: SchemaVersion
    required_bindings: frozenset[BindingAxis] = frozenset()
    required_analyses: frozenset[AnalysisKey] = frozenset()
    preserved_analyses: frozenset[AnalysisKey] = frozenset()
    produced_analyses: frozenset[AnalysisKey] = frozenset()
    mutation_model: MutationModel = MutationModel.IMMUTABLE
    verification: VerificationPolicy = VerificationPolicy.BOTH
    deterministic: bool = True
    uses_session_seed: bool = False
    rules: tuple[PassRule, ...] = ()
    normalizer: PassNormalizer | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("pass name must not be empty")
        if not isinstance(self.revision, str) or not self.revision:
            raise ValueError("pass revision must be a non-empty string")
        if not issubclass(self.input_type, CanonicalIRMixin) or not issubclass(self.output_type, CanonicalIRMixin):
            raise TypeError("pass input and output types must be canonical IR roots")
        if not self.input_schema.accepts(self.input_type.SCHEMA_VERSION):
            raise ValueError("pass input schema excludes the declared input IR type")
        if self.output_schema != self.output_type.SCHEMA_VERSION:
            raise ValueError("pass output schema must equal the declared output IR schema")
        object.__setattr__(self, "required_bindings", frozenset(self.required_bindings))
        object.__setattr__(self, "required_analyses", frozenset(self.required_analyses))
        object.__setattr__(self, "preserved_analyses", frozenset(self.preserved_analyses))
        object.__setattr__(self, "produced_analyses", frozenset(self.produced_analyses))
        rules = tuple(self.rules)
        if any(not isinstance(item, PassRule) for item in rules):
            raise TypeError("pass rules must contain only PassRule values")
        object.__setattr__(self, "rules", rules)
        if len({item.transform for item in self.rules}) != len(self.rules):
            raise ValueError("pass rule transforms must be unique")
        if self.uses_session_seed and not self.deterministic:
            raise ValueError("uses_session_seed requires a deterministic pass contract")
        if self.normalizer is not None and not callable(self.normalizer):
            raise TypeError("pass normalizer must be callable")
        overlap = self.preserved_analyses & self.produced_analyses
        if overlap:
            rendered = ", ".join(str(item) for item in sorted(overlap))
            raise ValueError(f"analyses cannot be both preserved and produced: {rendered}")

    @staticmethod
    def _callable_identity(value: Callable[..., Any] | None) -> str:
        if value is None:
            return ""
        return f"{value.__module__}.{value.__qualname__}"

    @property
    def normalizer_identity(self) -> str | None:
        """Stable diagnostic identity of the canonical derivation law."""

        identity = self._callable_identity(self.normalizer)
        return identity or None

    @property
    def digest(self) -> str:
        """Content identity of the contract and its declared callable identities."""

        rules = tuple(
            (
                rule.transform,
                rule.source_entity,
                rule.target_entity,
                rule.rewrite,
                tuple((claim.name, self._callable_identity(claim.verifier)) for claim in rule.preserves),
                rule.introduces,
                rule.forbids,
                self._callable_identity(rule.verifier),
            )
            for rule in self.rules
        )
        return content_digest(
            (
                self.name,
                self.revision,
                self.input_type.SCHEMA_NAME,
                str(self.input_schema.minimum),
                str(self.input_schema.maximum),
                self.output_type.SCHEMA_NAME,
                str(self.output_schema),
                tuple(sorted(axis.value for axis in self.required_bindings)),
                tuple(sorted(str(key) for key in self.required_analyses)),
                tuple(sorted(str(key) for key in self.preserved_analyses)),
                tuple(sorted(str(key) for key in self.produced_analyses)),
                self.mutation_model.value,
                self.verification.value,
                self.deterministic,
                self.uses_session_seed,
                rules,
                self._callable_identity(self.normalizer),
            ),
            "pass-contract",
        )

    @classmethod
    def create(
        cls,
        name: str,
        input_type: type[CanonicalIRMixin],
        output_type: type[CanonicalIRMixin],
        *,
        revision: str = "1",
        **options: Any,
    ) -> PassContract:
        """Build the common exact-schema contract without hiding its resolved values."""

        return cls(
            name=name,
            revision=revision,
            input_type=input_type,
            input_schema=SchemaRange.exact(input_type.SCHEMA_VERSION),
            output_type=output_type,
            output_schema=output_type.SCHEMA_VERSION,
            **options,
        )


@dataclass(frozen=True)
class PassContext:
    session: SynthesisSession
    analyses: AnalysisStore

    def analysis(self, ir: CanonicalIRMixin, key: AnalysisKey) -> Any:
        return self.analyses.get(ir.digest, key, self.session.fingerprint)


@dataclass(frozen=True)
class PassResult(Generic[OutputIR]):
    ir: OutputIR
    analyses: tuple[AnalysisProduct, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "analyses", tuple(self.analyses))


class DerivationPass(ABC, Generic[InputIR, OutputIR]):
    """Base class for one declaratively contracted lowering or analysis pass."""

    contract: PassContract

    @abstractmethod
    def run(self, ir: InputIR, context: PassContext) -> OutputIR | PassResult[OutputIR]:
        raise NotImplementedError


@dataclass(frozen=True)
class FunctionPass(DerivationPass[InputIR, OutputIR]):
    """Small adapter for pure functions; production passes may use named classes."""

    contract: PassContract
    function: Callable[[InputIR, PassContext], OutputIR | PassResult[OutputIR]]

    def run(self, ir: InputIR, context: PassContext) -> OutputIR | PassResult[OutputIR]:
        return self.function(ir, context)


@dataclass(frozen=True)
class PassPipeline:
    """Immutable, type-checked pass composition."""

    passes: tuple[DerivationPass[Any, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "passes", tuple(self.passes))
        for previous, following in zip(self.passes, self.passes[1:]):
            self._verify_link(previous.contract, following.contract)

    @staticmethod
    def _verify_link(previous: PassContract, following: PassContract) -> None:
        if previous.output_type is not following.input_type:
            raise PassContractError(
                f"pipeline type mismatch: {previous.name} outputs {previous.output_type.__name__}, "
                f"but {following.name} expects {following.input_type.__name__}"
            )
        if not following.input_schema.accepts(previous.output_schema):
            raise PassContractError(
                f"pipeline schema mismatch: {previous.name} outputs {previous.output_schema}, "
                f"but {following.name} accepts {following.input_schema}"
            )

    @classmethod
    def of(cls, *passes: DerivationPass[Any, Any]) -> PassPipeline:
        return cls(tuple(passes))

    def then(self, derivation_pass: DerivationPass[Any, Any]) -> PassPipeline:
        if self.passes:
            self._verify_link(self.passes[-1].contract, derivation_pass.contract)
        return PassPipeline(self.passes + (derivation_pass,))

    def __iter__(self) -> Iterator[DerivationPass[Any, Any]]:
        return iter(self.passes)

    def __len__(self) -> int:
        return len(self.passes)


@dataclass(frozen=True)
class PassRecord:
    pass_name: str
    contract_revision: str
    contract_digest: str
    input_digest: str
    output_digest: str
    session_fingerprint: str
    duration_ns: int
    mutation_model: MutationModel
    produced_analyses: tuple[AnalysisKey, ...]
    transition_report: TransitionReport


@dataclass(frozen=True)
class PassCheckpoint:
    """One inspectable lowering boundary for profilers and validation hooks."""

    record: PassRecord
    ir: CanonicalIRMixin
    analysis_products: tuple[AnalysisProduct, ...]


class PassObserver(ABC):
    """Synchronous read-only hook invoked before a pass transition is committed."""

    @abstractmethod
    def inspect(self, checkpoint: PassCheckpoint, session: SynthesisSession) -> None:
        raise NotImplementedError


@dataclass(frozen=True)
class PipelineResult(Generic[OutputIR]):
    ir: OutputIR
    records: tuple[PassRecord, ...]
    checkpoints: tuple[PassCheckpoint, ...]


class PassManager:
    """Executes pipelines while enforcing all pass contracts at the boundary."""

    def __init__(
        self,
        analyses: AnalysisStore | None = None,
        *,
        observers: Iterable[PassObserver] = (),
        determinism: DeterminismPolicy = DeterminismPolicy.OFF,
    ) -> None:
        self.analyses = analyses if analyses is not None else AnalysisStore()
        self.observers = tuple(observers)
        if not isinstance(determinism, DeterminismPolicy):
            raise TypeError("determinism must be a DeterminismPolicy")
        self.determinism = determinism

    @staticmethod
    def _result_signature(result: PassResult[Any]) -> tuple[str, tuple[tuple[str, str], ...]]:
        products = tuple(
            (
                str(product.key),
                content_digest(freeze(product.value), f"analysis:{product.key}"),
            )
            for product in result.analyses
        )
        return result.ir.digest, products

    def run(
        self,
        pipeline: PassPipeline,
        ir: InputIR,
        *,
        session: SynthesisSession,
    ) -> Checked[PipelineResult[Any]]:
        """Execute a pipeline and return expected contract failures as diagnostics."""

        try:
            return Ok(self._run(pipeline, ir, session=session))
        except PassExecutionError:
            raise
        except SynthesisError as error:
            return Err(
                DiagnosticSet.of(
                    Diagnostic(
                        _diagnostic_code(error),
                        str(error),
                        ("pipeline",),
                    )
                )
            )

    def require_run(
        self,
        pipeline: PassPipeline,
        ir: InputIR,
        *,
        session: SynthesisSession,
    ) -> PipelineResult[Any]:
        """Explicit exception adapter for application and legacy boundaries."""

        return self._run(pipeline, ir, session=session)

    def _run(
        self,
        pipeline: PassPipeline,
        ir: InputIR,
        *,
        session: SynthesisSession,
    ) -> PipelineResult[Any]:
        current: CanonicalIRMixin = ir
        records = []
        checkpoints = []

        for derivation_pass in pipeline:
            contract = derivation_pass.contract
            if type(current) is not contract.input_type:
                raise PassContractError(
                    f"pass {contract.name!r} expects {contract.input_type.__name__}, got {type(current).__name__}"
                )
            if not contract.input_schema.accepts(current.header.schema_version):
                raise PassContractError(
                    f"pass {contract.name!r} does not accept schema {current.header.schema_version}"
                )
            session.require(*tuple(sorted(contract.required_bindings, key=lambda item: item.value)))
            if contract.verification.verifies_input:
                current.require_valid()

            input_digest = current.digest
            missing = tuple(
                key
                for key in sorted(contract.required_analyses)
                if not self.analyses.has(input_digest, key, session.fingerprint)
            )
            if missing:
                rendered = ", ".join(str(item) for item in missing)
                raise MissingAnalysisError(f"pass {contract.name!r} requires missing analyses: {rendered}")

            def invoke(
                store: AnalysisStore,
                *,
                isolate_input: bool,
                snapshot: CanonicalIRMixin = current,
                current_contract: PassContract = contract,
                current_pass: DerivationPass[Any, Any] = derivation_pass,
            ) -> PassResult[Any]:
                working = (
                    type(snapshot).require_from_json(snapshot.to_json())
                    if isolate_input or current_contract.mutation_model is MutationModel.TRANSACTIONAL
                    else snapshot
                )
                context = PassContext(session=session, analyses=store)
                try:
                    raw_result = current_pass.run(working, context)
                except SynthesisError:
                    raise
                except Exception as error:
                    raise PassExecutionError(current_contract.name, error) from error
                return raw_result if isinstance(raw_result, PassResult) else PassResult(raw_result)

            started = time.perf_counter_ns()
            if self.determinism is DeterminismPolicy.VERIFY and contract.deterministic:
                result = invoke(self.analyses.clone(), isolate_input=True)
            else:
                result = invoke(self.analyses, isolate_input=False)
            duration_ns = time.perf_counter_ns() - started
            if self.determinism is DeterminismPolicy.VERIFY and contract.deterministic:
                replay = invoke(self.analyses.clone(), isolate_input=True)
                if self._result_signature(result) != self._result_signature(replay):
                    raise PassContractError(f"pass {contract.name!r} failed deterministic replay")

            if contract.mutation_model is MutationModel.IMMUTABLE:
                try:
                    post_pass_input_digest = current.digest
                except SynthesisError as error:
                    raise PassContractError(
                        f"immutable pass {contract.name!r} corrupted its input snapshot: {error}"
                    ) from error
                if post_pass_input_digest != input_digest:
                    raise PassContractError(f"immutable pass {contract.name!r} mutated its input snapshot")
            output = result.ir
            if type(output) is not contract.output_type:
                raise PassContractError(
                    f"pass {contract.name!r} returned {type(output).__name__}, expected {contract.output_type.__name__}"
                )
            if output.header.schema_version != contract.output_schema:
                raise PassContractError(
                    f"pass {contract.name!r} returned schema {output.header.schema_version}, "
                    f"expected {contract.output_schema}"
                )
            if contract.verification.verifies_output:
                output.require_valid()
            output_digest = output.digest
            if output_digest != input_digest and input_digest not in output.header.parent_digests:
                raise PassContractError(
                    f"pass {contract.name!r} changed the IR without retaining its input digest in lineage"
                )
            transition_report = TransitionVerifier.verify(current, output, contract, session)

            product_keys = tuple(product.key for product in result.analyses)
            if len(set(product_keys)) != len(product_keys):
                raise PassContractError(f"pass {contract.name!r} returned duplicate analysis products")
            if frozenset(product_keys) != contract.produced_analyses:
                expected = ", ".join(str(item) for item in sorted(contract.produced_analyses)) or "<none>"
                actual = ", ".join(str(item) for item in sorted(product_keys)) or "<none>"
                raise PassContractError(
                    f"pass {contract.name!r} analysis products differ from its contract: "
                    f"expected {expected}; got {actual}"
                )
            record = PassRecord(
                pass_name=contract.name,
                contract_revision=contract.revision,
                contract_digest=contract.digest,
                input_digest=input_digest,
                output_digest=output_digest,
                session_fingerprint=session.fingerprint,
                duration_ns=duration_ns,
                mutation_model=contract.mutation_model,
                produced_analyses=tuple(sorted(product_keys)),
                transition_report=transition_report,
            )
            checkpoint = PassCheckpoint(
                record=record,
                ir=output,
                analysis_products=result.analyses,
            )
            for observer in self.observers:
                try:
                    observer.inspect(checkpoint, session)
                except SynthesisError:
                    raise
                except Exception as error:
                    observer_name = f"{contract.name}:observer:{type(observer).__name__}"
                    raise PassExecutionError(observer_name, error) from error

            self.analyses.commit_transition(
                source_digest=input_digest,
                target_digest=output_digest,
                context_fingerprint=session.fingerprint,
                preserved=contract.preserved_analyses,
                products=result.analyses,
                producer=contract.name,
            )
            records.append(record)
            checkpoints.append(checkpoint)
            current = output

        return PipelineResult(ir=current, records=tuple(records), checkpoints=tuple(checkpoints))

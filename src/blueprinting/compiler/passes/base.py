"""Declarative, immutable compiler pass infrastructure.

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
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, TypeVar

from ..axes import BindingAxis
from ..codec import content_digest
from ..errors import (
    CompilerError,
    MissingAnalysisError,
    PassContractError,
    PassExecutionError,
    SerializationError,
)
from ..frozen import freeze
from ..ir.common import CanonicalIRMixin, SchemaVersion
from ..session import CompilationSession

InputIR = TypeVar("InputIR", bound=CanonicalIRMixin)
OutputIR = TypeVar("OutputIR", bound=CanonicalIRMixin)


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
    """Versioned identity of one derived analysis kind."""

    namespace: str
    name: str
    version: int = 1

    def __post_init__(self) -> None:
        if not self.namespace or not self.name:
            raise ValueError("analysis namespace and name must not be empty")
        if isinstance(self.version, bool) or not isinstance(self.version, int) or self.version <= 0:
            raise ValueError("analysis version must be a positive integer")

    def __str__(self) -> str:
        return f"{self.namespace}.{self.name}@{self.version}"


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


@dataclass(frozen=True)
class PassContract:
    """Complete static contract for one canonical IR transition."""

    name: str
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

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("pass name must not be empty")
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
        overlap = self.preserved_analyses & self.produced_analyses
        if overlap:
            rendered = ", ".join(str(item) for item in sorted(overlap))
            raise ValueError(f"analyses cannot be both preserved and produced: {rendered}")

    @classmethod
    def create(
        cls,
        name: str,
        input_type: type[CanonicalIRMixin],
        output_type: type[CanonicalIRMixin],
        **options: Any,
    ) -> PassContract:
        """Build the common exact-schema contract without hiding its resolved values."""

        return cls(
            name=name,
            input_type=input_type,
            input_schema=SchemaRange.exact(input_type.SCHEMA_VERSION),
            output_type=output_type,
            output_schema=output_type.SCHEMA_VERSION,
            **options,
        )


@dataclass(frozen=True)
class PassContext:
    session: CompilationSession
    analyses: AnalysisStore

    def analysis(self, ir: CanonicalIRMixin, key: AnalysisKey) -> Any:
        return self.analyses.get(ir.digest, key, self.session.fingerprint)


@dataclass(frozen=True)
class PassResult(Generic[OutputIR]):
    ir: OutputIR
    analyses: tuple[AnalysisProduct, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "analyses", tuple(self.analyses))


class CompilerPass(ABC, Generic[InputIR, OutputIR]):
    """Base class for one declaratively contracted lowering or analysis pass."""

    contract: PassContract

    @abstractmethod
    def run(self, ir: InputIR, context: PassContext) -> OutputIR | PassResult[OutputIR]:
        raise NotImplementedError


@dataclass(frozen=True)
class FunctionPass(CompilerPass[InputIR, OutputIR]):
    """Small adapter for pure functions; production passes may use named classes."""

    contract: PassContract
    function: Callable[[InputIR, PassContext], OutputIR | PassResult[OutputIR]]

    def run(self, ir: InputIR, context: PassContext) -> OutputIR | PassResult[OutputIR]:
        return self.function(ir, context)


@dataclass(frozen=True)
class PassPipeline:
    """Immutable, type-checked pass composition."""

    passes: tuple[CompilerPass[Any, Any], ...] = ()

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
    def of(cls, *passes: CompilerPass[Any, Any]) -> PassPipeline:
        return cls(tuple(passes))

    def then(self, compiler_pass: CompilerPass[Any, Any]) -> PassPipeline:
        if self.passes:
            self._verify_link(self.passes[-1].contract, compiler_pass.contract)
        return PassPipeline(self.passes + (compiler_pass,))

    def __iter__(self) -> Iterator[CompilerPass[Any, Any]]:
        return iter(self.passes)

    def __len__(self) -> int:
        return len(self.passes)


@dataclass(frozen=True)
class PassRecord:
    pass_name: str
    input_digest: str
    output_digest: str
    session_fingerprint: str
    duration_ns: int
    mutation_model: MutationModel
    produced_analyses: tuple[AnalysisKey, ...]


@dataclass(frozen=True)
class PassCheckpoint:
    """One inspectable lowering boundary for profilers and validation hooks."""

    record: PassRecord
    ir: CanonicalIRMixin
    analysis_products: tuple[AnalysisProduct, ...]


class PassObserver(ABC):
    """Synchronous read-only hook invoked before a pass transition is committed."""

    @abstractmethod
    def inspect(self, checkpoint: PassCheckpoint, session: CompilationSession) -> None:
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
    ) -> None:
        self.analyses = analyses if analyses is not None else AnalysisStore()
        self.observers = tuple(observers)

    def run(
        self,
        pipeline: PassPipeline,
        ir: InputIR,
        *,
        session: CompilationSession,
    ) -> PipelineResult[Any]:
        current: CanonicalIRMixin = ir
        records = []
        checkpoints = []
        context = PassContext(session=session, analyses=self.analyses)

        for compiler_pass in pipeline:
            contract = compiler_pass.contract
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

            if contract.mutation_model is MutationModel.TRANSACTIONAL:
                working = type(current).from_json(current.to_json())
            else:
                working = current

            started = time.perf_counter_ns()
            try:
                raw_result = compiler_pass.run(working, context)
            except CompilerError:
                raise
            except Exception as error:
                raise PassExecutionError(contract.name, error) from error
            duration_ns = time.perf_counter_ns() - started
            result = raw_result if isinstance(raw_result, PassResult) else PassResult(raw_result)

            if contract.mutation_model is MutationModel.IMMUTABLE:
                try:
                    post_pass_input_digest = current.digest
                except CompilerError as error:
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
                input_digest=input_digest,
                output_digest=output_digest,
                session_fingerprint=session.fingerprint,
                duration_ns=duration_ns,
                mutation_model=contract.mutation_model,
                produced_analyses=tuple(sorted(product_keys)),
            )
            checkpoint = PassCheckpoint(
                record=record,
                ir=output,
                analysis_products=result.analyses,
            )
            for observer in self.observers:
                try:
                    observer.inspect(checkpoint, session)
                except CompilerError:
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

"""Runtime contract compiler for Blueprinting's typed Python declarations.

This module deliberately has no dependency on mypy. It compiles the same
canonical record, closed ADT, and derivation declarations used by ordinary
runtime execution into one deterministic manifest.
"""

from __future__ import annotations

from dataclasses import dataclass

from .schema.contracts import TypeUniverse, compile_type_universe
from .schema.diagnostics import DiagnosticBag
from .schema.result import Checked, Err, Ok, checked
from .synthesizer.passes.deriving import pass_contract_manifest


@dataclass(frozen=True, order=True)
class DerivationContractDeclaration:
    """Stable projection of one loaded pass contract."""

    name: str
    revision: str
    python_type: str
    input_schema: str
    output_schema: str
    normalizer: str | None
    rules: tuple[str, ...]
    digest: str


@dataclass(frozen=True)
class RuntimeContractManifest:
    """Complete declaration universe checked without a static analyzer."""

    types: TypeUniverse
    derivations: tuple[DerivationContractDeclaration, ...]

    @property
    def digest(self) -> str:
        from .schema.codec import content_digest

        return content_digest(
            (
                self.types.digest,
                tuple(
                    (
                        item.name,
                        item.revision,
                        item.python_type,
                        item.input_schema,
                        item.output_schema,
                        item.normalizer,
                        item.rules,
                        item.digest,
                    )
                    for item in self.derivations
                ),
            ),
            "runtime-contract-manifest",
        )


def _qualified_name(value: type[object]) -> str:
    return f"{value.__module__}.{value.__qualname__}"


def _load_builtin_declarations() -> None:
    """Import declaration owners explicitly; imports are registration only."""

    from . import mapping as _mapping
    from .analysis.cost import protocol as _cost_protocol
    from .synthesizer import expr as _expr
    from .synthesizer.stages.concrete_plan import ir as _concrete_ir
    from .synthesizer.stages.concrete_plan import passes as _concrete_passes
    from .synthesizer.stages.distributed import ir as _distributed_ir
    from .synthesizer.stages.distributed import passes as _distributed_passes
    from .synthesizer.stages.machine import ir as _machine_ir
    from .synthesizer.stages.model import ir as _model_ir
    from .synthesizer.stages.portable_plan import ir as _portable_ir
    from .synthesizer.stages.portable_plan import passes as _portable_passes

    _ = (
        _mapping,
        _cost_protocol,
        _expr,
        _concrete_ir,
        _concrete_passes,
        _distributed_ir,
        _distributed_passes,
        _machine_ir,
        _model_ir,
        _portable_ir,
        _portable_passes,
    )


class ContractCompiler:
    """Compile and validate loaded runtime type and derivation declarations."""

    def __init__(self, *, load_builtins: bool = True) -> None:
        self.load_builtins = load_builtins

    def compile(self) -> Checked[RuntimeContractManifest]:
        if self.load_builtins:
            _load_builtin_declarations()

        bag = DiagnosticBag()
        declarations: list[DerivationContractDeclaration] = []
        identities: dict[tuple[str, str], str] = {}
        names: dict[str, str] = {}
        for pass_type, contract in pass_contract_manifest():
            python_type = _qualified_name(pass_type)
            identity = (contract.name, contract.revision)
            previous = identities.get(identity)
            if previous is not None and previous != python_type:
                bag.error(
                    "contract.pass.duplicate_identity",
                    f"{contract.name}@{contract.revision} is declared by both {previous} and {python_type}",
                    "pass",
                    contract.name,
                )
            identities[identity] = python_type
            previous_revision = names.get(contract.name)
            if previous_revision is not None and previous_revision != contract.revision:
                bag.error(
                    "contract.pass.multiple_revisions",
                    f"runtime loaded revisions {previous_revision!r} and {contract.revision!r}",
                    "pass",
                    contract.name,
                )
            names[contract.name] = contract.revision
            if not contract.verification.verifies_output:
                bag.error(
                    "contract.pass.output_not_verified",
                    "a committed derivation must structurally verify its output",
                    "pass",
                    contract.name,
                )
            if contract.input_type is not contract.output_type:
                if not contract.rules:
                    bag.error(
                        "contract.pass.missing_relations",
                        "a cross-stage derivation must declare typed lineage relations",
                        "pass",
                        contract.name,
                    )
                incomplete = tuple(
                    rule.transform for rule in contract.rules if rule.verifier is None and not rule.preserves
                )
                if incomplete:
                    bag.error(
                        "contract.pass.missing_invariants",
                        f"cross-stage relations lack independent executable invariants: {', '.join(incomplete)}",
                        "pass",
                        contract.name,
                    )
                if contract.normalizer is None:
                    bag.error(
                        "contract.pass.missing_normal_form",
                        "a cross-stage derivation must declare a complete executable normal form",
                        "pass",
                        contract.name,
                    )
            declarations.append(
                DerivationContractDeclaration(
                    contract.name,
                    contract.revision,
                    python_type,
                    f"{contract.input_type.SCHEMA_NAME}@{contract.input_schema.minimum}",
                    f"{contract.output_type.SCHEMA_NAME}@{contract.output_schema}",
                    contract.normalizer_identity,
                    tuple(rule.transform for rule in contract.rules),
                    contract.digest,
                )
            )

        type_result = compile_type_universe(additional=bag.report())
        if isinstance(type_result, Err):
            return type_result
        assert isinstance(type_result, Ok)
        manifest = RuntimeContractManifest(type_result.value, tuple(declarations))
        return checked(manifest, type_result.diagnostics)


def compile_runtime_contracts() -> Checked[RuntimeContractManifest]:
    """Compile the built-in declarations using the mypy-independent layer."""

    return ContractCompiler().compile()


def require_runtime_contracts() -> RuntimeContractManifest:
    """Exception adapter intended only for CLI and CI process boundaries."""

    return compile_runtime_contracts().or_raise()


__all__ = [
    "ContractCompiler",
    "DerivationContractDeclaration",
    "RuntimeContractManifest",
    "compile_runtime_contracts",
    "require_runtime_contracts",
]

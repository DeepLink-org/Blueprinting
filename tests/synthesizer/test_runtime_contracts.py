from __future__ import annotations

from dataclasses import replace

import blueprinting.mapping  # noqa: F401 - load its algebraic contracts
from blueprinting.contracts import compile_runtime_contracts
from blueprinting.schema import (
    Diagnostic,
    DiagnosticSet,
    Err,
    Ok,
    Severity,
    checked,
    collect_results,
)


def test_result_maps_chains_and_accumulates_independent_diagnostics() -> None:
    warning = DiagnosticSet.of(Diagnostic("fixture.warning", "warning", severity=Severity.WARNING))
    failure_one = DiagnosticSet.of(Diagnostic("fixture.one", "first failure"))
    failure_two = DiagnosticSet.of(Diagnostic("fixture.two", "second failure"))

    success = Ok(2, warning).map(lambda value: value + 1).and_then(lambda value: Ok(value * 2))
    failures = collect_results((Ok(1), Err(failure_one), Ok(2), Err(failure_two)))

    assert isinstance(success, Ok)
    assert success.value == 6
    assert success.diagnostics == warning
    assert isinstance(failures, Err)
    assert tuple(item.code for item in failures.error.errors) == ("fixture.one", "fixture.two")


def test_checked_selects_success_or_failure_from_diagnostics() -> None:
    assert isinstance(checked("value"), Ok)
    assert isinstance(checked("value", DiagnosticSet.of(Diagnostic("fixture.error", "broken"))), Err)


def test_runtime_type_universe_is_closed_and_deterministic() -> None:
    first = compile_runtime_contracts().or_raise().types
    second = compile_runtime_contracts().or_raise().types

    assert first == second
    assert first.digest == second.digest
    assert {item.wire for item in first.algebraic_families} >= {
        "blueprinting.expression.scalar",
        "blueprinting.ir.distributed-task.task",
        "blueprinting.ir.concrete-plan.command-body",
        "blueprinting.ir.concrete-plan.synchronization",
        "blueprinting.mapping.pipeline-schedule",
        "blueprinting.binding.workload-mode",
        "blueprinting.ir.portable-plan.task-body",
        "blueprinting.analysis.cost.support",
        "blueprinting.ir.distributed-task.collective",
    }


def test_runtime_contract_digest_covers_record_fields_defaults_and_enum_members() -> None:
    manifest = compile_runtime_contracts().or_raise()
    types = manifest.types
    workload = next(item for item in types.canonical_types if item.wire == "blueprinting.binding.workload")
    phase = next(item for item in types.canonical_types if item.wire == "blueprinting.binding.inference-phase")
    collective = next(
        item for item in types.algebraic_families if item.wire == "blueprinting.ir.distributed-task.collective"
    )

    assert tuple(field.name for field in workload.fields) == ("mode", "semantic", "attributes")
    assert workload.fields[0].annotation.startswith("union[")
    assert workload.fields[0].default_kind == "required"
    assert workload.fields[2].default_kind == "factory"
    assert workload.fields[2].default_identity == "blueprinting.schema.frozen.FrozenDict"
    assert workload.fields[2].default_value == '{"$map":[]}'
    assert tuple((member.name, member.value) for member in phase.enum_members) == (
        ("PREFILL", '"prefill"'),
        ("DECODE", '"decode"'),
    )

    def with_canonical_type(updated):
        return replace(
            types,
            canonical_types=tuple(updated if item.wire == updated.wire else item for item in types.canonical_types),
        )

    altered_universes = (
        with_canonical_type(replace(workload, fields=tuple(reversed(workload.fields)))),
        with_canonical_type(
            replace(
                workload,
                fields=(replace(workload.fields[0], annotation="tests.changed-shape"), *workload.fields[1:]),
            )
        ),
        with_canonical_type(
            replace(
                workload,
                fields=(*workload.fields[:2], replace(workload.fields[2], default_value='{"$map":[["x",1]]}')),
            )
        ),
        with_canonical_type(
            replace(
                phase,
                enum_members=(replace(phase.enum_members[0], value='"changed"'), *phase.enum_members[1:]),
            )
        ),
        replace(
            types,
            algebraic_families=tuple(
                replace(collective, variants=collective.variants[:-1]) if item is collective else item
                for item in types.algebraic_families
            ),
        ),
    )

    assert all(item.digest != types.digest for item in altered_universes)
    assert all(replace(manifest, types=item).digest != manifest.digest for item in altered_universes)


def test_runtime_contract_compiler_covers_all_builtin_derivations() -> None:
    first = compile_runtime_contracts()
    second = compile_runtime_contracts()

    assert isinstance(first, Ok)
    assert isinstance(second, Ok)
    assert first.value.digest == second.value.digest
    assert {item.name for item in first.value.derivations} >= {
        "transformer-distribute",
        "transformer-inference-distribute",
        "transformer-plan-work",
        "transformer-inference-plan-work",
        "reference-queue-bind",
        "reference-slot-bind",
    }
    production = tuple(item for item in first.value.derivations if not item.name.startswith("fixture."))
    assert all(item.normalizer is not None for item in production)

from __future__ import annotations

import pytest

from blueprinting.schema import FrozenDict, SerializationError, canonical_dumps, canonical_loads
from blueprinting.synthesizer import (
    BindingAxis,
    BindingError,
    DeploymentProfile,
    Symbol,
    SynthesisSession,
    TargetProfile,
    TargetRequirements,
    WorkloadBinding,
    WorkloadMode,
)
from blueprinting.synthesizer.ir import PortablePlanIR


def _target(name: str, architecture: str) -> TargetProfile:
    return TargetProfile(
        name=name,
        architecture=architecture,
        architecture_revision="1",
        runtime_stack="fixture-runtime",
        runtime_revision="1",
        target_abi="fixture-abi-v1",
        supported_dtypes=frozenset({"f16"}),
        supported_collectives=frozenset({"all_reduce"}),
        capabilities=frozenset({"matrix_multiply"}),
    )


def test_workload_binding_only_accepts_finite_workload_symbols() -> None:
    batch = Symbol("batch", BindingAxis.WORKLOAD, positive=True)

    assert WorkloadBinding(WorkloadMode.INFERENCE, batch_size=batch).batch_size is batch
    with pytest.raises(BindingError, match="workload symbols"):
        WorkloadBinding(
            WorkloadMode.INFERENCE,
            batch_size=Symbol("target_batch", BindingAxis.TARGET),
        )
    with pytest.raises(BindingError, match="finite"):
        WorkloadBinding(WorkloadMode.INFERENCE, batch_size=float("nan"))


def test_target_and_deployment_requirements_are_checked() -> None:
    requirements = TargetRequirements(
        device_count_range=(2, 4),
        minimum_memory_bytes=80,
        required_dtypes=frozenset({"f16"}),
        required_collectives=frozenset({"all_reduce"}),
        required_capabilities=frozenset({"matrix_multiply"}),
    )
    target = _target("virtual", "virtual-v1")
    deployment = DeploymentProfile(
        "fixture-deployment",
        2,
        FrozenDict({"links": ((0, 1),)}),
        "environment-v1",
        available_memory_bytes=(80, 96),
    )

    assert target.satisfies(requirements)
    assert deployment.satisfies(requirements)
    assert not DeploymentProfile(
        "too-small",
        1,
        FrozenDict(),
        "environment-v1",
        available_memory_bytes=(128,),
    ).satisfies(requirements)


def test_target_binding_changes_session_not_portable_plan(
    portable_ir: PortablePlanIR,
) -> None:
    first = SynthesisSession().with_binding(_target("virtual-a", "virtual-v1"))
    second = SynthesisSession().with_binding(_target("virtual-b", "virtual-v2"))
    portable_digest = portable_ir.digest

    assert first.fingerprint != second.fingerprint
    assert portable_ir.digest == portable_digest


LEGACY_TARGET_PROFILE_JSON = (
    '{"$type":"compiler.binding.target","fields":{'
    '"architecture":"virtual","architecture_revision":"1",'
    '"attributes":{"$map":[]},"capabilities":{"$frozenset":[]},'
    '"collective_library":"none","compiler_abi":"abi-v1",'
    '"execution_engines":{"$frozenset":[]},"kernel_library":"none",'
    '"memory_spaces":{"$frozenset":[]},"name":"fixture",'
    '"runtime_revision":"1","runtime_stack":"runtime",'
    '"supported_collectives":{"$frozenset":[]},'
    '"supported_dtypes":{"$frozenset":[]},'
    '"supported_operations":{"$frozenset":[]}}}'
)


def test_legacy_target_profile_field_decodes_and_reencodes_canonically() -> None:
    target = canonical_loads(LEGACY_TARGET_PROFILE_JSON)

    assert isinstance(target, TargetProfile)
    assert target.target_abi == "abi-v1"
    encoded = canonical_dumps(target)
    assert '"target_abi":"abi-v1"' in encoded
    assert '"compiler_abi"' not in encoded


def test_target_profile_rejects_legacy_and_current_field_together() -> None:
    conflicting = LEGACY_TARGET_PROFILE_JSON.replace(
        '"compiler_abi":"abi-v1",',
        '"compiler_abi":"abi-v1","target_abi":"abi-v1",',
    )

    with pytest.raises(SerializationError, match="both a current field and its legacy alias"):
        canonical_loads(conflicting)

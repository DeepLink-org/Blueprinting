from __future__ import annotations

import pytest

from blueprinting.compiler import (
    BindingAxis,
    BindingError,
    CompilationSession,
    DeploymentProfile,
    FrozenDict,
    Symbol,
    TargetProfile,
    TargetRequirements,
    WorkloadBinding,
    WorkloadMode,
)
from blueprinting.compiler.ir import PortablePlanIR


def _target(name: str, architecture: str) -> TargetProfile:
    return TargetProfile(
        name=name,
        architecture=architecture,
        architecture_revision="1",
        runtime_stack="fixture-runtime",
        runtime_revision="1",
        compiler_abi="fixture-abi-v1",
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
    first = CompilationSession().with_binding(_target("virtual-a", "virtual-v1"))
    second = CompilationSession().with_binding(_target("virtual-b", "virtual-v2"))
    portable_digest = portable_ir.digest

    assert first.fingerprint != second.fingerprint
    assert portable_ir.digest == portable_digest

from __future__ import annotations

import pytest

from blueprinting.schema import FrozenDict
from blueprinting.synthesizer import (
    BindingError,
    DeploymentProfile,
    InferencePhase,
    InferenceWorkload,
    StrategyBinding,
    SynthesisSession,
    TargetProfile,
    TargetRequirements,
    WorkloadBinding,
)
from blueprinting.synthesizer.dialects.transformer import TransformerInferenceWorkloadSemantic
from blueprinting.synthesizer.stages.portable_plan.ir import PortablePlanIR


def _target(name: str, architecture: str) -> TargetProfile:
    return TargetProfile(
        name=name,
        architecture=architecture,
        architecture_revision="0",
        runtime_stack="fixture-runtime",
        runtime_revision="0",
        target_abi="fixture-abi-v0",
        supported_dtypes=frozenset({"f16"}),
        supported_collectives=frozenset({"all_reduce"}),
        capabilities=frozenset({"matrix_multiply"}),
    )


def test_workload_binding_has_one_typed_source_for_workload_facts() -> None:
    mode = InferenceWorkload(InferencePhase.PREFILL)
    semantic = TransformerInferenceWorkloadSemantic(4, 16, "float16")
    binding = WorkloadBinding(mode, semantic)

    assert binding.semantic is semantic
    assert not hasattr(binding, "batch_size")
    with pytest.raises(TypeError, match="batch_size"):
        WorkloadBinding(mode, semantic, batch_size=4)  # type: ignore[call-arg]


def test_binding_semantics_cannot_be_smuggled_through_attributes() -> None:
    with pytest.raises(BindingError, match="typed semantic field"):
        WorkloadBinding(
            InferenceWorkload(InferencePhase.DECODE),
            attributes=FrozenDict({"nested": FrozenDict({"query_tokens": 8})}),
        )
    with pytest.raises(BindingError, match="typed semantic field"):
        StrategyBinding(attributes=FrozenDict({"mapping_spec": "legacy"}))
    with pytest.raises(BindingError, match="typed semantic field"):
        StrategyBinding(attributes=FrozenDict({"nested": FrozenDict({"tensor_parallel": 8})}))


def test_target_and_deployment_requirements_are_checked() -> None:
    requirements = TargetRequirements(
        device_count_range=(2, 4),
        minimum_memory_bytes=80,
        required_dtypes=frozenset({"f16"}),
        required_collectives=frozenset({"all_reduce"}),
        required_capabilities=frozenset({"matrix_multiply"}),
    )
    target = _target("virtual", "virtual-abi-v0")
    deployment = DeploymentProfile(
        "fixture-deployment",
        2,
        FrozenDict({"links": ((0, 1),)}),
        "environment-v0",
        available_memory_bytes=(80, 96),
    )

    assert target.satisfies(requirements)
    assert deployment.satisfies(requirements)
    assert not DeploymentProfile(
        "too-small",
        1,
        FrozenDict(),
        "environment-v0",
        available_memory_bytes=(128,),
    ).satisfies(requirements)


def test_target_binding_changes_session_not_portable_plan(
    portable_ir: PortablePlanIR,
) -> None:
    first = SynthesisSession().with_binding(_target("virtual-a", "virtual-abi-v0"))
    second = SynthesisSession().with_binding(_target("virtual-b", "virtual-abi-v0"))
    portable_digest = portable_ir.digest

    assert first.fingerprint != second.fingerprint
    assert portable_ir.digest == portable_digest

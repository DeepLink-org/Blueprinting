from __future__ import annotations

from dataclasses import replace

import pytest

from blueprinting.schema import FrozenDict
from blueprinting.synthesizer import DeploymentProfile, PassContractError, SynthesisSession, TargetProfile
from blueprinting.synthesizer.bindings import BindingSet
from blueprinting.synthesizer.passes import (
    DeterminismPolicy,
    PassManager,
    PassPipeline,
    TransitionVerificationStatus,
    TransitionVerifier,
)
from blueprinting.synthesizer.stages.concrete_plan.ir import (
    ConcretePlanIR,
    ImplementationRef,
    QueueScheduleExtension,
    SlotDataflowExtension,
)
from blueprinting.synthesizer.stages.concrete_plan.passes import (
    BindReferenceQueueTargetPass,
    BindReferenceSlotTargetPass,
)


def _session() -> SynthesisSession:
    target = TargetProfile(
        name="reference",
        architecture="virtual",
        architecture_revision="0",
        runtime_stack="reference-runtime",
        runtime_revision="0",
        target_abi="reference-abi-v0",
    )
    deployment = DeploymentProfile(
        name="reference-2",
        device_count=2,
        topology=FrozenDict({"kind": "fully-connected"}),
        environment_revision="0",
        available_memory_bytes=(4096, 4096),
    )
    return SynthesisSession(
        bindings=BindingSet(target=target, deployment=deployment),
        evidence_snapshot="reference-evidence-v0",
    )


def test_two_reference_binders_materialize_distinct_verified_concrete_plans(portable_ir) -> None:
    session = _session()
    queue_result = PassManager(determinism=DeterminismPolicy.VERIFY).require_run(
        PassPipeline.of(BindReferenceQueueTargetPass()),
        portable_ir,
        session=session,
    )
    slot_result = PassManager(determinism=DeterminismPolicy.VERIFY).require_run(
        PassPipeline.of(BindReferenceSlotTargetPass()),
        portable_ir,
        session=session,
    )

    queue = queue_result.ir
    slot = slot_result.ir
    assert isinstance(queue, ConcretePlanIR)
    assert isinstance(queue.target_extension, QueueScheduleExtension)
    assert isinstance(slot.target_extension, SlotDataflowExtension)
    assert queue.verify().is_ok and slot.verify().is_ok
    assert queue.digest != slot.digest
    assert queue_result.records[0].transition_report.verified_relations == len(queue.buffers) + len(queue.commands)
    assert slot_result.records[0].transition_report.verified_relations == len(slot.buffers) + len(slot.commands)
    assert queue_result.records[0].transition_report.status is TransitionVerificationStatus.RELATION_VERIFIED
    assert queue_result.records[0].contract_digest == BindReferenceQueueTargetPass.contract.digest
    assert queue_result.records[0].transition_report.verified_claims == len(
        queue_result.records[0].transition_report.relations
    )


def test_queue_extension_rejects_missing_command_coverage(portable_ir) -> None:
    plan = (
        PassManager()
        .require_run(
            PassPipeline.of(BindReferenceQueueTargetPass()),
            portable_ir,
            session=_session(),
        )
        .ir
    )
    broken = replace(plan, target_extension=QueueScheduleExtension(()))

    assert broken.verify().is_err
    with pytest.raises(Exception, match="queue orders must cover"):
        broken.require_valid()


def test_slot_extension_rejects_dependency_order_violation(portable_ir) -> None:
    plan = (
        PassManager()
        .require_run(
            PassPipeline.of(BindReferenceSlotTargetPass()),
            portable_ir,
            session=_session(),
        )
        .ir
    )
    slots = tuple(
        replace(slot, cycle=len(plan.target_extension.issue_slots) - index - 1)
        for index, slot in enumerate(plan.target_extension.issue_slots)
    )
    broken = replace(plan, target_extension=SlotDataflowExtension(slots, plan.target_extension.routes))

    assert broken.verify().is_err


def test_cross_boundary_rule_rejects_wrong_command_implementation(portable_ir) -> None:
    class WrongImplementationPass(BindReferenceQueueTargetPass):
        def run(self, ir, context):
            plan = super().run(ir, context)
            command = plan.commands[0]
            assert command.implementation is not None
            wrong = replace(
                command,
                body=replace(
                    command.body,
                    implementation=ImplementationRef(
                        command.implementation.namespace,
                        "another.operation",
                        command.implementation.version,
                        command.implementation.abi,
                        command.implementation.variant,
                    ),
                ),
            )
            return replace(plan, commands=(wrong, *plan.commands[1:]))

    with pytest.raises(PassContractError, match="canonical normal form"):
        PassManager().require_run(
            PassPipeline.of(WrongImplementationPass()),
            portable_ir,
            session=_session(),
        )


def test_cross_stage_reports_canonical_conformance_separately_from_relation_invariants(portable_ir) -> None:
    plan = (
        PassManager()
        .require_run(
            PassPipeline.of(BindReferenceQueueTargetPass()),
            portable_ir,
            session=_session(),
        )
        .ir
    )
    contract = replace(
        BindReferenceQueueTargetPass.contract,
        rules=tuple(replace(rule, preserves=()) for rule in BindReferenceQueueTargetPass.contract.rules),
    )

    report = TransitionVerifier.verify(portable_ir, plan, contract, _session())

    assert report.status is TransitionVerificationStatus.RELATION_VERIFIED
    assert report.canonical_conformant
    assert report.canonical_conformance is not None
    assert report.canonical_conformance.name == "canonical normal form"
    assert all(relation.evidence[0].name == "relation invariant" for relation in report.relations)


def test_independent_relation_invariant_rejects_a_canonically_accepted_bad_output(portable_ir) -> None:
    session = _session()
    plan = (
        PassManager()
        .require_run(
            PassPipeline.of(BindReferenceQueueTargetPass()),
            portable_ir,
            session=session,
        )
        .ir
    )
    command = plan.commands[0]
    assert command.implementation is not None
    wrong_command = replace(
        command,
        body=replace(
            command.body,
            implementation=replace(command.implementation, name="another.operation"),
        ),
    )
    wrong_plan = replace(plan, commands=(wrong_command, *plan.commands[1:]))

    # This deliberately permissive normalizer accepts the bad implementation.
    # The independently declared semantic relation must still reject it.
    contract = replace(BindReferenceQueueTargetPass.contract, normalizer=lambda _source, _session: wrong_plan)

    with pytest.raises(PassContractError, match="selected implementation does not match"):
        TransitionVerifier.verify(portable_ir, wrong_plan, contract, session)

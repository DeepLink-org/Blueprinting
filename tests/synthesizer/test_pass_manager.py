from __future__ import annotations

from dataclasses import replace

import pytest

from blueprinting.schema import Err, FrozenDict
from blueprinting.synthesizer import (
    BindingAxis,
    MissingAnalysisError,
    MissingBindingError,
    NodeId,
    PassContractError,
    SynthesisSession,
)
from blueprinting.synthesizer.errors import PassExecutionError
from blueprinting.synthesizer.passes import (
    AnalysisKey,
    AnalysisProduct,
    DeterminismPolicy,
    FunctionPass,
    MutationModel,
    PassCheckpoint,
    PassContract,
    PassManager,
    PassObserver,
    PassPipeline,
    PassResult,
)
from blueprinting.synthesizer.passes.authoring import (
    DerivationPass,
    PassContext,
    PassRule,
    RelationCheckContext,
    claim,
    derivation,
    relation,
)
from blueprinting.synthesizer.stages.common import make_header
from blueprinting.synthesizer.stages.distributed.ir import DistributedTaskIR
from blueprinting.synthesizer.stages.model.ir import ModelIR, ModelOperation


def _model_contract(name: str, **options: object) -> PassContract:
    return PassContract.create(name, ModelIR, ModelIR, **options)


def test_pass_contract_owns_typed_semantic_rule_declarations() -> None:
    rule = PassRule(
        "fixture.lower",
        "ModelOperation",
        "ModelOperation",
        "Preserve the fixture operation",
        preserves=(claim("operation identity", lambda _source, _target, _context: None),),
        introduces=("test marker",),
        forbids=("predicted time",),
    )

    contract = _model_contract("fixture.rules", rules=(rule,))

    assert contract.rules == (rule,)
    with pytest.raises(ValueError, match="transforms must be unique"):
        _model_contract("fixture.duplicate-rules", rules=(rule, rule))


def test_pass_contract_is_derived_from_generic_types_and_explicit_relations() -> None:
    def verify(source: ModelOperation, target: ModelOperation, _context: RelationCheckContext) -> None:
        if source.id != target.id:
            raise ValueError("operation identity changed")

    preserve = relation(
        "fixture.preserve",
        "Preserve the fixture operation",
        source=ModelOperation,
        target=ModelOperation,
        verifier=verify,
        preserves=(claim("operation identity", lambda _source, _target, _context: None),),
        introduces=("test marker",),
    )

    @derivation(
        "fixture.derived",
        revision="1",
        bindings=(BindingAxis.WORKLOAD,),
        rules=(preserve,),
    )
    class DerivedPass(DerivationPass[ModelIR, ModelIR]):
        def run(self, ir: ModelIR, _context: PassContext) -> ModelIR:
            return ir

    assert DerivedPass.contract.input_type is ModelIR
    assert DerivedPass.contract.output_type is ModelIR
    assert DerivedPass.contract.required_bindings == frozenset({BindingAxis.WORKLOAD})
    assert DerivedPass.contract.rules[0].source_entity == "ModelOperation"
    assert DerivedPass.contract.rules[0].target_entity == "ModelOperation"


def test_analysis_products_are_atomic_and_context_addressed(model_ir: ModelIR) -> None:
    key = AnalysisKey("fixture", "shape")
    analyze = FunctionPass(
        _model_contract("fixture.analyze", produced_analyses=frozenset({key})),
        lambda ir, _context: PassResult(ir, (AnalysisProduct(key, FrozenDict({"rank": 2})),)),
    )
    consume = FunctionPass(
        _model_contract(
            "fixture.consume",
            required_analyses=frozenset({key}),
            preserved_analyses=frozenset({key}),
        ),
        lambda ir, context: ir if context.analysis(ir, key)["rank"] == 2 else None,
    )
    session = SynthesisSession()
    manager = PassManager()

    result = manager.require_run(PassPipeline.of(analyze, consume), model_ir, session=session)

    assert result.ir is model_ir
    assert tuple(item.pass_name for item in result.records) == ("fixture.analyze", "fixture.consume")
    assert tuple(item.ir for item in result.checkpoints) == (model_ir, model_ir)
    assert manager.analyses.get(model_ir.digest, key, session.fingerprint) == FrozenDict({"rank": 2})


def test_observer_can_profile_each_lowering_checkpoint(model_ir: ModelIR) -> None:
    class CaptureObserver(PassObserver):
        def __init__(self) -> None:
            self.checkpoints: list[PassCheckpoint] = []

        def inspect(self, checkpoint: PassCheckpoint, _context: object) -> None:
            self.checkpoints.append(checkpoint)

    observer = CaptureObserver()
    derivation_pass = FunctionPass(_model_contract("fixture.profiled"), lambda ir, _context: ir)

    result = PassManager(observers=(observer,)).require_run(
        PassPipeline.of(derivation_pass),
        model_ir,
        session=SynthesisSession(),
    )

    assert observer.checkpoints == list(result.checkpoints)
    assert observer.checkpoints[0].record.output_digest == model_ir.digest


def test_observer_failure_prevents_analysis_commit(model_ir: ModelIR) -> None:
    class RejectObserver(PassObserver):
        def inspect(self, _checkpoint: PassCheckpoint, _session: object) -> None:
            raise RuntimeError("profiler rejected checkpoint")

    key = AnalysisKey("fixture", "rejected")
    derivation_pass = FunctionPass(
        _model_contract("fixture.observed", produced_analyses=frozenset({key})),
        lambda ir, _context: PassResult(ir, (AnalysisProduct(key, FrozenDict({"ok": True})),)),
    )
    session = SynthesisSession()
    manager = PassManager(observers=(RejectObserver(),))

    with pytest.raises(PassExecutionError, match="RejectObserver"):
        manager.require_run(PassPipeline.of(derivation_pass), model_ir, session=session)
    assert not manager.analyses.has(model_ir.digest, key, session.fingerprint)


def test_missing_analysis_fails_before_pass_runs(model_ir: ModelIR) -> None:
    key = AnalysisKey("fixture", "missing")
    derivation_pass = FunctionPass(
        _model_contract("fixture.requires-analysis", required_analyses=frozenset({key})),
        lambda ir, _context: ir,
    )

    with pytest.raises(MissingAnalysisError, match="requires missing analyses"):
        PassManager().require_run(PassPipeline.of(derivation_pass), model_ir, session=SynthesisSession())


def test_analysis_store_rejects_noncanonical_product_atomically(model_ir: ModelIR) -> None:
    key = AnalysisKey("fixture", "bad")
    derivation_pass = FunctionPass(
        _model_contract("fixture.bad-analysis", produced_analyses=frozenset({key})),
        lambda ir, _context: PassResult(ir, (AnalysisProduct(key, object()),)),
    )
    manager = PassManager()
    session = SynthesisSession()

    with pytest.raises(PassContractError, match="not a canonical immutable value"):
        manager.require_run(PassPipeline.of(derivation_pass), model_ir, session=session)
    assert not manager.analyses.has(model_ir.digest, key, session.fingerprint)


def test_analysis_store_defensively_freezes_extension_data(model_ir: ModelIR) -> None:
    key = AnalysisKey("fixture", "frozen")
    source = {"values": [1]}
    derivation_pass = FunctionPass(
        _model_contract("fixture.freeze-analysis", produced_analyses=frozenset({key})),
        lambda ir, _context: PassResult(ir, (AnalysisProduct(key, source),)),
    )
    manager = PassManager()
    session = SynthesisSession()

    manager.require_run(PassPipeline.of(derivation_pass), model_ir, session=session)
    source["values"].append(2)

    assert manager.analyses.get(model_ir.digest, key, session.fingerprint) == FrozenDict({"values": (1,)})


def test_required_binding_is_enforced(model_ir: ModelIR) -> None:
    derivation_pass = FunctionPass(
        _model_contract("fixture.target-bound", required_bindings=frozenset({BindingAxis.TARGET})),
        lambda ir, _context: ir,
    )

    with pytest.raises(MissingBindingError, match="target"):
        PassManager().require_run(PassPipeline.of(derivation_pass), model_ir, session=SynthesisSession())


def test_pipeline_rejects_declared_type_mismatch() -> None:
    first = FunctionPass(
        PassContract.create("fixture.distribute", ModelIR, DistributedTaskIR),
        lambda ir, _context: ir,  # runtime is irrelevant: construction must fail first
    )
    second = FunctionPass(_model_contract("fixture.normalize"), lambda ir, _context: ir)

    with pytest.raises(PassContractError, match="pipeline type mismatch"):
        PassPipeline.of(first, second)


def test_cross_stage_pass_without_rules_is_rejected(
    model_ir: ModelIR,
    distributed_ir: DistributedTaskIR,
) -> None:
    derivation_pass = FunctionPass(
        PassContract.create("fixture.missing-rules-cross-stage", ModelIR, DistributedTaskIR),
        lambda _ir, _context: distributed_ir,
    )

    with pytest.raises(PassContractError, match="must declare executable lineage rules"):
        PassManager().require_run(PassPipeline.of(derivation_pass), model_ir, session=SynthesisSession())


def test_immutable_pass_input_mutation_is_detected(model_ir: ModelIR) -> None:
    def mutate(ir: ModelIR, _context: object) -> ModelIR:
        object.__setattr__(ir, "name", "illegally-mutated")
        return ir

    derivation_pass = FunctionPass(_model_contract("fixture.illegal-mutation"), mutate)

    with pytest.raises(PassContractError, match="mutated its input"):
        PassManager().require_run(PassPipeline.of(derivation_pass), model_ir, session=SynthesisSession())


def test_transactional_pass_cannot_mutate_caller_snapshot(model_ir: ModelIR) -> None:
    original_digest = model_ir.digest

    def mutate_copy(ir: ModelIR, _context: object) -> ModelIR:
        object.__setattr__(ir, "name", "transactional-copy")
        object.__setattr__(
            ir,
            "header",
            make_header(ir.SCHEMA_NAME, ir.SCHEMA_VERSION, parent_digests=(original_digest,)),
        )
        return ir

    def normal_form(ir: ModelIR, _session: SynthesisSession) -> ModelIR:
        return replace(
            ir,
            name="transactional-copy",
            header=make_header(ir.SCHEMA_NAME, ir.SCHEMA_VERSION, parent_digests=(ir.digest,)),
        )

    derivation_pass = FunctionPass(
        _model_contract(
            "fixture.transaction",
            mutation_model=MutationModel.TRANSACTIONAL,
            normalizer=normal_form,
        ),
        mutate_copy,
    )

    result = PassManager().require_run(PassPipeline.of(derivation_pass), model_ir, session=SynthesisSession())

    assert model_ir.name == "fixture-model"
    assert model_ir.digest == original_digest
    assert result.ir.name == "transactional-copy"
    report = result.records[0].transition_report
    assert report.status.value == "canonical_conformant"
    assert report.canonical_conformant
    assert report.verified_relations == 0


def test_same_stage_rewrite_requires_a_law(model_ir: ModelIR) -> None:
    source_digest = model_ir.digest
    derivation_pass = FunctionPass(
        _model_contract("fixture.implicit-lawless-rewrite"),
        lambda ir, _context: replace(
            ir,
            name="rewritten",
            header=make_header(ir.SCHEMA_NAME, ir.SCHEMA_VERSION, parent_digests=(source_digest,)),
        ),
    )

    with pytest.raises(PassContractError, match="without a declared normal form or rules"):
        PassManager().require_run(PassPipeline.of(derivation_pass), model_ir, session=SynthesisSession())


def test_run_returns_expected_contract_failure_as_checked_diagnostics(model_ir: ModelIR) -> None:
    source_digest = model_ir.digest
    derivation_pass = FunctionPass(
        _model_contract("fixture.checked-failure"),
        lambda ir, _context: replace(
            ir,
            name="rewritten",
            header=make_header(ir.SCHEMA_NAME, ir.SCHEMA_VERSION, parent_digests=(source_digest,)),
        ),
    )

    result = PassManager().run(PassPipeline.of(derivation_pass), model_ir, session=SynthesisSession())

    assert isinstance(result, Err)
    assert result.error.errors[0].code == "pass.contract"
    assert "without a declared normal form or rules" in result.error.errors[0].message


def test_rewrite_must_retain_parent_digest(model_ir: ModelIR) -> None:
    derivation_pass = FunctionPass(
        _model_contract("fixture.bad-lineage"),
        lambda ir, _context: replace(ir, name="rewritten-without-lineage"),
    )

    with pytest.raises(PassContractError, match="without retaining its input digest"):
        PassManager().require_run(PassPipeline.of(derivation_pass), model_ir, session=SynthesisSession())


def test_unexpected_pass_failure_is_wrapped(model_ir: ModelIR) -> None:
    def fail(_ir: ModelIR, _context: object) -> ModelIR:
        raise RuntimeError("implementation bug")

    derivation_pass = FunctionPass(_model_contract("fixture.failure"), fail)

    with pytest.raises(PassExecutionError, match="fixture.failure"):
        PassManager().require_run(PassPipeline.of(derivation_pass), model_ir, session=SynthesisSession())


def test_determinism_verify_rejects_random_stable_ids(model_ir: ModelIR) -> None:
    input_digest = model_ir.digest

    def nondeterministic(ir: ModelIR, _context: object) -> ModelIR:
        operation = replace(ir.operations[0], id=NodeId.new())
        return replace(
            ir,
            operations=(operation,),
            header=make_header(ir.SCHEMA_NAME, ir.SCHEMA_VERSION, parent_digests=(input_digest,)),
        )

    derivation_pass = FunctionPass(_model_contract("fixture.nondeterministic"), nondeterministic)

    with pytest.raises(PassContractError, match="failed deterministic replay"):
        PassManager(determinism=DeterminismPolicy.VERIFY).require_run(
            PassPipeline.of(derivation_pass),
            model_ir,
            session=SynthesisSession(seed=7),
        )


def test_determinism_verify_invokes_observers_only_once(model_ir: ModelIR) -> None:
    class CountObserver(PassObserver):
        def __init__(self) -> None:
            self.count = 0

        def inspect(self, _checkpoint: PassCheckpoint, _session: object) -> None:
            self.count += 1

    observer = CountObserver()
    derivation_pass = FunctionPass(_model_contract("fixture.deterministic"), lambda ir, _context: ir)

    PassManager(observers=(observer,), determinism=DeterminismPolicy.VERIFY).require_run(
        PassPipeline.of(derivation_pass),
        model_ir,
        session=SynthesisSession(seed=7),
    )

    assert observer.count == 1

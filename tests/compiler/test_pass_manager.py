from __future__ import annotations

from dataclasses import replace

import pytest

from blueprinting.compiler import (
    BindingAxis,
    CompilationSession,
    FrozenDict,
    MissingAnalysisError,
    MissingBindingError,
    PassContractError,
)
from blueprinting.compiler.errors import PassExecutionError
from blueprinting.compiler.ir import DistributedTaskIR, ModelIR
from blueprinting.compiler.ir.common import make_header
from blueprinting.compiler.passes import (
    AnalysisKey,
    AnalysisProduct,
    FunctionPass,
    MutationModel,
    PassCheckpoint,
    PassContract,
    PassManager,
    PassObserver,
    PassPipeline,
    PassResult,
)


def _model_contract(name: str, **options: object) -> PassContract:
    return PassContract.create(name, ModelIR, ModelIR, **options)


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
    session = CompilationSession()
    manager = PassManager()

    result = manager.run(PassPipeline.of(analyze, consume), model_ir, session=session)

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
    compiler_pass = FunctionPass(_model_contract("fixture.profiled"), lambda ir, _context: ir)

    result = PassManager(observers=(observer,)).run(
        PassPipeline.of(compiler_pass),
        model_ir,
        session=CompilationSession(),
    )

    assert observer.checkpoints == list(result.checkpoints)
    assert observer.checkpoints[0].record.output_digest == model_ir.digest


def test_observer_failure_prevents_analysis_commit(model_ir: ModelIR) -> None:
    class RejectObserver(PassObserver):
        def inspect(self, _checkpoint: PassCheckpoint, _session: object) -> None:
            raise RuntimeError("profiler rejected checkpoint")

    key = AnalysisKey("fixture", "rejected")
    compiler_pass = FunctionPass(
        _model_contract("fixture.observed", produced_analyses=frozenset({key})),
        lambda ir, _context: PassResult(ir, (AnalysisProduct(key, FrozenDict({"ok": True})),)),
    )
    session = CompilationSession()
    manager = PassManager(observers=(RejectObserver(),))

    with pytest.raises(PassExecutionError, match="RejectObserver"):
        manager.run(PassPipeline.of(compiler_pass), model_ir, session=session)
    assert not manager.analyses.has(model_ir.digest, key, session.fingerprint)


def test_missing_analysis_fails_before_pass_runs(model_ir: ModelIR) -> None:
    key = AnalysisKey("fixture", "missing")
    compiler_pass = FunctionPass(
        _model_contract("fixture.requires-analysis", required_analyses=frozenset({key})),
        lambda ir, _context: ir,
    )

    with pytest.raises(MissingAnalysisError, match="requires missing analyses"):
        PassManager().run(PassPipeline.of(compiler_pass), model_ir, session=CompilationSession())


def test_analysis_store_rejects_noncanonical_product_atomically(model_ir: ModelIR) -> None:
    key = AnalysisKey("fixture", "bad")
    compiler_pass = FunctionPass(
        _model_contract("fixture.bad-analysis", produced_analyses=frozenset({key})),
        lambda ir, _context: PassResult(ir, (AnalysisProduct(key, object()),)),
    )
    manager = PassManager()
    session = CompilationSession()

    with pytest.raises(PassContractError, match="not a canonical immutable value"):
        manager.run(PassPipeline.of(compiler_pass), model_ir, session=session)
    assert not manager.analyses.has(model_ir.digest, key, session.fingerprint)


def test_analysis_store_defensively_freezes_extension_data(model_ir: ModelIR) -> None:
    key = AnalysisKey("fixture", "frozen")
    source = {"values": [1]}
    compiler_pass = FunctionPass(
        _model_contract("fixture.freeze-analysis", produced_analyses=frozenset({key})),
        lambda ir, _context: PassResult(ir, (AnalysisProduct(key, source),)),
    )
    manager = PassManager()
    session = CompilationSession()

    manager.run(PassPipeline.of(compiler_pass), model_ir, session=session)
    source["values"].append(2)

    assert manager.analyses.get(model_ir.digest, key, session.fingerprint) == FrozenDict({"values": (1,)})


def test_required_binding_is_enforced(model_ir: ModelIR) -> None:
    compiler_pass = FunctionPass(
        _model_contract("fixture.target-bound", required_bindings=frozenset({BindingAxis.TARGET})),
        lambda ir, _context: ir,
    )

    with pytest.raises(MissingBindingError, match="target"):
        PassManager().run(PassPipeline.of(compiler_pass), model_ir, session=CompilationSession())


def test_pipeline_rejects_declared_type_mismatch() -> None:
    first = FunctionPass(
        PassContract.create("fixture.distribute", ModelIR, DistributedTaskIR),
        lambda ir, _context: ir,  # runtime is irrelevant: construction must fail first
    )
    second = FunctionPass(_model_contract("fixture.normalize"), lambda ir, _context: ir)

    with pytest.raises(PassContractError, match="pipeline type mismatch"):
        PassPipeline.of(first, second)


def test_immutable_pass_input_mutation_is_detected(model_ir: ModelIR) -> None:
    def mutate(ir: ModelIR, _context: object) -> ModelIR:
        object.__setattr__(ir, "name", "illegally-mutated")
        return ir

    compiler_pass = FunctionPass(_model_contract("fixture.illegal-mutation"), mutate)

    with pytest.raises(PassContractError, match="mutated its input"):
        PassManager().run(PassPipeline.of(compiler_pass), model_ir, session=CompilationSession())


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

    compiler_pass = FunctionPass(
        _model_contract("fixture.transaction", mutation_model=MutationModel.TRANSACTIONAL),
        mutate_copy,
    )

    result = PassManager().run(PassPipeline.of(compiler_pass), model_ir, session=CompilationSession())

    assert model_ir.name == "fixture-model"
    assert model_ir.digest == original_digest
    assert result.ir.name == "transactional-copy"


def test_rewrite_must_retain_parent_digest(model_ir: ModelIR) -> None:
    compiler_pass = FunctionPass(
        _model_contract("fixture.bad-lineage"),
        lambda ir, _context: replace(ir, name="rewritten-without-lineage"),
    )

    with pytest.raises(PassContractError, match="without retaining its input digest"):
        PassManager().run(PassPipeline.of(compiler_pass), model_ir, session=CompilationSession())


def test_unexpected_pass_failure_is_wrapped(model_ir: ModelIR) -> None:
    def fail(_ir: ModelIR, _context: object) -> ModelIR:
        raise RuntimeError("implementation bug")

    compiler_pass = FunctionPass(_model_contract("fixture.failure"), fail)

    with pytest.raises(PassExecutionError, match="fixture.failure"):
        PassManager().run(PassPipeline.of(compiler_pass), model_ir, session=CompilationSession())

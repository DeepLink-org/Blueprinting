from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from blueprinting.analysis.cost_model import CalibrationMode, estimate_iteration
from blueprinting.mapping import NetworkTierBinding, TransformerTrainingMappingSpec
from blueprinting.synthesizer.dialects.transformer import EngineKind, TrainingPhase, TransformerTrainingPlanTaskSemantic
from blueprinting.synthesizer.frontend import build_transformer_model_ir, synthesis_session_for
from blueprinting.synthesizer.passes import PassManager, PassPipeline
from blueprinting.synthesizer.stages.distributed.passes import DistributeTransformerTrainingPass
from blueprinting.synthesizer.stages.portable_plan.passes import PlanTransformerTrainingPass
from blueprinting.system import SystemProfile
from blueprinting.validation import calculon as calculon_validation
from blueprinting.validation import discover_seqsel_tab5_cases, run_calculon_experiment
from blueprinting.workload import TransformerModelSpec, TransformerTrainingWorkloadSpec

ROOT = Path(__file__).resolve().parents[2]


def _json(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _derive(model_name: str, mode: str):
    model_data = _json(ROOT / "data" / "models" / f"{model_name}.json")
    execution_data = _json(ROOT / "data" / "validation" / "seqsel" / "tab5" / f"{model_name}_{mode}.json")
    model = TransformerModelSpec.from_mapping(model_name, model_data)
    workload = TransformerTrainingWorkloadSpec.from_mapping(execution_data)
    mapping = TransformerTrainingMappingSpec.from_mapping(execution_data)
    network_binding = NetworkTierBinding.from_mapping(execution_data)
    source = build_transformer_model_ir(model, datatype=workload.datatype)
    result = PassManager().require_run(
        PassPipeline.of(DistributeTransformerTrainingPass(), PlanTransformerTrainingPass()),
        source,
        session=synthesis_session_for(model, workload, mapping),
    )
    return model, workload, mapping, network_binding, source, result


def test_transformer_lowering_produces_auditable_ir_checkpoints():
    _, _, _, _, source, result = _derive("gpt3-175B", "seqsel")

    assert source.require_valid() is None
    assert tuple(record.pass_name for record in result.records) == (
        "transformer-distribute",
        "transformer-plan-work",
    )
    assert tuple(checkpoint.ir.header.schema_name for checkpoint in result.checkpoints) == (
        "blueprinting.distributed-task",
        "blueprinting.portable-plan",
    )
    assert result.ir.require_valid() is None
    assert all("invocation" not in task.attributes for task in result.ir.tasks)
    assert all("network_tier" not in resource.capabilities for task in result.ir.tasks for resource in task.resources)


def test_selective_recompute_is_structural_and_linear_gradients_are_derived():
    _, _, _, _, _, result = _derive("gpt3-175B", "seqsel")
    recomputed_layers = {
        task.semantic.source_layer
        for task in result.ir.tasks
        if isinstance(task.semantic, TransformerTrainingPlanTaskSemantic)
        and task.semantic.phase is TrainingPhase.RECOMPUTE
    }

    assert recomputed_layers == {
        "attention.query_key",
        "attention.softmax",
        "attention.probability_dropout",
    }
    query = {
        task.semantic.phase: task.workload.operations
        for task in result.ir.tasks
        if isinstance(task.semantic, TransformerTrainingPlanTaskSemantic)
        and task.semantic.source_layer == "attention.query"
        and task.semantic.engine is EngineKind.MATRIX
    }
    assert query[TrainingPhase.FORWARD] == query[TrainingPhase.ACTIVATION_GRADIENT]
    assert query[TrainingPhase.FORWARD] == query[TrainingPhase.WEIGHT_GRADIENT]


def test_hardware_evidence_is_shared_and_does_not_change_workload():
    _, workload, _, network_binding, _, result = _derive("gpt3-175B", "full")
    hardware = SystemProfile.from_mapping(
        "a100_80g",
        _json(ROOT / "data" / "systems" / "a100_80g.json"),
        datatype=workload.datatype,
    )
    digests_before = tuple(task.workload for task in result.ir.tasks)

    peak = estimate_iteration(
        result.ir,
        hardware,
        CalibrationMode.PEAK_ONLY,
        network_binding=network_binding,
    )
    calibrated = estimate_iteration(
        result.ir,
        hardware,
        CalibrationMode.SYSTEM_EVIDENCE,
        network_binding=network_binding,
    )
    alternate_network = NetworkTierBinding(
        tensor_parallel=1,
        pipeline_parallel=network_binding.pipeline_parallel,
        data_parallel=network_binding.data_parallel,
    )
    alternate = estimate_iteration(
        result.ir,
        hardware,
        CalibrationMode.SYSTEM_EVIDENCE,
        network_binding=alternate_network,
    )

    assert peak.total < calibrated.total
    assert alternate.tensor_parallel != calibrated.tensor_parallel
    assert tuple(task.workload for task in result.ir.tasks) == digests_before
    assert result.ir.attributes.get("duration") is None


@pytest.mark.parametrize("case_index", range(8))
def test_seqsel_cases_match_calculon_without_case_fitting(case_index: int):
    case = discover_seqsel_tab5_cases(ROOT / "data")[case_index]
    report = run_calculon_experiment((case,))
    result = report.cases[0]

    assert max(abs(metric.relative_error_percent) for metric in result.workload.values()) < 1e-9
    assert abs(result.calibrated_error_percent) < 1e-9
    assert abs(result.calibrated.memory.total - result.calculon_stats["proc_mem_tier1_cap_req"]) < 1
    assert report.calibration_policy["fit_against_case_outputs"] is False
    assert "Calculon duration" in report.calibration_policy["forbidden_inputs"]


def test_explicit_recompute_does_not_copy_calculon_prefix_counter():
    case = discover_seqsel_tab5_cases(ROOT / "data")[0]
    result = run_calculon_experiment((case,)).cases[0]
    audit = result.to_dict()["recompute_counter_audit"]

    assert audit["derived_explicit_operations"] < audit["calculon_block_re_flops"]
    assert result.calibrated.recompute == pytest.approx(result.calculon_stats["recompute_time"], rel=1e-12)


def test_alignment_report_freezes_oracle_and_input_provenance():
    case = discover_seqsel_tab5_cases(ROOT / "data")[0]
    report = run_calculon_experiment((case,))
    payload = report.to_dict()

    assert report.schema == "blueprinting.calculon-calibration-experiment.v0"
    assert report.oracle["name"] == "Calculon"
    assert report.oracle["package_version"] == "0.1.0"
    assert len(report.oracle["source_digest"]) == 64
    assert payload["paper_baseline"]["source"] == "https://arxiv.org/abs/2205.05198"
    assert payload["cases"][0]["inputs"] == {
        "model": {"file": case.model_path.name, "sha256": _sha256(case.model_path)},
        "execution": {"file": case.execution_path.name, "sha256": _sha256(case.execution_path)},
        "system": {"file": case.system_path.name, "sha256": _sha256(case.system_path)},
    }


def test_oracle_runs_only_after_both_blueprinting_estimates(monkeypatch):
    events = []
    estimate = calculon_validation.estimate_iteration
    oracle = calculon_validation._run_calculon

    def record_estimate(*args, **kwargs):
        events.append(f"estimate:{args[2].value}")
        return estimate(*args, **kwargs)

    def record_oracle(*args, **kwargs):
        events.append("oracle")
        return oracle(*args, **kwargs)

    monkeypatch.setattr(calculon_validation, "estimate_iteration", record_estimate)
    monkeypatch.setattr(calculon_validation, "_run_calculon", record_oracle)
    run_calculon_experiment((discover_seqsel_tab5_cases(ROOT / "data")[0],))

    assert events == ["estimate:peak_only", "estimate:system_evidence", "oracle"]


def test_committed_alignment_artifact_and_bilingual_report_match_current_experiment():
    report = run_calculon_experiment(discover_seqsel_tab5_cases(ROOT / "data"))
    committed = _json(ROOT / "examples" / "calculon_calibration_result.json")

    assert committed == report.to_dict()
    for locale in ("en", "zh"):
        document = (ROOT / "docs" / "experiments" / f"calculon-calibration.{locale}.md").read_text()
        for expected in (
            report.schema,
            report.oracle["source_digest"],
            "12.99%",
            "3.65%",
            "8.87%",
            "152",
            "1e-9%",
        ):
            assert expected in document

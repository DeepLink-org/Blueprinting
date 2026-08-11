from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprinting.analysis import (
    CalibrationMode,
    CostResolver,
    CostSubject,
    EstimateMethod,
    EvidenceProvenance,
    PerformanceDatabase,
    PerformanceDatabaseProvider,
    PerformanceRecord,
    RooflineCostProvider,
)
from blueprinting.application import BlueprintingService, InferenceAnalysisDraft
from blueprinting.schema.frozen import FrozenDict
from blueprinting.system import SystemProfile

ROOT = Path(__file__).resolve().parents[2]


def _json(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _draft(*, generated_tokens: int = 4) -> InferenceAnalysisDraft:
    return InferenceAnalysisDraft.from_mappings(
        model_name="gpt3-175B",
        model_data=_json(ROOT / "data" / "models" / "gpt3-175B.json"),
        execution_name="static-inference",
        execution_data={
            "tensor_par": 8,
            "pipeline_par": 1,
            "replicas": 1,
            "datatype": "float16",
            "tensor_par_net": 0,
            "pipeline_par_net": 0,
        },
        request_data={
            "batch_size": 1,
            "prompt_tokens": 128,
            "generated_tokens": generated_tokens,
        },
        hardware_name="a100_80g",
        hardware_data=_json(ROOT / "data" / "systems" / "a100_80g.json"),
    )


def test_service_composes_prefill_and_each_decode_context_without_a_scheduler():
    outcome = BlueprintingService().analyze_inference(_draft(generated_tokens=4))

    assert outcome.ok
    assert outcome.report is not None
    report = outcome.report
    assert report.prefill_seconds > 0
    assert report.mean_decode_step_seconds > 0
    assert report.model_execution_seconds == pytest.approx(
        report.prefill_seconds + sum(step.seconds for step in report.decode_steps)
    )
    assert tuple(step.context_tokens for step in report.decode_steps) == (129, 130, 131)
    assert report.memory["kv_cache"] > 0
    assert report.memory["max_kv_cache"] >= report.memory["kv_cache"]
    assert report.stages[-1].stage == "decode.portable"
    assert all(stage.valid for stage in report.stages)


def test_single_generated_token_stops_after_prefill():
    outcome = BlueprintingService().analyze_inference(_draft(generated_tokens=1))

    assert outcome.ok
    assert outcome.report is not None
    assert outcome.report.decode_steps == ()
    assert outcome.report.mean_decode_step_seconds == 0
    assert outcome.report.model_execution_seconds == outcome.report.prefill_seconds


def test_service_accepts_canonical_inference_mapping_names():
    preset = _draft(generated_tokens=1)
    execution = dict(preset.execution_data.items())
    execution["tensor_parallel"] = execution.pop("tensor_par")
    execution["pipeline_parallel"] = execution.pop("pipeline_par")
    canonical = InferenceAnalysisDraft.from_mappings(
        model_name=preset.model_name,
        model_data=dict(preset.model_data.items()),
        execution_name=preset.execution_name,
        execution_data=execution,
        request_data=dict(preset.request_data.items()),
        hardware_name=preset.hardware_name,
        hardware_data=dict(preset.hardware_data.items()),
    )

    outcome = BlueprintingService().analyze_inference(canonical)

    assert outcome.ok
    assert outcome.report is not None
    assert outcome.report.world_size == 8


def test_request_past_model_context_returns_a_structured_diagnostic():
    draft = _draft(generated_tokens=4)
    request = dict(draft.request_data.items())
    request["prompt_tokens"] = 2048
    too_long = InferenceAnalysisDraft.from_mappings(
        model_name=draft.model_name,
        model_data=dict(draft.model_data.items()),
        execution_name=draft.execution_name,
        execution_data=dict(draft.execution_data.items()),
        request_data=request,
        hardware_name=draft.hardware_name,
        hardware_data=dict(draft.hardware_data.items()),
    )

    outcome = BlueprintingService().analyze_inference(too_long)

    assert not outcome.ok
    assert outcome.diagnostics[0].code == "inference.configuration.invalid"
    assert "cannot exceed" in outcome.diagnostics[0].message


def test_service_routes_resolver_evidence_and_uncertainty_to_task_reports():
    draft = _draft(generated_tokens=2)
    hardware = SystemProfile.from_mapping(
        draft.hardware_name,
        dict(draft.hardware_data.items()),
        datatype="float16",
    )
    provenance = EvidenceProvenance(
        source="fixture-simulator",
        source_revision="sim-r1",
        importer="fixture-importer-v0",
        data_digest="fixture-data",
        method=EstimateMethod.SIMULATED,
    )
    records = tuple(
        PerformanceRecord(
            record_id=f"attention-{index}",
            subject=CostSubject.OPERATOR,
            operation="attention_core",
            hardware=hardware.name,
            datatype=hardware.datatype,
            seconds=seconds,
            selector=FrozenDict({"semantic_operation": "attention_core"}),
            provenance=provenance,
        )
        for index, seconds in enumerate((0.001, 0.003), start=1)
    )
    database_provider = PerformanceDatabaseProvider(PerformanceDatabase("application", records))
    resolver = CostResolver(
        (
            database_provider,
            RooflineCostProvider(hardware, mode=CalibrationMode.SYSTEM_EVIDENCE),
        )
    )

    outcome = BlueprintingService(inference_cost_resolver=resolver).analyze_inference(draft)

    assert outcome.ok
    assert outcome.report is not None
    attention = next(task for task in outcome.report.tasks if task.source_layer == "attention.core")
    fallback = next(task for task in outcome.report.tasks if task.source_layer == "attention.input_norm")
    assert attention.evidence_provider == database_provider.name
    assert attention.evidence_source_revision == "sim-r1"
    assert attention.evidence_record_ids == ("attention-1", "attention-2")
    assert attention.evidence_method == "simulated"
    assert attention.evidence_uncertainty["sample_count"] == 2
    assert attention.evidence_uncertainty["standard_deviation_seconds"] == pytest.approx(0.001)
    assert attention.evidence_assumptions
    assert fallback.evidence_method == "analytical"
    assert outcome.report.evidence["cost_resolver_revision"] == resolver.revision

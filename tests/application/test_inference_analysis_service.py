from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprinting.application import BlueprintingService, InferenceAnalysisDraft

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

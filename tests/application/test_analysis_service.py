from __future__ import annotations

import pickle

from blueprinting.analysis import CalibrationMode
from blueprinting.application import AnalysisDraft, BlueprintingService, SweepRequest
from blueprinting.synthesizer.frontend import build_transformer_model_ir
from blueprinting.synthesizer.frozen import FrozenDict
from blueprinting.workbench import default_catalog
from blueprinting.workload import TransformerModelSpec


def _draft(**execution_changes) -> AnalysisDraft:
    execution = {
        "num_procs": 8,
        "tensor_par": 2,
        "pipeline_par": 2,
        "data_par": 2,
        "tensor_par_net": 0,
        "pipeline_par_net": 1,
        "data_par_net": 1,
        "batch_size": 8,
        "microbatch_size": 1,
        "datatype": "float16",
        "fused_activation": False,
        "attention_type": "multihead",
        "activation_recompute": "none",
        "pipeline_interleaving": 1,
        "optimizer_sharding": False,
        "tensor_par_comm_type": "ar",
        "tensor_par_overlap": "none",
        "seq_par_ag_redo": False,
        "data_par_overlap": False,
        "weight_offload": False,
        "activations_offload": False,
        "optimizer_offload": False,
        "training": True,
    }
    execution.update(execution_changes)
    catalog = default_catalog()
    return AnalysisDraft.from_mappings(
        model_name="fixture-transformer",
        model_data={
            "hidden": 128,
            "feedforward": 512,
            "seq_size": 32,
            "attn_heads": 8,
            "attn_size": 16,
            "num_blocks": 4,
        },
        execution_name="fixture-execution",
        execution_data=execution,
        hardware_name="a100_80g",
        hardware_data=catalog.load("systems", "a100_80g.json"),
        calibration_mode=CalibrationMode.SYSTEM_EVIDENCE,
    )


def test_analysis_service_is_the_complete_client_boundary() -> None:
    outcome = BlueprintingService().analyze(_draft(num_procs=999))

    assert outcome.ok
    assert outcome.report is not None
    report = outcome.report
    assert report.world_size == 8
    assert report.total_seconds > 0
    assert report.total_tokens_per_second > 0
    assert report.memory["total"] > 0
    assert report.plan_digest == report.stages[-1].digest
    assert tuple(stage.stage for stage in report.stages) == ("model", "distributed", "portable")
    assert all(stage.valid for stage in report.stages)
    assert report.tasks
    assert report.evidence_revision == report.evidence["revision"]


def test_expected_configuration_failure_is_a_diagnostic() -> None:
    outcome = BlueprintingService().analyze(_draft(batch_size=7))

    assert not outcome.ok
    assert outcome.report is None
    assert outcome.diagnostics[0].code == "configuration.invalid"
    assert "divisible" in outcome.diagnostics[0].message


def test_sweep_preserves_invalid_candidates_and_marks_pareto() -> None:
    service = BlueprintingService()
    report = service.sweep(
        SweepRequest(
            base=_draft(),
            tensor_parallel=(2, 3),
            pipeline_parallel=(2,),
            data_parallel=(2,),
        )
    )

    assert len(report.cases) == 2
    assert {case.status for case in report.cases} == {"success", "invalid"}
    assert sum(case.pareto for case in report.cases) == 1
    invalid = next(case for case in report.cases if case.status == "invalid")
    assert invalid.diagnostics


def test_model_ir_datatype_is_not_hard_coded() -> None:
    model = TransformerModelSpec("dtype", 128, 512, 32, 8, 16, 2)

    ir = build_transformer_model_ir(model, datatype="bfloat16")

    assert {value.type.dtype for value in ir.values} == {"bfloat16"}


def test_frozen_configuration_is_safe_for_ui_cache_round_trip() -> None:
    value = FrozenDict({"nested": {"values": [1, 2, 3]}})

    restored = pickle.loads(pickle.dumps(value))

    assert restored == value
    assert hash(restored) == hash(value)

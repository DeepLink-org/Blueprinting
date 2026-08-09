from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprinting.analysis import (
    InferenceCostProvider,
    InferenceEvidenceQuery,
    VidurProfileBaseline,
    estimate_inference_phase,
)
from blueprinting.mapping import NetworkTierBinding, TransformerInferenceMappingSpec
from blueprinting.synthesizer.bindings import InferencePhase
from blueprinting.synthesizer.frontend import (
    build_transformer_inference_model_ir,
    inference_synthesis_session_for,
)
from blueprinting.synthesizer.lowering import DistributeTransformerInferencePass, PlanTransformerInferencePass
from blueprinting.synthesizer.passes import PassManager, PassPipeline
from blueprinting.system import SystemProfile
from blueprinting.validation import (
    VidurExperimentCase,
    compare_inference_phase_to_vidur,
    run_vidur_experiment,
)
from blueprinting.workload import (
    TransformerInferenceRequestSpec,
    TransformerModelSpec,
)

ROOT = Path(__file__).resolve().parents[2]


def _model() -> TransformerModelSpec:
    return TransformerModelSpec(
        name="fixture-inference",
        hidden_size=64,
        feedforward_size=256,
        sequence_length=512,
        attention_heads=8,
        attention_head_size=8,
        block_count=8,
    )


def _execution() -> TransformerInferenceMappingSpec:
    return TransformerInferenceMappingSpec(
        tensor_parallel=2,
        pipeline_parallel=2,
        replicas=1,
    )


def _derive(phase: InferencePhase, context_tokens: int):
    model = _model()
    execution = _execution()
    source = build_transformer_inference_model_ir(model)
    result = PassManager().run(
        PassPipeline.of(DistributeTransformerInferencePass(), PlanTransformerInferencePass()),
        source,
        session=inference_synthesis_session_for(
            model,
            execution,
            phase=phase,
            batch_size=3,
            context_tokens=context_tokens,
            datatype="float16",
        ),
    )
    return source, result.ir


def _task(plan, primitive: str):
    return next(task for task in plan.tasks if task.workload.attributes.get("primitive") == primitive)


@pytest.mark.parametrize("phase", tuple(InferencePhase))
def test_inference_lowering_has_valid_auditable_phase_plans(phase: InferencePhase):
    source, plan = _derive(phase, 64)

    assert source.verify().ok
    assert plan.verify().ok
    assert plan.attributes["inference_phase"] is phase
    assert all("invocation" not in task.attributes for task in plan.tasks)
    assert all(task.workload.attributes.get("phase") == phase.value for task in plan.tasks)
    assert all("network_tier" not in task.workload.attributes for task in plan.tasks)
    assert all("network_tier" not in resource.capabilities for task in plan.tasks for resource in task.resources)
    assert {buffer.attributes.get("semantic") for buffer in plan.buffers} >= {
        "block_weights",
        "block_working_upper_bound",
        "kv_cache",
    }
    assert sum(task.workload.message_bytes > 0 for task in plan.tasks) == 2


def test_prefill_attention_is_quadratic_and_decode_attention_is_linear_in_context():
    _, prefill_32 = _derive(InferencePhase.PREFILL, 32)
    _, prefill_64 = _derive(InferencePhase.PREFILL, 64)
    _, decode_32 = _derive(InferencePhase.DECODE, 32)
    _, decode_64 = _derive(InferencePhase.DECODE, 64)

    assert (
        _task(prefill_64, "attention_core").workload.operations
        == 4 * _task(prefill_32, "attention_core").workload.operations
    )
    assert (
        _task(decode_64, "attention_core").workload.operations
        == 2 * _task(decode_32, "attention_core").workload.operations
    )


def test_kv_cache_capacity_is_derived_from_shape_not_a_correction_factor():
    _, plan = _derive(InferencePhase.DECODE, 96)
    kv_buffer = next(buffer for buffer in plan.buffers if buffer.attributes.get("semantic") == "kv_cache")
    workspace = next(
        buffer for buffer in plan.buffers if buffer.attributes.get("semantic") == "block_working_upper_bound"
    )

    assert kv_buffer.size_bytes == 2 * 3 * 96 * (64 // 2) * 2
    assert workspace.size_bytes > 0


def test_network_tier_binding_changes_cost_without_changing_portable_plan():
    _, plan = _derive(InferencePhase.DECODE, 96)
    hardware = SystemProfile.from_mapping(
        "fixture-hardware",
        json.loads((ROOT / "data" / "systems" / "a100_80g.json").read_text(encoding="utf-8")),
        datatype="float16",
    )
    digest = plan.digest

    fast = estimate_inference_phase(plan, hardware, network_binding=NetworkTierBinding(tensor_parallel=0))
    slow = estimate_inference_phase(plan, hardware, network_binding=NetworkTierBinding(tensor_parallel=1))

    assert fast.total_seconds != slow.total_seconds
    assert plan.digest == digest


def test_request_semantics_count_prefill_as_the_first_output_token():
    one = TransformerInferenceRequestSpec(batch_size=1, prompt_tokens=64, generated_tokens=1)
    four = TransformerInferenceRequestSpec(batch_size=1, prompt_tokens=64, generated_tokens=4)

    assert one.decode_iterations == 0
    assert tuple(one.decode_contexts()) == ()
    assert four.decode_iterations == 3
    assert tuple(four.decode_contexts()) == (65, 66, 67)


def test_invalid_mapping_is_rejected_before_lowering():
    model = _model()
    invalid = TransformerInferenceMappingSpec(
        tensor_parallel=3,
        pipeline_parallel=1,
        replicas=1,
    )

    with pytest.raises(ValueError, match="hidden_size must be divisible"):
        invalid.validate_model(model)


def test_vidur_adapter_uses_only_exact_shape_matches(tmp_path: Path):
    attention = tmp_path / "attention.csv"
    compute = tmp_path / "mlp.csv"
    attention.write_text(
        "n_embd,n_q_head,n_kv_head,num_tensor_parallel_workers,block_size,max_model_len,"
        "batch_size,prefill_chunk_size,"
        "kv_cache_size,is_prefill,attention_backend,time_stats.attn_prefill.median,"
        "time_stats.attn_decode.median\n"
        "64,8,8,2,16,512,3,64,0,True,AttentionBackend.FLASH_ATTENTION,1.5,\n"
        "64,8,8,2,16,512,3.5,0,95,False,AttentionBackend.FLASH_ATTENTION,,99.0\n"
        "64,8,8,2,16,512,3,0,95,False,AttentionBackend.FLASH_ATTENTION,,0.25\n",
        encoding="utf-8",
    )
    compute.write_text(
        "n_embd,n_expanded_embd,n_head,n_kv_head,num_tensor_parallel_workers,num_tokens,use_gated_mlp,"
        "time_stats.attn_pre_proj.median\n"
        "64,256,8,8,2,192,False,0.75\n"
        "64,256,8,2,2,192,False,99.0\n",
        encoding="utf-8",
    )
    baseline = VidurProfileBaseline.from_csv(
        attention_csv=attention,
        compute_csv=compute,
        model_name="fixture-inference",
        hardware_name="fixture-hardware",
        attention_backend="AttentionBackend.FLASH_ATTENTION",
        block_size=16,
        source_revision="vidur-test-commit",
    )
    query = InferenceEvidenceQuery(
        phase=InferencePhase.DECODE,
        primitive="attention_core",
        source_layer="attention.core",
        model_name="fixture-inference",
        hardware_name="fixture-hardware",
        model_sequence_length=512,
        hidden_size=64,
        feedforward_size=256,
        attention_heads=8,
        batch_size=3,
        query_tokens=1,
        context_tokens=96,
        tensor_parallel=2,
        datatype="float16",
    )

    exact = baseline.lookup(query)
    assert exact is not None
    assert exact.seconds == pytest.approx(0.00025)
    assert exact.provider == "vidur-profile-baseline"
    assert baseline.lookup(InferenceEvidenceQuery(**{**query.__dict__, "context_tokens": 95})) is None
    compute_exact = baseline.lookup(
        InferenceEvidenceQuery(
            **{
                **query.__dict__,
                "phase": InferencePhase.PREFILL,
                "primitive": "attention_pre_projection",
                "query_tokens": 64,
                "context_tokens": 64,
            }
        )
    )
    assert compute_exact is not None
    assert compute_exact.seconds == pytest.approx(0.00075)

    _, plan = _derive(InferencePhase.DECODE, 96)
    hardware = SystemProfile.from_mapping(
        "fixture-hardware",
        json.loads((ROOT / "data" / "systems" / "a100_80g.json").read_text(encoding="utf-8")),
        datatype="float16",
    )
    digest_before = plan.digest
    network_binding = NetworkTierBinding()
    estimate = estimate_inference_phase(plan, hardware, network_binding=network_binding)
    attention_estimate = next(item for item in estimate.tasks if item.invocation.primitive == "attention_core")
    comparison = compare_inference_phase_to_vidur(plan, estimate, hardware, baseline)
    attention_comparison = next(item for item in comparison.components if item.primitive == "attention_core")

    assert attention_estimate.invocation.work.operations == _task(plan, "attention_core").workload.operations
    assert attention_estimate.total_seconds == attention_estimate.analytical_seconds
    assert attention_estimate.evidence_provider == "analytical-system-profile"
    assert attention_comparison.baseline_seconds == pytest.approx(0.00025)
    assert attention_comparison.estimated_seconds == attention_estimate.total_seconds
    assert not isinstance(baseline, InferenceCostProvider)
    assert plan.digest == digest_before

    experiment = run_vidur_experiment(
        (
            VidurExperimentCase(
                name="fixture/decode-96",
                model=_model(),
                mapping=_execution(),
                network_binding=network_binding,
                datatype="float16",
                hardware=hardware,
                phase=InferencePhase.DECODE,
                batch_size=3,
                context_tokens=96,
            ),
        ),
        baseline,
    )
    assert experiment.policy["baseline_role"] == "post-hoc-comparison-only"
    assert experiment.policy["oracle_read_during_lowering"] is False
    assert experiment.policy["oracle_read_during_costing"] is False
    assert experiment.policy["validation_claim"] == "drift-detection-not-accuracy-validation"
    assert experiment.cases[0].portable_digest == plan.digest
    assert experiment.cases[0].system_evidence.baseline_comparable_block_seconds > 0
    assert experiment.cases[0].system_evidence.component_mean_absolute_error_percent >= abs(
        experiment.cases[0].system_evidence.comparable_subtotal_relative_error_percent
    )

from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprinting.analysis import (
    AIConfiguratorPerformanceImporter,
    CalibrationMode,
    CostQuery,
    CostQueryContext,
    CostResolver,
    CostSubject,
    EstimateMethod,
    EvidenceProvenance,
    HardwareProfile,
    LatencyUnit,
    PerformanceDatabase,
    PerformanceDatabaseProvider,
    PerformanceRecord,
    RooflineCostProvider,
    SimulatorPerformanceImporter,
    TabularImportSpec,
    VidurProfileImporter,
    cost_query_for_inference_task,
    estimate_inference_phase,
)
from blueprinting.analysis.cost import InvalidCostEvidenceError
from blueprinting.compiler.bindings import InferencePhase
from blueprinting.compiler.frozen import FrozenDict
from blueprinting.compiler.lowering import DistributeTransformerInferencePass, PlanTransformerInferencePass
from blueprinting.compiler.models import (
    TransformerInferenceExecutionSpec,
    TransformerModelSpec,
    build_transformer_inference_model_ir,
    inference_compilation_session_for,
)
from blueprinting.compiler.passes import PassManager, PassPipeline

ROOT = Path(__file__).resolve().parents[2]


def _hardware(name: str = "fixture-hardware") -> HardwareProfile:
    return HardwareProfile.from_mapping(
        name,
        json.loads((ROOT / "data" / "systems" / "a100_80g.json").read_text(encoding="utf-8")),
        datatype="float16",
    )


def _provenance(*, digest: str = "fixture-data") -> EvidenceProvenance:
    return EvidenceProvenance(
        source="fixture-simulator",
        source_revision="sim-r1",
        importer="fixture-importer-v1",
        data_digest=digest,
        method=EstimateMethod.SIMULATED,
    )


def test_roofline_exposes_components_and_uses_max_bound():
    hardware = _hardware()
    provider = RooflineCostProvider(
        hardware,
        mode=CalibrationMode.PEAK_ONLY,
        processing_mode="roofline",
    )
    query = CostQuery(
        subject=CostSubject.OPERATOR,
        operation="gemm",
        hardware=hardware.name,
        datatype=hardware.datatype,
        operations=312_000_000_000_000,
        read_bytes=4_096_000_000_000,
        engine="matrix",
        hardware_revision=hardware.evidence_revision,
    )

    estimate = provider.estimate(query)

    assert estimate.seconds == pytest.approx(2.0)
    assert estimate.components["compute_seconds"] == pytest.approx(1.0)
    assert estimate.components["memory_seconds"] == pytest.approx(2.0)
    assert estimate.components["bottleneck"] == "memory"
    assert estimate.method is EstimateMethod.ANALYTICAL


def test_database_aggregates_only_one_exact_provenance_group_and_round_trips():
    records = tuple(
        PerformanceRecord(
            record_id=f"sample-{index}",
            subject=CostSubject.OPERATOR,
            operation="gemm",
            hardware="fixture-hardware",
            datatype="float16",
            seconds=seconds,
            selector=FrozenDict({"m": 8, "n": 16, "k": 32}),
            provenance=_provenance(),
        )
        for index, seconds in enumerate((0.001, 0.003), start=1)
    )
    database = PerformanceDatabase("fixture", records)
    restored = PerformanceDatabase.from_json(database.to_json())
    provider = PerformanceDatabaseProvider(restored)
    query = CostQuery(
        CostSubject.OPERATOR,
        "gemm",
        "fixture-hardware",
        "float16",
        engine="matrix",
        dimensions=FrozenDict({"m": 8, "n": 16, "k": 32, "semantic_operation": "mlp_up_projection"}),
    )

    result = provider.estimate(query)

    assert restored.revision == database.revision
    assert result.seconds == pytest.approx(0.002)
    assert result.uncertainty.sample_count == 2
    assert result.uncertainty.standard_deviation_seconds == pytest.approx(0.001)
    assert result.raw_record_ids == ("sample-1", "sample-2")


def test_ambiguous_database_evidence_is_not_hidden_by_roofline_fallback():
    selector = FrozenDict({"m": 8})
    records = (
        PerformanceRecord(
            "one",
            CostSubject.OPERATOR,
            "gemm",
            "fixture-hardware",
            "float16",
            0.001,
            selector,
            _provenance(digest="run-one"),
        ),
        PerformanceRecord(
            "two",
            CostSubject.OPERATOR,
            "gemm",
            "fixture-hardware",
            "float16",
            0.002,
            selector,
            _provenance(digest="run-two"),
        ),
    )
    resolver = CostResolver(
        (
            PerformanceDatabaseProvider(PerformanceDatabase("ambiguous", records)),
            RooflineCostProvider(_hardware()),
        )
    )
    query = CostQuery(
        CostSubject.OPERATOR,
        "gemm",
        "fixture-hardware",
        "float16",
        engine="matrix",
        dimensions=FrozenDict({"m": 8}),
    )

    with pytest.raises(InvalidCostEvidenceError, match="equally specific"):
        resolver.resolve(query)


def test_simulator_import_maps_units_and_communication_selectors(tmp_path: Path):
    source = tmp_path / "network.csv"
    source.write_text(
        "run_id,op,dtype,latency_us,message_size,participants\nrun-7,all_reduce,float16,125.5,1048576,8\n",
        encoding="utf-8",
    )
    spec = TabularImportSpec(
        name="network-sim",
        subject=CostSubject.COMMUNICATION,
        source="network-simulator",
        source_revision="network-sim-r7",
        method=EstimateMethod.SIMULATED,
        latency_column="latency_us",
        latency_unit=LatencyUnit.MICROSECONDS,
        operation_column="op",
        hardware="candidate-lpu",
        datatype_column="dtype",
        selector_columns=FrozenDict({"message_bytes": "message_size", "participants": "participants"}),
        selector_types=FrozenDict({"message_bytes": "int", "participants": "int"}),
        record_id_column="run_id",
    )

    database = SimulatorPerformanceImporter.from_file(source, spec)
    query = CostQuery(
        CostSubject.COMMUNICATION,
        "all_reduce",
        "candidate-lpu",
        "float16",
        message_bytes=1_048_576,
        participants=8,
    )
    result = PerformanceDatabaseProvider(database).estimate(query)

    assert result.seconds == pytest.approx(125.5e-6)
    assert result.method is EstimateMethod.SIMULATED
    assert result.raw_record_ids == ("run-7",)


def test_aiconfigurator_imports_gemm_and_custom_allreduce_schemas(tmp_path: Path):
    gemm = tmp_path / "gemm_perf.csv"
    gemm.write_text(
        "framework,version,device,op_name,kernel_source,gemm_dtype,m,n,k,latency\n"
        "VLLM,0.24.0,NVIDIA H100 80GB HBM3,gemm,torch.nn.functional.linear,bfloat16,8,16,32,1.5\n",
        encoding="utf-8",
    )
    gemm_db = AIConfiguratorPerformanceImporter.from_file(
        gemm,
        hardware_name="h100-sxm",
        source_revision="aic-commit",
    )
    gemm_query = CostQuery(
        CostSubject.OPERATOR,
        "gemm",
        "h100-sxm",
        "bfloat16",
        engine="matrix",
        implementation="torch.nn.functional.linear",
        runtime="vllm",
        runtime_revision="0.24.0",
        dimensions=FrozenDict({"m": 8, "n": 16, "k": 32}),
    )
    assert PerformanceDatabaseProvider(gemm_db).estimate(gemm_query).seconds == pytest.approx(1.5e-3)

    communication = tmp_path / "custom_allreduce_perf.csv"
    communication.write_text(
        "framework,version,device,op_name,kernel_source,allreduce_dtype,num_gpus,message_size,latency,backend\n"
        "vLLM,0.24.0,NVIDIA H100 80GB HBM3,all_reduce,vLLM_custom_graph,bfloat16,8,1048576,0.25,vllm_graph\n",
        encoding="utf-8",
    )
    communication_db = AIConfiguratorPerformanceImporter.from_file(
        communication,
        hardware_name="h100-sxm",
        source_revision="aic-commit",
    )
    communication_query = CostQuery(
        CostSubject.COMMUNICATION,
        "all_reduce",
        "h100-sxm",
        "bfloat16",
        message_bytes=1_048_576,
        participants=8,
        implementation="vLLM_custom_graph",
        runtime="vllm",
        runtime_revision="0.24.0",
        dimensions=FrozenDict({"backend": "vllm_graph"}),
    )
    assert PerformanceDatabaseProvider(communication_db).estimate(communication_query).seconds == pytest.approx(0.25e-3)


def test_aiconfigurator_parquet_path_uses_the_same_strict_schema(tmp_path: Path):
    pandas = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    source = tmp_path / "gemm_perf.parquet"
    pandas.DataFrame(
        [
            {
                "framework": "SGLang",
                "version": "0.5.14",
                "device": "NVIDIA H100 80GB HBM3",
                "op_name": "gemm",
                "kernel_source": "torch.mm",
                "gemm_dtype": "bfloat16",
                "m": 4,
                "n": 8,
                "k": 16,
                "latency": 0.75,
            }
        ]
    ).to_parquet(source)

    database = AIConfiguratorPerformanceImporter.from_file(
        source,
        hardware_name="h100-sxm",
        source_revision="aic-commit",
    )

    assert len(database.records) == 1
    assert database.records[0].seconds == pytest.approx(0.75e-3)
    assert database.records[0].selector["runtime"] == "sglang"


@pytest.mark.parametrize(
    ("file_name", "upstream_operation", "isl", "step", "phase", "query_tokens", "context_tokens"),
    (
        ("context_attention_perf.csv", "context_attention", 128, 0, "prefill", 128, 128),
        ("generation_attention_perf.csv", "generation_attention", 128, 7, "decode", 1, 135),
    ),
)
def test_aiconfigurator_attention_tables_normalize_phase_and_visible_context(
    tmp_path: Path,
    file_name: str,
    upstream_operation: str,
    isl: int,
    step: int,
    phase: str,
    query_tokens: int,
    context_tokens: int,
):
    source = tmp_path / file_name
    source.write_text(
        "framework,version,device,op_name,kernel_source,batch_size,isl,num_heads,num_key_value_heads,"
        "head_dim,beam_width,attn_dtype,kv_cache_dtype,step,window_size,latency\n"
        f"VLLM,0.24.0,NVIDIA H100 80GB HBM3,{upstream_operation},vllm_flash_attn_fa3,"
        f"4,{isl},4,4,128,1,bfloat16,bfloat16,{step},0,2.0\n",
        encoding="utf-8",
    )
    database = AIConfiguratorPerformanceImporter.from_file(
        source,
        hardware_name="h100-sxm",
        source_revision="aic-commit",
    )
    query = CostQuery(
        CostSubject.OPERATOR,
        "attention_core",
        "h100-sxm",
        "bfloat16",
        engine="matrix",
        implementation="vllm_flash_attn_fa3",
        runtime="vllm",
        runtime_revision="0.24.0",
        dimensions=FrozenDict(
            {
                "semantic_operation": "attention_core",
                "phase": phase,
                "batch_size": 4,
                "query_tokens": query_tokens,
                "context_tokens": context_tokens,
                "local_attention_heads": 4,
                "local_kv_heads": 4,
                "head_size": 128,
                "beam_width": 1,
                "window_size": 0,
                "kv_cache_datatype": "bfloat16",
            }
        ),
    )

    assert PerformanceDatabaseProvider(database).estimate(query).seconds == pytest.approx(2e-3)


def test_vidur_import_is_explicit_and_preserves_exact_profile_context():
    profile = ROOT / "data" / "validation" / "vidur" / "phi2_a100_tp1"
    database = VidurProfileImporter.from_csv(
        attention_csv=profile / "attention.csv",
        compute_csv=profile / "mlp.csv",
        model_name="microsoft/phi-2",
        hardware_name="a100_80g",
        source_revision="8383d2935bc62723a212090baa9f98ada206fc14",
    )
    query = CostQuery(
        CostSubject.OPERATOR,
        "attention_core",
        "a100_80g",
        "float16",
        engine="matrix",
        implementation="AttentionBackend.FLASH_ATTENTION",
        dimensions=FrozenDict(
            {
                "semantic_operation": "attention_core",
                "source_layer": "attention.core",
                "phase": "decode",
                "model_name": "microsoft/phi-2",
                "model_sequence_length": 4096,
                "hidden_size": 2560,
                "attention_heads": 32,
                "kv_heads": 32,
                "batch_size": 1,
                "query_tokens": 1,
                "context_tokens": 33,
                "tensor_parallel": 1,
                "block_size": 16,
            }
        ),
    )

    result = PerformanceDatabaseProvider(database).estimate(query)

    assert result.seconds == pytest.approx(9e-6)
    assert result.method is EstimateMethod.MEASURED
    assert result.source_revision == "8383d2935bc62723a212090baa9f98ada206fc14"


def _inference_fixture():
    model = TransformerModelSpec(
        name="cost-inference",
        hidden_size=64,
        feedforward_size=256,
        sequence_length=512,
        attention_heads=8,
        attention_head_size=8,
        block_count=8,
    )
    execution = TransformerInferenceExecutionSpec(
        world_size=4,
        tensor_parallel=2,
        pipeline_parallel=2,
        replicas=1,
        datatype="float16",
        tensor_parallel_network=0,
        pipeline_parallel_network=0,
    )
    plan = (
        PassManager()
        .run(
            PassPipeline.of(DistributeTransformerInferencePass(), PlanTransformerInferencePass()),
            build_transformer_inference_model_ir(model),
            session=inference_compilation_session_for(
                model,
                execution,
                phase=InferencePhase.DECODE,
                batch_size=3,
                context_tokens=96,
            ),
        )
        .ir
    )
    return model, execution, plan


def test_inference_costing_uses_database_then_explicit_roofline_fallback():
    model, execution, plan = _inference_fixture()
    hardware = _hardware()
    attention_task = next(task for task in plan.tasks if task.workload.attributes.get("primitive") == "attention_core")
    query = cost_query_for_inference_task(
        attention_task,
        hardware=hardware,
        execution=execution,
        model=model,
        batch_size=3,
        query_tokens=1,
        context_tokens=96,
        context=CostQueryContext(),
    )
    record = PerformanceRecord(
        "attention-sim-1",
        CostSubject.OPERATOR,
        "attention_core",
        hardware.name,
        execution.datatype,
        0.123,
        FrozenDict(
            {
                "semantic_operation": "attention_core",
                "context_tokens": 96,
                "batch_size": 3,
            }
        ),
        _provenance(),
    )
    database_provider = PerformanceDatabaseProvider(PerformanceDatabase("inference", (record,)))
    resolver = CostResolver(
        (
            database_provider,
            RooflineCostProvider(hardware, mode=CalibrationMode.PEAK_ONLY),
        )
    )

    estimate = estimate_inference_phase(
        plan,
        hardware,
        mode=CalibrationMode.PEAK_ONLY,
        cost_resolver=resolver,
    )
    attention = next(item for item in estimate.tasks if item.invocation.primitive == "attention_core")
    fallback = next(item for item in estimate.tasks if item.invocation.primitive == "input_layernorm")

    assert query.operation == "attention_core"
    assert attention.total_seconds == pytest.approx(0.123)
    assert attention.evidence_provider == database_provider.name
    assert attention.evidence_source_revision == "sim-r1"
    assert attention.evidence_record_ids == ("attention-sim-1",)
    assert attention.evidence_method == "simulated"
    assert fallback.evidence_provider == f"roofline:{hardware.name}"
    assert fallback.evidence_method == "analytical"
    assert estimate.evidence_revisions[resolver.revision] == "cost-resolver-policy"

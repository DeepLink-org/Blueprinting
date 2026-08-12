from __future__ import annotations

import shutil
import subprocess

import pytest

from blueprinting.analysis import CalibrationMode
from blueprinting.application import (
    AnalysisDraft,
    BlueprintingService,
    CanonicalIRStage,
    EntityRef,
    IRGraphEdge,
    IRGraphNode,
    IRGraphView,
    SweepCase,
    SweepReport,
)
from blueprinting.workbench import default_catalog
from blueprinting.workbench.ir_expressions import lowering_correspondence_rows, lowering_expression
from blueprinting.workbench.presentation import (
    analysis_metrics,
    dependency_timeline_chart_options,
    ir_graph_chart_options,
    ir_stage_narrative,
    lowering_narrative,
    memory_chart_options,
    semantic_boundary_rows,
    sweep_chart_options,
    sweep_distribution_chart_options,
    task_dependency_projection,
    time_breakdown_rows,
    time_breakdown_tree,
    timeline_summary,
)


def _analysis_report():
    catalog = default_catalog()
    outcome = BlueprintingService().analyze(
        AnalysisDraft.from_mappings(
            model_name="small-transformer",
            model_data={
                "hidden": 128,
                "feedforward": 512,
                "seq_size": 32,
                "attn_heads": 8,
                "attn_size": 16,
                "num_blocks": 4,
            },
            execution_name="small-execution",
            execution_data={
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
                "attention_type": "multihead",
                "activation_recompute": "none",
                "pipeline_interleaving": 1,
                "optimizer_sharding": False,
                "tensor_par_comm_type": "ar",
                "tensor_par_overlap": "none",
                "data_par_overlap": False,
                "weight_offload": False,
                "activations_offload": False,
                "optimizer_offload": False,
                "training": True,
            },
            hardware_name="a100_80g",
            hardware_data=catalog.load("systems", "a100_80g.json"),
            calibration_mode=CalibrationMode.SYSTEM_EVIDENCE,
        )
    )
    assert outcome.report is not None
    return outcome.report


def test_analysis_summary_has_four_decision_metrics() -> None:
    metrics = analysis_metrics(_analysis_report())

    assert [metric.label for metric in metrics] == ["迭代延迟", "全局吞吐", "单设备吞吐", "单设备内存"]


def test_large_ir_graph_uses_a_bounded_semantic_group_projection() -> None:
    digest = "a" * 40
    refs = tuple(EntityRef(CanonicalIRStage.PORTABLE, digest, "task", f"node:{index:05d}") for index in range(10_000))
    graph = IRGraphView(
        CanonicalIRStage.PORTABLE,
        digest,
        tuple(IRGraphNode(ref, ref.entity_id, f"phase-{index % 100}") for index, ref in enumerate(refs)),
        tuple(IRGraphEdge(refs[index - 1], refs[index], "dependency") for index in range(1, len(refs))),
    )

    options = ir_graph_chart_options(graph)

    entity_nodes = [
        item for item in options["series"][0]["data"] if not item.get("is_phase") and not item.get("is_lane")
    ]

    assert len(entity_nodes) <= 20
    assert all(item["kind"] == "group" for item in entity_nodes)
    assert "10,000 canonical entities" in options["graphic"][0]["style"]["text"]


def test_distributed_structure_uses_ordered_phase_and_subsystem_lanes() -> None:
    trace = _analysis_report().derivation_trace
    graph = trace.graph(CanonicalIRStage.DISTRIBUTED, trace.branches[0])
    assert graph is not None

    options = ir_graph_chart_options(graph, max_nodes=20)
    series = options["series"][0]
    phase_nodes = [item for item in series["data"] if item.get("is_phase")]
    lane_nodes = [item for item in series["data"] if item.get("is_lane")]
    entity_nodes = [item for item in series["data"] if not item.get("is_phase") and not item.get("is_lane")]

    assert series["layout"] == "none"
    assert not series["roam"]
    assert not series["draggable"]
    assert [item["name"] for item in phase_nodes] == ["输入", "前向", "激活梯度", "权重梯度", "优化器", "输出"]
    assert [item["x"] for item in phase_nodes] == sorted(item["x"] for item in phase_nodes)
    assert any(item["name"] == "attention\nCollective" for item in lane_nodes)
    assert any(item["name"].startswith("attention\n") for item in entity_nodes)
    assert any(item["entity_kind"] == "collective" for item in entity_nodes)
    assert all("other" not in item["name"] for item in entity_nodes)
    lane_positions = {(item["subsystem"], item["entity_kind"]): item["y"] for item in lane_nodes}
    assert all(item["y"] == lane_positions[(item["subsystem"], item["entity_kind"])] for item in entity_nodes)


def test_ir_stage_narrative_explains_ownership_and_snapshot_result() -> None:
    report = _analysis_report()
    model_stage = report.derivation_trace.stages[0]

    narrative = ir_stage_narrative(model_stage.ir)

    assert narrative.question == "这个模型在语义上做什么？"
    assert "1 个 operation" in narrative.result
    assert "不包含并行 placement" in narrative.excludes
    assert "逻辑 mesh" in narrative.next_step


def test_boundary_presentation_explains_and_groups_one_to_many_lowering() -> None:
    trace = _analysis_report().derivation_trace
    transition = trace.transitions[0]
    boundary = transition.boundary
    branch = trace.branches[0]
    source_graph = trace.graph(boundary.source_stage, branch)
    target_graph = trace.graph(boundary.target_stage, branch)
    assert source_graph is not None
    assert target_graph is not None

    narrative = lowering_narrative(boundary, source_graph, target_graph)
    rows = semantic_boundary_rows(boundary, source_graph, target_graph)

    assert "transformer.decoder_training 被展开" in narrative.headline
    assert "typed lineage" in narrative.detail
    assert len(rows) < len(boundary.relations)
    decoder_rows = [row for row in rows if row["source"] == "transformer.decoder_training"]
    assert len(decoder_rows) > 1
    assert {row["mapping"] for row in decoder_rows} == {"1 → 57"}
    assert sum(row["source_group_start"] for row in decoder_rows) == 1
    assert len({row["transform_span_key"] for row in decoder_rows}) == 1
    assert any(
        row["source"] == "transformer.decoder_training"
        and row["target"] == "前向 · attention"
        and row["target_kind"] == "local compute"
        and row["mapping_kind"] == "展开"
        and row["mapping"].startswith("1 → ")
        for row in rows
    )
    assert all("node:" not in row["source"] for row in rows)

    expression = lowering_expression(transition, rows)
    assert "ModelOperation -> DistributedTask" in expression.short
    assert "normal_form blueprinting.synthesizer.dialects.transformer.training_derivation." in expression.detailed
    assert "normalize_training_distribution" in expression.detailed
    assert "transition_status relation_verified" in expression.detailed
    assert transition.canonical_conformance is not None
    assert "canonical_conformance blueprinting.synthesizer.dialects.transformer.training_derivation." in (
        expression.detailed
    )
    assert "invariant  @blueprinting.synthesizer.dialects.transformer.training_derivation." in expression.detailed
    assert "verified_claims" in expression.detailed
    assert "forbids    [physical device, queue, kernel, predicted time]" in expression.detailed

    correspondence = lowering_correspondence_rows(transition, rows)
    decoder_correspondence = [row for row in correspondence if row["source"] == "transformer.decoder_training"]
    assert decoder_correspondence
    assert {row["qualified_rule"] for row in decoder_correspondence} == {"transformer-distribute.transformer-decompose"}
    assert {row["rule_signature"] for row in decoder_correspondence} == {"ModelOperation ⇒ DistributedTask"}
    assert all(row["rule_declared"] for row in decoder_correspondence)
    assert all("relation=展开, cardinality=1 → 57" in row["transform_expression"] for row in decoder_correspondence)
    assert all(
        "Expand one semantic decoder-training operation into an ordered local/collective task DAG"
        in row["transform_expression"]
        for row in decoder_correspondence
    )
    assert {row["source_type"] for row in decoder_correspondence} == {"Operation"}
    assert {row["source_tone"] for row in decoder_correspondence} == {"blue"}
    assert all(row["pass_type"] == "Pass" for row in decoder_correspondence)
    assert all(
        row["pass_expression_parameters"]
        == (
            {"name": "relation", "value": "展开", "category": "mapping"},
            {"name": "cardinality", "value": "1 → 57", "category": "mapping"},
        )
        for row in decoder_correspondence
    )
    assert all(
        row["source_expression"]
        == "transformer.decoder_training(shape=[microbatch_size, sequence_length, 128], dtype=float16)"
        for row in decoder_correspondence
    )
    assert any(
        row["target_expression"] == "MLP.forward(ranks=2, count=6)"
        and row["target_type"] == "Local compute"
        and row["target_tone"] == "blue"
        for row in decoder_correspondence
    )


def test_portable_target_expression_exposes_exact_workload_facts() -> None:
    trace = _analysis_report().derivation_trace
    transition = trace.transitions[1]
    boundary = transition.boundary
    branch = trace.branches[0]
    rows = semantic_boundary_rows(
        boundary,
        trace.graph(boundary.source_stage, branch),
        trace.graph(boundary.target_stage, branch),
    )

    value_to_buffer = next(row for row in rows if row["source_kind"] == "value" and row["target_kind"] == "buffer")
    assert value_to_buffer["source_expression_name"].startswith("value#")
    assert value_to_buffer["target_expression_name"].startswith("buffer#")
    assert value_to_buffer["source_entity_ids"][0].startswith("value:")
    assert value_to_buffer["target_entity_ids"][0].startswith("buffer:")

    mlp_forward = next(row for row in rows if row["target"] == "前向 · mlp" and row["target_kind"] == "compute")

    assert mlp_forward["source_expression"] == "MLP.forward(ranks=2, count=6)"
    assert mlp_forward["target_type"] == "Compute"
    assert mlp_forward["target_tone"] == "blue"
    assert mlp_forward["target_expression_name"] == "MLP.forward"
    assert mlp_forward["target_expression_parameters"] == (
        {"name": "ranks", "value": "2", "category": "topology"},
        {"name": "ops", "value": "4304896", "category": "workload"},
        {"name": "read_B", "value": "217600", "category": "workload"},
        {"name": "write_B", "value": "57344", "category": "workload"},
        {"name": "count", "value": "6", "category": "workload"},
    )
    assert mlp_forward["target_expression"] == (
        "MLP.forward(ranks=2, ops=4304896, read_B=217600, write_B=57344, count=6)"
    )


def test_memory_chart_stacks_components_and_marks_capacity() -> None:
    options = memory_chart_options(_analysis_report())

    assert all(series["stack"] == "memory" for series in options["series"])
    assert options["series"][0]["markLine"]["data"][0]["xAxis"] > 0


def test_time_breakdown_preserves_iteration_total_at_every_parent() -> None:
    report = _analysis_report()
    tree = time_breakdown_tree(report)

    assert sum(node["value"] for node in tree) == pytest.approx(report.total_seconds)
    for node in tree:
        assert sum(child["value"] for child in node["children"]) == pytest.approx(node["value"])

    rows = time_breakdown_rows(report)
    expected_categories = {name for name, value in report.latency.items() if float(value) > 0}
    assert {row["category"] for row in rows if row["level"] == 1} == expected_categories


def test_dependency_timeline_contains_every_task_and_respects_edges() -> None:
    report = _analysis_report()
    projection = task_dependency_projection(report)
    by_id = {row["task_id"]: row for row in projection}

    assert len(projection) == len(report.tasks)
    for task in report.tasks:
        for dependency in task.dependencies:
            assert by_id[task.task_id]["start_seconds"] >= by_id[dependency]["end_seconds"]

    summary = timeline_summary(report)
    options = dependency_timeline_chart_options(report)
    assert summary.task_count == len(report.tasks)
    assert len(options["series"][0]["data"]) == len(report.tasks)
    assert options["series"][0]["type"] == "custom"
    assert len(options["dataZoom"]) == 2


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for JavaScript rendering test")
def test_dependency_timeline_render_item_executes_without_echarts_global() -> None:
    render_item = dependency_timeline_chart_options(_analysis_report())["series"][0][":renderItem"]
    script = """
const renderItem = new Function('return (' + process.argv[1] + ');')();
const values = [0, 10, 25, 15];
const result = renderItem(
  {coordSys: {x: 0, y: 0, width: 500, height: 300}},
  {
    value: index => values[index],
    coord: point => [point[0] * 2, 50 + point[1] * 40],
    size: () => [0, 40],
    style: () => ({fill: '#2563eb'}),
  },
);
if (!result || result.type !== 'rect') throw new Error('renderItem did not return a rectangle');
if (result.shape.width <= 0 || result.shape.height <= 0) throw new Error('rectangle has no area');
process.stdout.write('rendered');
"""

    completed = subprocess.run(
        ["node", "-e", script, render_item],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout == "rendered"


def test_sweep_chart_distinguishes_feasible_nondominated_and_capacity_failure() -> None:
    report = SweepReport(
        schema="test",
        request_digest="request",
        cases=(
            SweepCase(1, 1, 1, 1, "success", True, True, 1.0, 10, 1.0, "forward", "a", ()),
            SweepCase(2, 1, 1, 2, "success", True, False, 2.0, 12, 0.8, "memory", "b", ()),
            SweepCase(4, 1, 1, 4, "success", False, False, 0.5, 20, 1.4, "memory", "c", ()),
        ),
    )

    series = sweep_chart_options(report)["series"]

    assert [item["name"] for item in series] == ["可行候选", "非支配候选", "容量不可行"]
    assert [len(item["data"]) for item in series] == [1, 1, 1]


def test_sweep_distribution_tracks_all_visible_successful_cases() -> None:
    report = SweepReport(
        schema="test",
        request_digest="request",
        cases=(
            SweepCase(1, 1, 1, 1, "success", True, True, 1.0, 10 * 1024**3, 1.0, "forward", "a", ()),
            SweepCase(2, 1, 1, 2, "success", True, False, 2.0, 12 * 1024**3, 0.8, "memory", "b", ()),
            SweepCase(4, 1, 1, 4, "failed", False, False, None, None, None, None, "c", ()),
        ),
    )

    options = sweep_distribution_chart_options(report)

    assert [series["name"] for series in options["series"]] == ["Case 数", "Case 数"]
    assert [sum(series["data"]) for series in options["series"]] == [2, 2]

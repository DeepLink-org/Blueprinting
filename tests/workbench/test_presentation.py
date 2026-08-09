from __future__ import annotations

import shutil
import subprocess

import pytest

from blueprinting.analysis import CalibrationMode
from blueprinting.application import AnalysisDraft, BlueprintingService, SweepCase, SweepReport
from blueprinting.workbench import default_catalog
from blueprinting.workbench.presentation import (
    analysis_metrics,
    dependency_timeline_chart_options,
    memory_chart_options,
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

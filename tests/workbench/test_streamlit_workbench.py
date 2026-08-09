from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest


@pytest.mark.parametrize(
    "path,button_label",
    (
        ("pages/Blueprinting/overview.py", "运行 Blueprinting 分析"),
        ("pages/Blueprinting/ir_audit.py", "生成并审计 IR"),
        ("pages/Blueprinting/strategy_explorer.py", "探索策略空间"),
    ),
)
def test_workbench_pages_load_without_eager_analysis(path: str, button_label: str) -> None:
    app = AppTest.from_file(path, default_timeout=30).run()

    assert not app.exception
    assert [button.label for button in app.button] == [button_label]
    assert not app.metric


def test_overview_runs_blueprinting_analysis() -> None:
    app = AppTest.from_file("pages/Blueprinting/overview.py", default_timeout=30).run()

    app.button[0].click().run()

    assert not app.exception
    assert not app.error
    assert app.success
    assert {metric.label for metric in app.metric} == {
        "迭代延迟",
        "Token/s",
        "Token/s/设备",
        "单设备内存",
        "主导项",
    }


def test_ir_audit_exposes_all_current_boundaries() -> None:
    app = AppTest.from_file("pages/Blueprinting/ir_audit.py", default_timeout=30).run()

    app.button[0].click().run()

    assert not app.exception
    stage_table = app.dataframe[0].value
    assert tuple(stage_table["stage"]) == ("model", "distributed", "portable")
    assert stage_table["valid"].all()
    assert len(app.get("download_button")) == 3


def test_strategy_explorer_keeps_candidate_status() -> None:
    app = AppTest.from_file("pages/Blueprinting/strategy_explorer.py", default_timeout=30).run()

    app.button[0].click().run()

    assert not app.exception
    assert {metric.label for metric in app.metric} == {"候选", "成功", "可行", "Pareto"}
    table = app.dataframe[0].value
    assert {"status", "feasible", "pareto", "request_digest"} <= set(table.columns)


def test_navigation_defaults_to_blueprinting_without_path_collisions() -> None:
    app = AppTest.from_file("streamlit_app.py", default_timeout=30).run()

    assert not app.exception
    assert app.title[0].value == "🧭 分析总览"

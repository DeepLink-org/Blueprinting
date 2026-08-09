from __future__ import annotations

from streamlit.testing.v1 import AppTest


def test_workbench_page_loads_without_eager_analysis() -> None:
    app = AppTest.from_file("pages/Blueprinting/overview.py", default_timeout=30).run()

    assert not app.exception
    assert [button.label for button in app.button] == ["运行 Blueprinting 分析"]
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


def test_navigation_defaults_to_blueprinting_without_path_collisions() -> None:
    app = AppTest.from_file("streamlit_app.py", default_timeout=30).run()

    assert not app.exception
    assert app.title[0].value == "🧭 分析总览"

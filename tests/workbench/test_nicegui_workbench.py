from __future__ import annotations

import importlib
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from typing import Any

import pytest
from nicegui import core
from nicegui.testing import User, user_simulation

from blueprinting.workbench.nicegui_app import build_parser, run_workbench
from blueprinting.workbench.nicegui_theme import WORKBENCH_CSS
from blueprinting.workbench.nicegui_ui import create_workbench_root


@pytest.fixture
def simulated_user(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[Callable[[], None]], AbstractAsyncContextManager[User]]:
    """Work around NiceGUI 3.x resetting itself into script mode in its test helper.

    Production root-function mode is covered separately by starting the real ASGI
    server.  The upstream simulator currently calls ``app.reset()`` before its
    setup hook; resetting Quasar colors flips ``core.script_mode`` back on.
    """

    simulation_module = importlib.import_module("nicegui.testing.user_simulation")
    original_prepare = simulation_module.prepare_simulation

    def prepare_simulation() -> None:
        original_prepare()
        core.script_mode = False

    monkeypatch.setattr(simulation_module, "prepare_simulation", prepare_simulation)
    return user_simulation


async def test_nicegui_workbench_loads_without_eager_analysis(
    simulated_user: Callable[[Callable[[], None]], AbstractAsyncContextManager[User]],
) -> None:
    async with simulated_user(create_workbench_root()) as user:
        await user.open("/")

        await user.should_see("单点剖析")
        await user.should_see("观察尺度")
        await user.should_see("当前 Case")
        await user.should_see("这个视图回答什么")
        await user.should_see("等待运行")
        await user.should_see("当前分析对象")
        await user.should_see("当前显示解析任务贡献，不是事件级 Timeline")
        await user.should_see(marker="run-analysis")
        await user.should_see(marker="sidebar-run-analysis")
        await user.should_see(marker="mode-evidence")
        await user.should_see(marker="mode-float")


async def test_sidebar_quick_controls_sync_with_full_configuration(
    simulated_user: Callable[[Callable[[], None]], AbstractAsyncContextManager[User]],
) -> None:
    async with simulated_user(create_workbench_root()) as user:
        await user.open("/")

        user.find(marker="quick-tp").clear().type("8").trigger("update:model-value")
        user.find(marker="open-full-configuration").click()

        await user.should_see(marker="configuration-modal")
        await user.should_see("模型语义参数")
        tp_input = next(iter(user.find(marker="tp-input").elements))
        assert tp_input.value == 8.0


async def test_nicegui_analysis_reuses_one_result_across_views(
    simulated_user: Callable[[Callable[[], None]], AbstractAsyncContextManager[User]],
) -> None:
    async with simulated_user(create_workbench_root()) as user:
        await user.open("/")
        user.find(marker="run-analysis").click()

        await user.should_see("迭代延迟", retries=100)
        await user.should_see("结果已就绪", retries=100)
        await user.should_see("自顶向下时间分解", retries=100)
        await user.should_see("ITERATION HIERARCHY", retries=100)
        await user.should_see("第一级严格使用 iteration estimate", retries=100)
        await user.should_see("Portable task timeline", retries=100)
        await user.should_see("DEPENDENCY PROJECTION", retries=100)
        await user.should_see(marker="open-perfetto", retries=100)
        await user.should_see("查看完整层次明细", retries=100)
        await user.should_see("Portable task audit", retries=100)
        await user.should_see("Canonical derivation checkpoints", retries=100)
        await user.should_see("模型语义", retries=100)
        await user.should_see("分布式任务", retries=100)
        await user.should_see("可移植计划", retries=100)


async def test_nicegui_marks_results_stale_after_configuration_change(
    simulated_user: Callable[[Callable[[], None]], AbstractAsyncContextManager[User]],
) -> None:
    async with simulated_user(create_workbench_root()) as user:
        await user.open("/")
        user.find(marker="run-analysis").click()
        await user.should_see("迭代延迟", retries=100)

        user.find(marker="edit-configuration").click()
        user.find(marker="tp-input").clear().type("8").trigger("update:model-value")

        await user.should_see("配置已经变化；当前页面仍显示上一次结果")
        await user.should_see("结果需要更新")
        await user.should_see(marker="rerun-analysis")


async def test_nicegui_strategy_sweep_keeps_candidate_status(
    simulated_user: Callable[[Callable[[], None]], AbstractAsyncContextManager[User]],
) -> None:
    async with simulated_user(create_workbench_root()) as user:
        await user.open("/")
        user.find(marker="mode-sweep").click()
        await user.should_see("批量探索")
        await user.should_see("批量候选定义")
        quick_candidates = next(iter(user.find(marker="quick-tp-candidates").elements))
        assert quick_candidates.props["popup-content-class"] == "bp-sidebar-menu"
        user.find(marker="run-sweep").click()

        await user.should_see("CaseSet 过滤器", retries=100)
        await user.should_see("延迟—内存空间", retries=100)
        await user.should_see("可见 Case", retries=100)
        await user.should_see("延迟上下界", retries=100)
        await user.should_see("内存上下界", retries=100)
        await user.should_see("非支配", retries=100)
        await user.should_see(marker="batch-status-filter", retries=100)
        await user.should_see(marker="batch-open-point", retries=100)


async def test_nicegui_float_analysis_is_available_without_running_workload_analysis(
    simulated_user: Callable[[Callable[[], None]], AbstractAsyncContextManager[User]],
) -> None:
    async with simulated_user(create_workbench_root()) as user:
        await user.open("/")
        user.find(marker="mode-float").click()

        await user.should_see("浮点数分析")
        await user.should_see("格式位宽对比")
        await user.should_see("FP8(E5M2) 位级计算器")
        await user.should_see("动态范围与可表示值")
        await user.should_see("四则运算范围影响")
        await user.should_see(marker="float-exponent-bits")
        await user.should_see(marker="float-mantissa-bits")

        user.find(marker="float-exponent-bits").clear().type("4").trigger("update:model-value")
        await user.should_see("FP7(E4M2) 位级计算器")


async def test_nicegui_evidence_lab_compares_measured_and_analytical_curves(
    simulated_user: Callable[[Callable[[], None]], AbstractAsyncContextManager[User]],
) -> None:
    async with simulated_user(create_workbench_root()) as user:
        await user.open("/")
        user.find(marker="mode-evidence").click()

        await user.should_see("性能证据实验室")
        await user.should_see("Evidence catalog")
        await user.should_see("Operation coverage")
        await user.should_see("Measured vs analytical")
        await user.should_see("Roofline relative error")
        await user.should_see("Roofline 全部偏乐观")
        await user.should_see("Vidur exact records")
        await user.should_see(marker="evidence-semantic-operation")
        await user.should_see(marker="evidence-query-inspector")


def test_workbench_cli_defaults_to_local_only() -> None:
    args = build_parser().parse_args([])

    assert args.host == "127.0.0.1"
    assert args.port == 8080
    assert not args.no_open
    assert not args.reload


def test_sidebar_mode_switch_uses_a_non_scrolling_two_by_two_grid() -> None:
    assert "grid-template-columns: repeat(2, minmax(0, 1fr))" in WORKBENCH_CSS
    assert ".bp-mode-switch .bp-mode-button" in WORKBENCH_CSS
    assert ".bp-mode-switch .bp-mode-button--active" in WORKBENCH_CSS
    assert ".bp-mode-switch .bp-mode-icon" in WORKBENCH_CSS
    assert "position: static" in WORKBENCH_CSS
    assert "q-tabs__arrow" not in WORKBENCH_CSS
    assert "q-tab__indicator" not in WORKBENCH_CSS


def test_workbench_cli_accepts_server_overrides() -> None:
    args: Any = build_parser().parse_args(["--host", "0.0.0.0", "--port", "9000", "--no-open", "--reload"])

    assert (args.host, args.port, args.no_open, args.reload) == ("0.0.0.0", 9000, True, True)


def test_workbench_runtime_starts_in_light_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def capture_run(_root: Callable[[], None], **options: Any) -> None:
        captured.update(options)

    monkeypatch.setattr("blueprinting.workbench.nicegui_app.ui.run", capture_run)

    run_workbench(show=False)

    assert captured["dark"] is False

from __future__ import annotations

import importlib
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from typing import Any

import pytest
from nicegui import core
from nicegui.testing import User, user_simulation

from blueprinting.workbench.nicegui_app import build_parser
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

        await user.should_see("从工作负载语义走向可审计的硬件蓝图")
        await user.should_see("等待第一张硬件蓝图")
        await user.should_see(marker="run-analysis")
        await user.should_see(marker="run-sweep")
        await user.should_see("Calculon / Streamlit Legacy")


async def test_nicegui_analysis_reuses_one_result_across_views(
    simulated_user: Callable[[Callable[[], None]], AbstractAsyncContextManager[User]],
) -> None:
    async with simulated_user(create_workbench_root()) as user:
        await user.open("/")
        user.find(marker="run-analysis").click()

        await user.should_see("迭代延迟", retries=100)
        await user.should_see("Portable workload facts", retries=100)
        await user.should_see("Portable task audit", retries=100)
        await user.should_see("模型语义", retries=100)
        await user.should_see("分布式任务", retries=100)
        await user.should_see("可移植计划", retries=100)


async def test_nicegui_strategy_sweep_keeps_candidate_status(
    simulated_user: Callable[[Callable[[], None]], AbstractAsyncContextManager[User]],
) -> None:
    async with simulated_user(create_workbench_root()) as user:
        await user.open("/")
        user.find(marker="run-sweep").click()

        await user.should_see("延迟—内存空间", retries=100)
        await user.should_see("全部候选", retries=100)
        await user.should_see("strategy candidates", retries=100)


def test_workbench_cli_defaults_to_local_only() -> None:
    args = build_parser().parse_args([])

    assert args.host == "127.0.0.1"
    assert args.port == 8080
    assert not args.no_open
    assert not args.reload


def test_workbench_cli_accepts_server_overrides() -> None:
    args: Any = build_parser().parse_args(["--host", "0.0.0.0", "--port", "9000", "--no-open", "--reload"])

    assert (args.host, args.port, args.no_open, args.reload) == ("0.0.0.0", 9000, True, True)

"""Program abstraction for progressive collapse.

Program 表示拓扑塌缩后的残留程序（ScheduleIR），支持：
- 多次 bind（代数塌缩）
- 多目的 collapse（仿真 / 符号估算 / 自定义后端）
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .estimate import SymbolicEstimate
from .passes.overlap import OverlapAnalysisPass
from .passes.simulate import SimulatePass
from .passes.symbolic_estimate import SymbolicEstimatePass
from .passes.timeline import TimelinePass
from .types import ScheduleIR


class CollapseBackend(ABC):
    """塌缩后端：将 ScheduleIR 投影到特定目的产物."""

    @abstractmethod
    def collapse(self, ir: ScheduleIR, **params) -> Any:
        """对 ScheduleIR 执行目的特定的塌缩."""


class SimulationBackend(CollapseBackend):
    """事件仿真后端：ScheduleIR -> TimelineIR -> SimulationResult."""

    def __init__(
        self,
        timeline_pass: TimelinePass | None = None,
        overlap_pass: OverlapAnalysisPass | None = None,
        simulate_pass: SimulatePass | None = None,
    ):
        self.timeline_pass = timeline_pass or TimelinePass(track_memory=True)
        self.overlap_pass = overlap_pass
        self.simulate_pass = simulate_pass or SimulatePass()

    def collapse(self, ir: ScheduleIR, **params) -> Any:
        working_ir = ir.subs(params) if params else ir
        timeline = self.timeline_pass.run(working_ir)
        if self.overlap_pass is not None:
            timeline = self.overlap_pass.run(timeline)

        base_subs = dict(getattr(self.simulate_pass, "subs", {}) or {})
        merged_subs = {**base_subs, **params}
        peak_tflops = getattr(self.simulate_pass, "peak_tflops", 312.0)
        return SimulatePass(subs=merged_subs, peak_tflops=peak_tflops).run(timeline)


class SymbolicBackend(CollapseBackend):
    """符号估算后端：ScheduleIR -> SymbolicEstimate."""

    def collapse(self, ir: ScheduleIR, **params) -> SymbolicEstimate | None:
        working_ir = ir.subs(params) if params else ir
        estimate = working_ir.metadata.get("symbolic_estimate")
        if estimate is not None:
            return estimate
        enriched = SymbolicEstimatePass().run(working_ir)
        return enriched.metadata.get("symbolic_estimate")


class Program:
    """拓扑塌缩后的残留程序.

    Program 只持有 ScheduleIR，不预设最终目标。
    """

    def __init__(
        self,
        ir: ScheduleIR,
        simulation_backend: SimulationBackend | None = None,
        symbolic_backend: SymbolicBackend | None = None,
    ):
        self._ir = ir
        self._simulation_backend = simulation_backend or SimulationBackend()
        self._symbolic_backend = symbolic_backend or SymbolicBackend()

    def bind(self, **params) -> Program:
        """代入参数子集，返回新的 Program."""
        return Program(
            ir=self._ir.subs(params),
            simulation_backend=self._simulation_backend,
            symbolic_backend=self._symbolic_backend,
        )

    @property
    def free_symbols(self):
        """当前残留的自由符号集合."""
        return self._ir.free_symbols

    @property
    def is_concrete(self) -> bool:
        """是否不存在未绑定符号."""
        return len(self.free_symbols) == 0

    def collapse(self, backend: CollapseBackend, **params) -> Any:
        """使用指定后端执行目的特定塌缩."""
        working_ir = self._ir.subs(params) if params else self._ir
        return backend.collapse(working_ir, **params)

    def simulate(self, **params):
        """精确仿真（事件化路径）."""
        return self._simulation_backend.collapse(self._ir, **params)

    @property
    def estimate(self) -> SymbolicEstimate | None:
        """符号估算（若无则惰性生成）."""
        return self._symbolic_backend.collapse(self._ir)

    @property
    def schedule_ir(self) -> ScheduleIR:
        """暴露内部 ScheduleIR 供检视."""
        return self._ir

    def sweep(self, **param_ranges) -> list[tuple[dict[str, Any], Any]]:
        """参数扫描：对每组参数执行 simulate."""
        import itertools

        if not param_ranges:
            return [({}, self.simulate())]

        keys = list(param_ranges.keys())
        values = [param_ranges[k] for k in keys]
        results = []
        for combo in itertools.product(*values):
            params = dict(zip(keys, combo))
            results.append((params, self.simulate(**params)))
        return results

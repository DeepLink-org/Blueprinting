"""SymbolicEstimate - 符号化聚合估算结果.

从 ScheduleIR 聚合为符号表达式，支持参数扫描、敏感性分析、瓶颈识别和校准。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any

from sympy import Expr, Symbol

from blueprinting.core import eval_lazy, get_symbol, sym_min


@dataclass
class SymbolicEstimate:
    """符号化估算结果 - 从 ScheduleIR 聚合的符号表达式.

    所有时间和内存字段均保持符号形式（Expr | float），
    支持参数扫描、敏感性分析等操作。
    """

    # === 时间分解（符号表达式，保持未 eval）===
    forward_time: Any = 0  # 总前向计算时间（per-stage）
    backward_time: Any = 0  # 总反向计算时间（per-stage，不含 recompute）
    recompute_time: Any = 0  # 重计算时间（per-stage）
    comm_time: Any = 0  # 总通信时间（per-stage）
    bubble_time: Any = 0  # PP bubble 时间
    overlap_ratio: float = 0.0  # 计算/通信重叠比例（可校准，0~1）

    # === 内存分解（符号表达式）===
    weight_memory: Any = 0  # 权重内存（per-GPU）
    activation_memory: Any = 0  # 激活内存（per-GPU）
    gradient_memory: Any = 0  # 梯度内存（per-GPU）
    optimizer_memory: Any = 0  # 优化器状态内存（per-GPU）

    # === 可追溯的分解字典（用于关键因素分析）===
    time_breakdown_by_op: dict[str, Any] = field(default_factory=dict)
    time_breakdown_by_phase: dict[str, Any] = field(default_factory=dict)
    comm_breakdown: dict[str, Any] = field(default_factory=dict)

    # === 配置 metadata ===
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def compute_time(self) -> Any:
        """总计算时间 = forward + backward + recompute."""
        return self.forward_time + self.backward_time + self.recompute_time

    @property
    def overlap_time(self) -> Any:
        """计算/通信重叠时间（基于可校准系数）."""
        if self.overlap_ratio == 0:
            return 0
        return self.overlap_ratio * sym_min(self.compute_time, self.comm_time)

    @property
    def e2e_time(self) -> Any:
        """端到端时间估算.

        e2e = compute + comm + bubble - overlap
        """
        return self.compute_time + self.comm_time + self.bubble_time - self.overlap_time

    @property
    def peak_memory(self) -> Any:
        """峰值内存估算."""
        return (
            self.weight_memory
            + self.activation_memory
            + self.gradient_memory
            + self.optimizer_memory
        )

    def _eval_field(self, value: Any, subs: dict) -> float:
        """对单个字段求值."""
        result = eval_lazy(value, subs)
        if isinstance(result, (int, float)):
            return float(result)
        # 尝试 SymPy subs
        if isinstance(result, Expr):
            try:
                return float(result)
            except (TypeError, ValueError):
                return result
        return result

    def _eval_all(self, subs: dict) -> dict[str, float]:
        """对所有关键字段求值，返回数值字典."""
        return {
            "forward_time": self._eval_field(self.forward_time, subs),
            "backward_time": self._eval_field(self.backward_time, subs),
            "recompute_time": self._eval_field(self.recompute_time, subs),
            "comm_time": self._eval_field(self.comm_time, subs),
            "bubble_time": self._eval_field(self.bubble_time, subs),
            "compute_time": self._eval_field(self.compute_time, subs),
            "overlap_time": self._eval_field(self.overlap_time, subs),
            "e2e_time": self._eval_field(self.e2e_time, subs),
            "weight_memory": self._eval_field(self.weight_memory, subs),
            "activation_memory": self._eval_field(self.activation_memory, subs),
            "gradient_memory": self._eval_field(self.gradient_memory, subs),
            "optimizer_memory": self._eval_field(self.optimizer_memory, subs),
            "peak_memory": self._eval_field(self.peak_memory, subs),
        }

    def sweep(self, **param_ranges) -> list[tuple[dict, dict]]:
        """参数空间扫描.

        对给定参数范围做笛卡尔积，返回每组参数对应的所有指标数值。

        Args:
            **param_ranges: 参数范围，如 batch=[4,8,16], tp=[1,2,4]

        Returns:
            list of (params_dict, metrics_dict)

        Example:
            >>> for params, m in est.sweep(batch=[4,8], tp=[1,2]):
            ...     print(f"batch={params['batch']}, tp={params['tp']}")
            ...     print(f"  e2e={m['e2e_time']:.4f}s, mem={m['peak_memory']:.0f}")
        """
        keys = list(param_ranges.keys())
        values_list = list(param_ranges.values())

        # 将字符串 key 转换为 Symbol
        symbol_keys = [get_symbol(k) if isinstance(k, str) else k for k in keys]

        results = []
        for values in product(*values_list):
            params = dict(zip(keys, values))
            symbol_params = dict(zip(symbol_keys, values))
            metrics = self._eval_all(symbol_params)
            results.append((params, metrics))
        return results

    def sensitivity(self, wrt: str | Symbol) -> dict[str, Any]:
        """敏感性分析 - 对指定符号求偏导.

        将每个聚合字段转换为 SymPy 表达式后求偏导，
        用于分析参数变化对各指标的边际影响。

        Args:
            wrt: 求导变量名 (str) 或 Symbol

        Returns:
            dict: {field_name: derivative_expr}

        Example:
            >>> sens = est.sensitivity('tp')
            >>> print(sens['e2e_time'])  # d(e2e)/d(tp)
        """
        from sympy import diff

        sym = get_symbol(wrt) if isinstance(wrt, str) else wrt

        def _to_sympy_and_diff(value):
            if isinstance(value, (int, float)):
                return 0
            if hasattr(value, "to_sympy"):
                return diff(value.to_sympy(), sym)
            if isinstance(value, Expr):
                return diff(value, sym)
            return 0

        return {
            "forward_time": _to_sympy_and_diff(self.forward_time),
            "backward_time": _to_sympy_and_diff(self.backward_time),
            "recompute_time": _to_sympy_and_diff(self.recompute_time),
            "comm_time": _to_sympy_and_diff(self.comm_time),
            "bubble_time": _to_sympy_and_diff(self.bubble_time),
            "e2e_time": _to_sympy_and_diff(self.e2e_time),
            "weight_memory": _to_sympy_and_diff(self.weight_memory),
            "activation_memory": _to_sympy_and_diff(self.activation_memory),
            "peak_memory": _to_sympy_and_diff(self.peak_memory),
        }

    def bottleneck(
        self, subs: dict | None = None
    ) -> dict[str, list[tuple[str, float, float]]]:
        """关键因素分析 - 识别时间和内存的主导项.

        将各分解项 eval 后排序，返回 [(name, value, fraction)]，
        用于快速定位计算/通信/bubble 中哪个是瓶颈。

        Args:
            subs: 符号替换字典（用于将符号表达式求值为数值）

        Returns:
            dict with 'time' and 'memory' keys, each a sorted list of
            (item_name, value, fraction_of_total)

        Example:
            >>> bn = est.bottleneck(subs={'batch': 16, 'tp': 4, 'pp': 8})
            >>> for name, val, frac in bn['time']:
            ...     print(f"  {name}: {val:.4f}s ({frac:.1%})")
        """
        subs = subs or {}
        # 规范化 subs
        norm_subs = {}
        for k, v in subs.items():
            if isinstance(k, str):
                norm_subs[get_symbol(k)] = v
            else:
                norm_subs[k] = v

        # 时间分解
        time_items = {
            "forward": self.forward_time,
            "backward": self.backward_time,
            "recompute": self.recompute_time,
            "communication": self.comm_time,
            "bubble": self.bubble_time,
        }
        time_values = {}
        for name, expr in time_items.items():
            val = self._eval_field(expr, norm_subs)
            time_values[name] = float(val) if isinstance(val, (int, float)) else 0.0

        time_total = sum(time_values.values())
        time_sorted = sorted(
            [
                (name, val, val / time_total if time_total > 0 else 0)
                for name, val in time_values.items()
            ],
            key=lambda x: -x[1],
        )

        # 内存分解
        mem_items = {
            "weights": self.weight_memory,
            "activations": self.activation_memory,
            "gradients": self.gradient_memory,
            "optimizer_states": self.optimizer_memory,
        }
        mem_values = {}
        for name, expr in mem_items.items():
            val = self._eval_field(expr, norm_subs)
            mem_values[name] = float(val) if isinstance(val, (int, float)) else 0.0

        mem_total = sum(mem_values.values())
        mem_sorted = sorted(
            [
                (name, val, val / mem_total if mem_total > 0 else 0)
                for name, val in mem_values.items()
            ],
            key=lambda x: -x[1],
        )

        return {
            "time": time_sorted,
            "memory": mem_sorted,
        }

    def calibrate(self, sim_result) -> SymbolicEstimate:
        """用 SimulationResult 校准符号化估算.

        主要校准 overlap_ratio：通过对比 Timeline 精确结果，
        反推实际的计算/通信重叠比例。

        Args:
            sim_result: SimulationResult (来自 Timeline 路径)

        Returns:
            校准后的新 SymbolicEstimate
        """
        import copy

        calibrated = copy.copy(self)

        # 校准 overlap_ratio
        if sim_result.total_time_breakdown:
            tb = sim_result.total_time_breakdown
            actual_compute = tb.forward + tb.backward + tb.recompute
            actual_comm = tb.communication
            actual_e2e = sim_result.e2e_time

            # e2e = compute + comm + bubble - overlap
            # overlap = compute + comm + bubble - e2e
            actual_bubble = tb.bubble
            implied_overlap = actual_compute + actual_comm + actual_bubble - actual_e2e

            # overlap_ratio = overlap / min(compute, comm)
            min_cc = min(actual_compute, actual_comm) if actual_comm > 0 else 0
            if min_cc > 0 and implied_overlap > 0:
                calibrated.overlap_ratio = min(1.0, implied_overlap / min_cc)
            else:
                calibrated.overlap_ratio = 0.0

        return calibrated

    def to_latex(self) -> dict[str, str]:
        """导出各字段的 LaTeX 公式.

        Returns:
            dict: {field_name: latex_string}
        """
        from sympy import latex

        def _to_latex(value) -> str:
            if isinstance(value, (int, float)):
                return str(value)
            if hasattr(value, "to_sympy"):
                return latex(value.to_sympy())
            if isinstance(value, Expr):
                return latex(value)
            return str(value)

        return {
            "forward_time": _to_latex(self.forward_time),
            "backward_time": _to_latex(self.backward_time),
            "recompute_time": _to_latex(self.recompute_time),
            "comm_time": _to_latex(self.comm_time),
            "bubble_time": _to_latex(self.bubble_time),
            "e2e_time": _to_latex(self.e2e_time),
            "weight_memory": _to_latex(self.weight_memory),
            "activation_memory": _to_latex(self.activation_memory),
            "gradient_memory": _to_latex(self.gradient_memory),
            "optimizer_memory": _to_latex(self.optimizer_memory),
            "peak_memory": _to_latex(self.peak_memory),
        }

    def __repr__(self) -> str:
        def _fmt(v):
            if isinstance(v, (int, float)):
                if v == 0:
                    return "0"
                return f"{v:.6g}"
            return repr(v)

        return (
            f"SymbolicEstimate(\n"
            f"  time: fw={_fmt(self.forward_time)}, bw={_fmt(self.backward_time)}, "
            f"re={_fmt(self.recompute_time)}, comm={_fmt(self.comm_time)}, "
            f"bubble={_fmt(self.bubble_time)}\n"
            f"  memory: w={_fmt(self.weight_memory)}, act={_fmt(self.activation_memory)}, "
            f"grad={_fmt(self.gradient_memory)}, opt={_fmt(self.optimizer_memory)}\n"
            f"  overlap_ratio={self.overlap_ratio:.2f}\n"
            f")"
        )

"""Floating-point analysis models and NiceGUI presentation panel."""

from __future__ import annotations

from dataclasses import dataclass
from math import inf, isnan
from typing import Any

import numpy as np
from nicegui import ui

from blueprinting.fp import float_point_values_table

DEFAULT_FLOAT_FORMATS = {
    "fp32": (1, 8, 23),
    "tf32": (1, 8, 10),
    "bf16": (1, 8, 7),
    "fp16": (1, 5, 10),
    "fp8(E5M2)": (1, 5, 2),
    "fp8(E4M3)": (1, 4, 3),
    "fp4(E2M1)": (1, 2, 1),
}


@dataclass(frozen=True)
class FloatFormatSpec:
    """Small IEEE-style binary floating-point format."""

    exponent_bits: int
    mantissa_bits: int
    sign_bit: bool = True

    def __post_init__(self) -> None:
        if not 2 <= self.exponent_bits <= 8:
            raise ValueError("exponent_bits must be between 2 and 8")
        if not 0 <= self.mantissa_bits <= 23:
            raise ValueError("mantissa_bits must be between 0 and 23")

    @property
    def total_bits(self) -> int:
        return int(self.sign_bit) + self.exponent_bits + self.mantissa_bits

    @property
    def bias(self) -> int:
        return 2 ** (self.exponent_bits - 1) - 1

    @property
    def min_normal(self) -> float:
        return 2.0 ** (1 - self.bias)

    @property
    def min_subnormal(self) -> float:
        return self.min_normal * 2.0 ** (-self.mantissa_bits)

    @property
    def max_finite(self) -> float:
        return (2.0 - 2.0 ** (-self.mantissa_bits)) * 2.0**self.bias

    @property
    def name(self) -> str:
        return f"FP{self.total_bits}(E{self.exponent_bits}M{self.mantissa_bits})"


@dataclass(frozen=True)
class FloatDecode:
    value: float
    category: str
    raw_exponent: int
    unbiased_exponent: int | None
    significand: float


def decode_float_bits(
    spec: FloatFormatSpec,
    *,
    negative: bool,
    exponent: tuple[bool, ...],
    mantissa: tuple[bool, ...],
) -> FloatDecode:
    """Decode one bit pattern with IEEE zero/subnormal/special handling."""

    if len(exponent) != spec.exponent_bits or len(mantissa) != spec.mantissa_bits:
        raise ValueError("bit vector lengths do not match the floating-point format")
    raw_exponent = _bits_to_int(exponent)
    fraction = sum(int(bit) * 2.0 ** -(index + 1) for index, bit in enumerate(mantissa))
    sign = -1.0 if negative and spec.sign_bit else 1.0
    max_exponent = 2**spec.exponent_bits - 1
    if raw_exponent == 0:
        if fraction == 0:
            return FloatDecode(sign * 0.0, "zero", raw_exponent, 1 - spec.bias, 0.0)
        return FloatDecode(
            sign * 2.0 ** (1 - spec.bias) * fraction,
            "subnormal",
            raw_exponent,
            1 - spec.bias,
            fraction,
        )
    if raw_exponent == max_exponent:
        category = "infinity" if fraction == 0 else "nan"
        return FloatDecode(
            sign * inf if category == "infinity" else float("nan"), category, raw_exponent, None, fraction
        )
    unbiased = raw_exponent - spec.bias
    significand = 1.0 + fraction
    return FloatDecode(sign * 2.0**unbiased * significand, "normal", raw_exponent, unbiased, significand)


def format_layout_chart_options(custom: FloatFormatSpec) -> dict[str, Any]:
    formats = {**DEFAULT_FLOAT_FORMATS, "custom": (int(custom.sign_bit), custom.exponent_bits, custom.mantissa_bits)}
    names = list(formats)
    return {
        "backgroundColor": "transparent",
        "animationDuration": 250,
        "legend": {"top": 0, "data": ["Sign", "Exponent", "Mantissa"]},
        "grid": {"left": 100, "right": 32, "top": 42, "bottom": 32},
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "xAxis": {"type": "value", "name": "bits", "max": 32},
        "yAxis": {"type": "category", "data": names},
        "series": [
            _format_series("Sign", "#ef4444", [formats[name][0] for name in names]),
            _format_series("Exponent", "#22c55e", [formats[name][1] for name in names]),
            _format_series("Mantissa", "#3b82f6", [formats[name][2] for name in names]),
        ],
    }


def representable_values(spec: FloatFormatSpec) -> list[float]:
    """Enumerate finite values for an interaction-sized format."""

    if spec.exponent_bits + spec.mantissa_bits > 12:
        raise ValueError("interactive value enumeration is limited to exponent_bits + mantissa_bits <= 12")
    return float_point_values_table(
        sign_bit=spec.sign_bit,
        exponent_bits=spec.exponent_bits,
        mantissa_bits=spec.mantissa_bits,
    )


def distribution_chart_options(values: list[float], spec: FloatFormatSpec, limit: float) -> dict[str, Any]:
    visible = _downsample([value for value in values if abs(value) <= limit], 1200)
    return {
        "backgroundColor": "transparent",
        "animationDuration": 200,
        "grid": {"left": 54, "right": 24, "top": 24, "bottom": 42},
        "tooltip": {"trigger": "item", "formatter": "{c}"},
        "xAxis": {"type": "value", "name": "representable value", "min": -limit, "max": limit},
        "yAxis": {"type": "value", "show": False, "min": -1, "max": 1},
        "series": [
            {
                "name": "Subnormal" if category else "Normal",
                "type": "scatter",
                "symbol": "rect",
                "symbolSize": [2, 18],
                "itemStyle": {"color": "#dc2626" if category else "#2563eb"},
                "data": [[value, 0] for value in visible if (0 < abs(value) < spec.min_normal) == category],
            }
            for category in (False, True)
        ],
    }


def quantization_chart_options(values: list[float], limit: float) -> dict[str, Any]:
    finite = np.asarray(sorted(set(values)), dtype=np.float64)
    samples = np.linspace(-limit, limit, 501)
    indices = np.searchsorted(finite, samples, side="left")
    indices = np.clip(indices, 1, len(finite) - 1)
    lower = finite[indices - 1]
    upper = finite[indices]
    nearest = np.where(np.abs(samples - lower) <= np.abs(upper - samples), lower, upper)
    error = np.abs(samples - nearest)
    return {
        "backgroundColor": "transparent",
        "animationDuration": 200,
        "grid": {"left": 64, "right": 24, "top": 24, "bottom": 48},
        "tooltip": {"trigger": "axis"},
        "xAxis": {"type": "value", "name": "input"},
        "yAxis": {"type": "value", "name": "absolute error"},
        "series": [
            {
                "name": "Quantization error",
                "type": "line",
                "showSymbol": False,
                "lineStyle": {"width": 1, "color": "#7c3aed"},
                "data": [[float(x), float(y)] for x, y in zip(samples, error)],
            }
        ],
    }


def operation_impact_rows(values: list[float], spec: FloatFormatSpec) -> list[dict[str, Any]]:
    sample = np.asarray(_downsample(values, 64), dtype=np.float64)
    left = sample[:, None]
    right = sample[None, :]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        results = {
            "A + B": left + right,
            "A − B": left - right,
            "A × B": left * right,
            "A ÷ B": left / right,
        }
    rows = []
    for name, result in results.items():
        absolute = np.abs(result)
        rows.append(
            {
                "operation": name,
                "samples": int(result.size),
                "normal": int(
                    np.count_nonzero(
                        np.isfinite(result) & (absolute >= spec.min_normal) & (absolute <= spec.max_finite)
                    )
                ),
                "subnormal": int(np.count_nonzero((absolute >= spec.min_subnormal) & (absolute < spec.min_normal))),
                "underflow_or_zero": int(np.count_nonzero(absolute < spec.min_subnormal)),
                "overflow_or_nan": int(np.count_nonzero(~np.isfinite(result) | (absolute > spec.max_finite))),
            }
        )
    return rows


class FloatAnalysisPanel:
    """Stateful NiceGUI panel for interactive floating-point exploration."""

    def __init__(self) -> None:
        self.spec = FloatFormatSpec(5, 2)
        self.negative = False
        self.exponent = (False,) * self.spec.exponent_bits
        self.mantissa = (False,) * self.spec.mantissa_bits
        self.limit = 0.1
        self.content: Any | None = None

    def build(self) -> None:
        with ui.column().classes("w-full gap-4").mark("float-analysis"):
            with (
                ui.element("section").classes("bp-evidence-surface"),
                ui.element("div").classes("bp-evidence-section"),
            ):
                ui.label("NUMERIC LENS").classes("bp-kicker")
                ui.label("浮点数分析").classes("bp-result-title mt-1")
                ui.label("探索格式位宽、数值编码、动态范围、可表示值分布和运算边界。 ").classes("bp-card-copy mt-1")
                with ui.row().classes("w-full items-end gap-3 mt-3"):
                    self.exponent_control = (
                        ui.number(
                            "指数位",
                            value=self.spec.exponent_bits,
                            min=2,
                            max=6,
                            step=1,
                            precision=0,
                            on_change=self._format_changed,
                        )
                        .props("outlined dense")
                        .mark("float-exponent-bits")
                    )
                    self.mantissa_control = (
                        ui.number(
                            "尾数位",
                            value=self.spec.mantissa_bits,
                            min=0,
                            max=6,
                            step=1,
                            precision=0,
                            on_change=self._format_changed,
                        )
                        .props("outlined dense")
                        .mark("float-mantissa-bits")
                    )
                    self.sign_control = ui.switch(
                        "符号位", value=self.spec.sign_bit, on_change=self._format_changed
                    ).mark("float-sign-bit")
                    self.range_control = (
                        ui.select(
                            {0.001: "±0.001", 0.01: "±0.01", 0.1: "±0.1", 1.0: "±1", 10.0: "±10"},
                            value=self.limit,
                            label="观察范围",
                            on_change=self._range_changed,
                        )
                        .props("outlined dense")
                        .mark("float-range")
                    )
            self.content = ui.column().classes("w-full gap-4")
            self._render_content()

    def _format_changed(self, _: Any) -> None:
        if self.exponent_control.value is None or self.mantissa_control.value is None:
            return
        exp = int(self.exponent_control.value)
        man = int(self.mantissa_control.value)
        sign = bool(self.sign_control.value)
        self.spec = FloatFormatSpec(exp, man, sign)
        self.exponent = (False,) * exp
        self.mantissa = (False,) * man
        self._render_content()

    def _range_changed(self, event: Any) -> None:
        self.limit = float(event.value)
        self._render_content()

    def _set_sign(self, event: Any) -> None:
        self.negative = bool(event.value)
        self._render_content()

    def _set_exponent_bit(self, index: int, event: Any) -> None:
        bits = list(self.exponent)
        bits[index] = bool(event.value)
        self.exponent = tuple(bits)
        self._render_content()

    def _set_mantissa_bit(self, index: int, event: Any) -> None:
        bits = list(self.mantissa)
        bits[index] = bool(event.value)
        self.mantissa = tuple(bits)
        self._render_content()

    def _render_content(self) -> None:
        if self.content is None:
            return
        self.content.clear()
        values = representable_values(self.spec)
        decoded = decode_float_bits(
            self.spec,
            negative=self.negative,
            exponent=self.exponent,
            mantissa=self.mantissa,
        )
        with self.content:
            with (
                ui.element("section").classes("bp-evidence-surface"),
                ui.element("div").classes("bp-evidence-section"),
            ):
                ui.label("格式位宽对比").classes("bp-card-title")
                ui.echart(format_layout_chart_options(self.spec), renderer="svg").classes("w-full h-80")
            with (
                ui.element("section").classes("bp-evidence-surface"),
                ui.element("div").classes("bp-evidence-section"),
            ):
                ui.label(f"{self.spec.name} 位级计算器").classes("bp-card-title")
                ui.label("指数全 0/全 1 分别按 subnormal/zero 与 Inf/NaN 处理。 ").classes("bp-card-copy")
                with ui.row().classes("w-full items-start gap-5 mt-3"):
                    ui.switch("S", value=self.negative, on_change=self._set_sign).mark("float-value-sign")
                    with ui.column().classes("gap-1"):
                        ui.label("Exponent").classes("bp-summary-label")
                        with ui.row().classes("gap-1"):
                            for index, value in enumerate(self.exponent):
                                ui.checkbox(
                                    f"E{index}",
                                    value=value,
                                    on_change=lambda event, i=index: self._set_exponent_bit(i, event),
                                ).props("dense")
                    with ui.column().classes("gap-1"):
                        ui.label("Mantissa").classes("bp-summary-label")
                        with ui.row().classes("gap-1"):
                            for index, value in enumerate(self.mantissa):
                                ui.checkbox(
                                    f"M{index}",
                                    value=value,
                                    on_change=lambda event, i=index: self._set_mantissa_bit(i, event),
                                ).props("dense")
                with ui.element("div").classes("bp-chain-stats mt-3"):
                    for label, value in (
                        ("Category", decoded.category),
                        ("Raw exponent", str(decoded.raw_exponent)),
                        ("Significand", f"{decoded.significand:.8g}"),
                        ("Value", _format_float(decoded.value)),
                    ):
                        with ui.column().classes("gap-0"):
                            ui.label(label).classes("bp-summary-label")
                            ui.label(value).classes("bp-result-title bp-mono")
            with (
                ui.element("section").classes("bp-evidence-surface"),
                ui.element("div").classes("bp-evidence-section"),
            ):
                ui.label("动态范围与可表示值").classes("bp-card-title")
                with ui.element("div").classes("bp-chain-stats mt-2"):
                    for label, value in (
                        ("Bias", str(self.spec.bias)),
                        ("Min subnormal", f"{self.spec.min_subnormal:.4e}"),
                        ("Min normal", f"{self.spec.min_normal:.4e}"),
                        ("Max finite", f"{self.spec.max_finite:.4e}"),
                    ):
                        with ui.column().classes("gap-0"):
                            ui.label(label).classes("bp-summary-label")
                            ui.label(value).classes("bp-card-copy bp-mono")
                ui.echart(distribution_chart_options(values, self.spec, self.limit)).classes("w-full h-64")
                ui.echart(quantization_chart_options(values, self.limit)).classes("w-full h-64")
            with (
                ui.element("section").classes("bp-evidence-surface"),
                ui.element("div").classes("bp-evidence-section"),
            ):
                ui.label("四则运算范围影响").classes("bp-card-title")
                ui.label("对最多 64 个均匀抽样的可表示值进行笛卡尔组合；这是数值范围分析，不是硬件执行时间。 ").classes(
                    "bp-card-copy"
                )
                ui.aggrid(
                    {
                        "columnDefs": [
                            {"headerName": "Operation", "field": "operation", "pinned": "left"},
                            {"headerName": "Samples", "field": "samples", "type": "numericColumn"},
                            {"headerName": "Normal", "field": "normal", "type": "numericColumn"},
                            {"headerName": "Subnormal", "field": "subnormal", "type": "numericColumn"},
                            {"headerName": "Zero/underflow", "field": "underflow_or_zero", "type": "numericColumn"},
                            {"headerName": "Overflow/NaN", "field": "overflow_or_nan", "type": "numericColumn"},
                        ],
                        "rowData": operation_impact_rows(values, self.spec),
                    },
                    theme="quartz",
                ).classes("w-full").style("height: 250px")


def _format_series(name: str, color: str, data: list[int]) -> dict[str, Any]:
    return {"name": name, "type": "bar", "stack": "bits", "data": data, "itemStyle": {"color": color}}


def _bits_to_int(bits: tuple[bool, ...]) -> int:
    return sum(int(bit) << (len(bits) - index - 1) for index, bit in enumerate(bits))


def _downsample(values: list[float], maximum: int) -> list[float]:
    if len(values) <= maximum:
        return values
    indices = np.linspace(0, len(values) - 1, maximum, dtype=int)
    return [values[index] for index in indices]


def _format_float(value: float) -> str:
    if isnan(value):
        return "NaN"
    if value == inf:
        return "+Inf"
    if value == -inf:
        return "−Inf"
    return f"{value:.10g}"


__all__ = [
    "DEFAULT_FLOAT_FORMATS",
    "FloatAnalysisPanel",
    "FloatDecode",
    "FloatFormatSpec",
    "decode_float_bits",
    "distribution_chart_options",
    "format_layout_chart_options",
    "operation_impact_rows",
    "quantization_chart_options",
    "representable_values",
]

from __future__ import annotations

from math import inf, isnan

import pytest

from blueprinting.workbench.float_analysis import (
    FloatFormatSpec,
    decode_float_bits,
    distribution_chart_options,
    operation_impact_rows,
    quantization_chart_options,
    representable_values,
)


def test_float_format_reports_ieee_style_dynamic_range() -> None:
    spec = FloatFormatSpec(5, 10)

    assert spec.name == "FP16(E5M10)"
    assert spec.bias == 15
    assert spec.min_normal == pytest.approx(2**-14)
    assert spec.min_subnormal == pytest.approx(2**-24)
    assert spec.max_finite == pytest.approx(65504)


def test_decode_float_bits_distinguishes_zero_subnormal_normal_and_special_values() -> None:
    spec = FloatFormatSpec(2, 1)

    zero = decode_float_bits(spec, negative=False, exponent=(False, False), mantissa=(False,))
    subnormal = decode_float_bits(spec, negative=False, exponent=(False, False), mantissa=(True,))
    normal = decode_float_bits(spec, negative=True, exponent=(False, True), mantissa=(True,))
    infinity = decode_float_bits(spec, negative=False, exponent=(True, True), mantissa=(False,))
    nan = decode_float_bits(spec, negative=False, exponent=(True, True), mantissa=(True,))

    assert (zero.category, zero.value) == ("zero", 0.0)
    assert (subnormal.category, subnormal.value) == ("subnormal", 0.5)
    assert (normal.category, normal.value) == ("normal", -1.5)
    assert infinity.value == inf
    assert isnan(nan.value)


def test_interactive_enumeration_rejects_explosive_formats() -> None:
    with pytest.raises(ValueError, match="limited"):
        representable_values(FloatFormatSpec(8, 23))


def test_float_analysis_charts_and_operation_summary_are_complete() -> None:
    spec = FloatFormatSpec(4, 3)
    values = representable_values(spec)

    distribution = distribution_chart_options(values, spec, 1.0)
    quantization = quantization_chart_options(values, 1.0)
    operations = operation_impact_rows(values, spec)

    assert {series["name"] for series in distribution["series"]} == {"Normal", "Subnormal"}
    assert len(quantization["series"][0]["data"]) == 501
    assert [row["operation"] for row in operations] == ["A + B", "A − B", "A × B", "A ÷ B"]
    assert all(
        row["normal"] + row["subnormal"] + row["underflow_or_zero"] + row["overflow_or_nan"] == row["samples"]
        for row in operations
    )

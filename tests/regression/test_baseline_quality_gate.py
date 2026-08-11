from __future__ import annotations

from pathlib import Path

import pytest

from blueprinting.validation import (
    BaselineRegressionGate,
    RegressionCheck,
    run_inference_baseline_regression,
    run_training_baseline_regression,
)

ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.baseline_regression


def test_training_calculon_baseline_regression_gate():
    gate = run_training_baseline_regression(ROOT)

    gate.require()
    assert gate.ok
    assert len(gate.checks) >= 20


def test_inference_vidur_baseline_regression_gate():
    gate = run_inference_baseline_regression(ROOT)

    gate.require()
    assert gate.ok
    assert len(gate.checks) >= 30


def test_regression_gate_reports_every_failed_predicate():
    gate = BaselineRegressionGate(
        schema="blueprinting.baseline-regression-gate.v0",
        domain="test/baseline",
        checks=(
            RegressionCheck("first", False, "<= 1", 2),
            RegressionCheck("second", False, "exactly 'stable'", "drifted"),
        ),
    )

    with pytest.raises(AssertionError) as captured:
        gate.require()

    assert "first: expected <= 1; actual=2" in str(captured.value)
    assert "second: expected exactly 'stable'; actual='drifted'" in str(captured.value)

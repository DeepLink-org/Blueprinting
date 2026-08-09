"""Reproducible formal-synthesis validation experiments."""

from .calculon import (
    CalculonCase,
    CalculonExperimentReport,
    discover_seqsel_tab5_cases,
    run_calculon_experiment,
)
from .regression import (
    BaselineRegressionGate,
    RegressionCheck,
    run_inference_baseline_regression,
    run_training_baseline_regression,
)
from .vidur import (
    VidurCaseReport,
    VidurComponentComparison,
    VidurExperimentCase,
    VidurExperimentReport,
    VidurPhaseComparison,
    compare_inference_phase_to_vidur,
    run_vidur_experiment,
)

__all__ = [
    "CalculonCase",
    "CalculonExperimentReport",
    "BaselineRegressionGate",
    "RegressionCheck",
    "discover_seqsel_tab5_cases",
    "run_calculon_experiment",
    "run_inference_baseline_regression",
    "run_training_baseline_regression",
    "VidurCaseReport",
    "VidurComponentComparison",
    "VidurExperimentCase",
    "VidurExperimentReport",
    "VidurPhaseComparison",
    "compare_inference_phase_to_vidur",
    "run_vidur_experiment",
]

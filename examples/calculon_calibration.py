#!/usr/bin/env python3
"""Run the canonical Blueprinting/Calculon calibration experiment.

Examples:
    uv run python examples/calculon_calibration.py
    uv run python examples/calculon_calibration.py --output /tmp/report.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from blueprinting.synthesizer.experiments import (  # noqa: E402
    discover_seqsel_tab5_cases,
    run_calculon_experiment,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Derive and calibrate the SeqSel/Calculon comparison cases")
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    arguments = parser.parse_args()

    report = run_calculon_experiment(discover_seqsel_tab5_cases(ROOT / "data"))
    payload = report.to_json()
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload, encoding="utf-8")

    print("Blueprinting synthesis ↔ Calculon calibration")
    print(f"hardware evidence: {report.hardware_name} / {report.evidence_revision}")
    print(
        "mean absolute error: "
        f"peak-only={report.peak_mean_absolute_error_percent:.3f}%  "
        f"system-evidence={report.calibrated_mean_absolute_error_percent:.6f}%"
    )
    print()
    print(f"{'case':42} {'peak':>10} {'calibrated':>12} {'Calculon':>10} {'error':>9}")
    for case in report.cases:
        print(
            f"{case.case:42} "
            f"{case.peak_only.total:10.4f} "
            f"{case.calibrated.total:12.4f} "
            f"{case.calculon_total_seconds:10.4f} "
            f"{case.calibrated_error_percent:+8.4f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

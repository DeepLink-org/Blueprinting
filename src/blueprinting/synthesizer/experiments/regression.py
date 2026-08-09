"""Repository quality gates for independent training and inference baselines."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...analysis import HardwareProfile, VidurProfileBaseline
from ..bindings import InferencePhase
from ..models import TransformerInferenceExecutionSpec, TransformerModelSpec
from .calculon import CalculonExperimentReport, discover_seqsel_tab5_cases, run_calculon_experiment
from .vidur import VidurExperimentCase, VidurExperimentReport, run_vidur_experiment

_CONTRACT_PATH = Path("data/validation/baseline_regression_contract.json")


@dataclass(frozen=True)
class RegressionCheck:
    """One named, inspectable quality-gate predicate."""

    name: str
    passed: bool
    expected: str
    actual: Any

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "expected": self.expected,
            "actual": self.actual,
        }


@dataclass(frozen=True)
class BaselineRegressionGate:
    """A deterministic set of checks suitable for tests and CI artifacts."""

    schema: str
    domain: str
    checks: tuple[RegressionCheck, ...]

    @property
    def ok(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def failures(self) -> tuple[RegressionCheck, ...]:
        return tuple(check for check in self.checks if not check.passed)

    def require(self) -> None:
        if self.ok:
            return
        details = "\n".join(
            f"- {check.name}: expected {check.expected}; actual={check.actual!r}" for check in self.failures
        )
        raise AssertionError(f"{self.domain} baseline regression failed:\n{details}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "domain": self.domain,
            "ok": self.ok,
            "checks": [check.to_dict() for check in self.checks],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _exact(name: str, actual: Any, expected: Any) -> RegressionCheck:
    return RegressionCheck(name, actual == expected, f"exactly {expected!r}", actual)


def _close(name: str, actual: float | None, expected: float, *, absolute_tolerance: float) -> RegressionCheck:
    passed = actual is not None and math.isclose(actual, expected, rel_tol=0.0, abs_tol=absolute_tolerance)
    return RegressionCheck(name, passed, f"{expected!r} ± {absolute_tolerance:g}", actual)


def _at_most(name: str, actual: float | None, maximum: float) -> RegressionCheck:
    return RegressionCheck(name, actual is not None and actual <= maximum, f"<= {maximum!r}", actual)


def _at_least(name: str, actual: float | None, minimum: float) -> RegressionCheck:
    return RegressionCheck(name, actual is not None and actual >= minimum, f">= {minimum!r}", actual)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _contract(repository_root: Path) -> dict[str, Any]:
    contract = _read_json(repository_root / _CONTRACT_PATH)
    if contract.get("schema") != "blueprinting.baseline-regression-contract.v1":
        raise ValueError("unsupported baseline regression contract schema")
    return contract


def _sha256(path: Path) -> str:
    digester = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digester.update(chunk)
    return digester.hexdigest()


def _fixture_path(fixture_root: Path, relative_path: str) -> Path:
    path = (fixture_root / relative_path).resolve()
    if not path.is_relative_to(fixture_root):
        raise ValueError(f"fixture path escapes validation slice: {relative_path!r}")
    return path


def _training_checks(report: CalculonExperimentReport, contract: dict[str, Any]) -> tuple[RegressionCheck, ...]:
    budgets = contract["budgets"]
    golden = contract["golden"]
    checks = [
        _exact("training.case_count", len(report.cases), contract["case_count"]),
        _exact("training.evidence_revision", report.evidence_revision, contract["evidence_revision"]),
        _at_most(
            "training.workload_max_absolute_error_percent",
            report.workload_max_absolute_error_percent,
            budgets["workload_max_absolute_error_percent"],
        ),
        _at_most(
            "training.calculon_mean_absolute_error_percent",
            report.calibrated_mean_absolute_error_percent,
            budgets["calculon_mean_absolute_error_percent"],
        ),
        _at_most(
            "training.calculon_max_absolute_error_percent",
            report.calibrated_max_absolute_error_percent,
            budgets["calculon_max_absolute_error_percent"],
        ),
        _at_most(
            "training.paper_mean_absolute_error_percent",
            report.paper_mean_absolute_error_percent,
            budgets["paper_mean_absolute_error_percent"],
        ),
        _at_most(
            "training.paper_max_absolute_error_percent",
            report.paper_max_absolute_error_percent,
            budgets["paper_max_absolute_error_percent"],
        ),
        _close(
            "training.golden.peak_mean_absolute_error_percent",
            report.peak_mean_absolute_error_percent,
            golden["peak_mean_absolute_error_percent"],
            absolute_tolerance=1e-12,
        ),
        _close(
            "training.golden.calculon_mean_absolute_error_percent",
            report.calibrated_mean_absolute_error_percent,
            golden["calculon_mean_absolute_error_percent"],
            absolute_tolerance=1e-12,
        ),
        _close(
            "training.golden.paper_mean_absolute_error_percent",
            report.paper_mean_absolute_error_percent,
            golden["paper_mean_absolute_error_percent"],
            absolute_tolerance=1e-12,
        ),
        _close(
            "training.golden.paper_max_absolute_error_percent",
            report.paper_max_absolute_error_percent,
            golden["paper_max_absolute_error_percent"],
            absolute_tolerance=1e-12,
        ),
        _exact(
            "training.policy.fit_against_case_outputs", report.calibration_policy["fit_against_case_outputs"], False
        ),
    ]
    memory_error = max(
        abs(case.calibrated.memory.total - case.calculon_stats["proc_mem_tier1_cap_req"]) for case in report.cases
    )
    checks.append(
        _at_most(
            "training.memory_max_absolute_error_bytes",
            memory_error,
            budgets["memory_max_absolute_error_bytes"],
        )
    )
    reports_by_name = {case.case: case for case in report.cases}
    checks.append(
        _exact(
            "training.case_names",
            tuple(reports_by_name),
            tuple(golden["portable_digests"]),
        )
    )
    for case_name, expected_digest in golden["portable_digests"].items():
        case = reports_by_name.get(case_name)
        checks.append(
            _exact(
                f"training.{case_name}.portable_digest",
                case.portable_digest if case is not None else None,
                expected_digest,
            )
        )
    return tuple(checks)


def run_training_baseline_regression(repository_root: str | Path) -> BaselineRegressionGate:
    """Run all eight Calculon/SeqSel cases against their frozen contract."""

    root = Path(repository_root).resolve()
    contract = _contract(root)["training"]
    report = run_calculon_experiment(discover_seqsel_tab5_cases(root / "data"))
    return BaselineRegressionGate(
        schema="blueprinting.baseline-regression-gate.v1",
        domain="training/calculon",
        checks=_training_checks(report, contract),
    )


def _load_vidur_report(
    repository_root: Path,
    contract: dict[str, Any],
) -> tuple[VidurExperimentReport, dict[str, Any], tuple[RegressionCheck, ...]]:
    fixture_root = (repository_root / contract["fixture"]).resolve()
    if not fixture_root.is_relative_to(repository_root):
        raise ValueError("Vidur fixture path escapes the repository")
    manifest_path = fixture_root / "manifest.json"
    manifest = _read_json(manifest_path)
    license_path = _fixture_path(fixture_root, manifest["source"]["license_file"])
    checks = [
        _exact(
            "inference.fixture.manifest_sha256",
            _sha256(manifest_path),
            contract["fixture_manifest_sha256"],
        ),
        _exact("inference.fixture.schema", manifest.get("schema"), "blueprinting.vidur-validation-slice.v1"),
        _exact(
            "inference.fixture.source_repository",
            manifest["source"].get("repository"),
            contract["source_repository"],
        ),
        _exact(
            "inference.fixture.source_revision",
            manifest["source"].get("revision"),
            contract["source_revision"],
        ),
        _exact("inference.fixture.license", manifest["source"].get("license"), "MIT"),
        _exact(
            "inference.fixture.license_file",
            manifest["source"].get("license_file"),
            contract["license_file"],
        ),
        _exact(
            "inference.fixture.license_file_exists",
            license_path.is_file(),
            True,
        ),
        _exact(
            "inference.fixture.license_sha256",
            _sha256(license_path),
            contract["license_sha256"],
        ),
    ]
    for file_name, metadata in manifest["files"].items():
        fixture_path = _fixture_path(fixture_root, file_name)
        checks.append(
            _exact(
                f"inference.fixture.{file_name}.sha256",
                _sha256(fixture_path),
                metadata["sha256"],
            )
        )

    model = TransformerModelSpec(**manifest["blueprinting"]["model"])
    execution = TransformerInferenceExecutionSpec(**manifest["blueprinting"]["execution"])
    hardware_manifest = manifest["blueprinting"]["hardware"]
    hardware = HardwareProfile.from_mapping(
        hardware_manifest["name"],
        _read_json(repository_root / hardware_manifest["profile"]),
        datatype=execution.datatype,
    )
    baseline = VidurProfileBaseline.from_csv(
        attention_csv=_fixture_path(fixture_root, "attention.csv"),
        compute_csv=_fixture_path(fixture_root, "mlp.csv"),
        model_name=model.name,
        hardware_name=hardware.name,
        attention_backend=manifest["selection"]["attention_backend"],
        block_size=manifest["selection"]["block_size"],
        source_revision=manifest["source"]["revision"],
        datatype=execution.datatype,
    )
    cases = tuple(
        VidurExperimentCase(
            name=(f"phi2-a100-tp1/{case_data['phase']}/b{case_data['batch_size']}-c{case_data['context_tokens']}"),
            model=model,
            execution=execution,
            hardware=hardware,
            phase=InferencePhase(case_data["phase"]),
            batch_size=case_data["batch_size"],
            context_tokens=case_data["context_tokens"],
        )
        for case_data in manifest["selection"]["cases"]
    )
    return run_vidur_experiment(cases, baseline), manifest, tuple(checks)


def _inference_checks(
    report: VidurExperimentReport,
    contract: dict[str, Any],
    fixture_checks: tuple[RegressionCheck, ...],
) -> tuple[RegressionCheck, ...]:
    budgets = contract["budgets"]
    golden = contract["golden"]
    comparable_subtotal_errors = tuple(
        abs(case.system_evidence.comparable_subtotal_relative_error_percent)
        for case in report.cases
        if case.system_evidence.comparable_subtotal_relative_error_percent is not None
    )
    maximum_comparable_subtotal_error = max(comparable_subtotal_errors) if comparable_subtotal_errors else None
    peak_subtotal_mape = report.peak_comparable_subtotal_mean_absolute_error_percent
    system_subtotal_mape = report.system_evidence_comparable_subtotal_mean_absolute_error_percent
    system_component_mape = report.system_evidence_component_mean_absolute_error_percent
    system_component_max = report.system_evidence_component_max_absolute_error_percent
    improvement = (
        peak_subtotal_mape - system_subtotal_mape
        if peak_subtotal_mape is not None and system_subtotal_mape is not None
        else None
    )
    checks = [
        *fixture_checks,
        _exact("inference.case_count", len(report.cases), contract["case_count"]),
        _exact("inference.baseline_revision", report.baseline_revision, contract["baseline_revision"]),
        _exact("inference.policy.baseline_role", report.policy["baseline_role"], "post-hoc-comparison-only"),
        _exact("inference.policy.oracle_read_during_lowering", report.policy["oracle_read_during_lowering"], False),
        _exact("inference.policy.oracle_read_during_costing", report.policy["oracle_read_during_costing"], False),
        _exact("inference.policy.fit_against_case_outputs", report.policy["fit_against_case_outputs"], False),
        _exact(
            "inference.policy.validation_claim",
            report.policy["validation_claim"],
            contract["validation_claim"],
        ),
        _exact(
            "inference.policy.topology_equivalence",
            report.policy["topology_equivalence"],
            "not-claimed-by-raw-component-profile-alignment",
        ),
        _at_most(
            "inference.system_evidence_comparable_subtotal_mean_absolute_error_percent",
            system_subtotal_mape,
            budgets["system_evidence_comparable_subtotal_mean_absolute_error_percent"],
        ),
        _at_most(
            "inference.system_evidence_comparable_subtotal_max_absolute_error_percent",
            maximum_comparable_subtotal_error,
            budgets["system_evidence_comparable_subtotal_max_absolute_error_percent"],
        ),
        _at_least(
            "inference.improvement_over_peak_percentage_points",
            improvement,
            budgets["minimum_improvement_over_peak_percentage_points"],
        ),
        _close(
            "inference.golden.peak_comparable_subtotal_mean_absolute_error_percent",
            peak_subtotal_mape,
            golden["peak_comparable_subtotal_mean_absolute_error_percent"],
            absolute_tolerance=1e-9,
        ),
        _close(
            "inference.golden.system_evidence_comparable_subtotal_mean_absolute_error_percent",
            system_subtotal_mape,
            golden["system_evidence_comparable_subtotal_mean_absolute_error_percent"],
            absolute_tolerance=1e-9,
        ),
        _close(
            "inference.golden.system_evidence_component_mean_absolute_error_percent",
            system_component_mape,
            golden["system_evidence_component_mean_absolute_error_percent"],
            absolute_tolerance=1e-9,
        ),
        _close(
            "inference.golden.system_evidence_component_max_absolute_error_percent",
            system_component_max,
            golden["system_evidence_component_max_absolute_error_percent"],
            absolute_tolerance=1e-9,
        ),
    ]
    reports_by_name = {case.case: case for case in report.cases}
    checks.append(_exact("inference.case_names", tuple(reports_by_name), tuple(golden["cases"])))
    for case_name, expected in golden["cases"].items():
        case = reports_by_name.get(case_name)
        if case is None:
            checks.append(_exact(f"inference.{case_name}.present", False, True))
            continue
        comparison = case.system_evidence
        checks.extend(
            (
                _exact(f"inference.{case_name}.model_digest", case.model_digest, contract["model_digest"]),
                _exact(
                    f"inference.{case_name}.distributed_digest",
                    case.distributed_digest,
                    expected["distributed_digest"],
                ),
                _exact(
                    f"inference.{case_name}.portable_digest",
                    case.portable_digest,
                    expected["portable_digest"],
                ),
                _at_least(
                    f"inference.{case_name}.component_coverage_budget",
                    comparison.component_coverage,
                    budgets["minimum_semantic_component_coverage"],
                ),
                _close(
                    f"inference.{case_name}.component_coverage_golden",
                    comparison.component_coverage,
                    expected["component_coverage"],
                    absolute_tolerance=1e-12,
                ),
                _close(
                    f"inference.{case_name}.baseline_comparable_block_seconds",
                    comparison.baseline_comparable_block_seconds,
                    expected["baseline_comparable_block_seconds"],
                    absolute_tolerance=1e-15,
                ),
                _close(
                    f"inference.{case_name}.system_evidence_comparable_block_seconds",
                    comparison.estimated_comparable_block_seconds,
                    expected["system_evidence_comparable_block_seconds"],
                    absolute_tolerance=1e-15,
                ),
                _close(
                    f"inference.{case_name}.system_evidence_comparable_subtotal_error_percent",
                    comparison.comparable_subtotal_relative_error_percent,
                    expected["system_evidence_comparable_subtotal_error_percent"],
                    absolute_tolerance=1e-9,
                ),
                _close(
                    f"inference.{case_name}.system_evidence_component_mean_absolute_error_percent",
                    comparison.component_mean_absolute_error_percent,
                    expected["system_evidence_component_mean_absolute_error_percent"],
                    absolute_tolerance=1e-9,
                ),
                _close(
                    f"inference.{case_name}.system_evidence_component_max_absolute_error_percent",
                    comparison.component_max_absolute_error_percent,
                    expected["system_evidence_component_max_absolute_error_percent"],
                    absolute_tolerance=1e-9,
                ),
            )
        )
    return tuple(checks)


def run_inference_baseline_regression(repository_root: str | Path) -> BaselineRegressionGate:
    """Run the pinned offline Vidur slice against its accuracy and golden contract."""

    root = Path(repository_root).resolve()
    contract = _contract(root)["inference"]
    report, _, fixture_checks = _load_vidur_report(root, contract)
    return BaselineRegressionGate(
        schema="blueprinting.baseline-regression-gate.v1",
        domain="inference/vidur",
        checks=_inference_checks(report, contract, fixture_checks),
    )

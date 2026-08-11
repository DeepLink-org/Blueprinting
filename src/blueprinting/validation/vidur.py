"""Vidur baseline comparison for Blueprinting's independent inference path.

Vidur data is read only after ModelIR -> DistributedTaskIR -> PortablePlanIR
lowering and Blueprinting cost evaluation have completed.  The comparison
therefore measures agreement; it cannot make the estimated result agree by
construction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from blueprinting.analysis import (
    CalibrationMode,
    InferenceBaseline,
    InferencePhaseEstimate,
    estimate_inference_phase,
    inference_evidence_query_for,
)
from blueprinting.mapping import NetworkTierBinding, TransformerInferenceMappingSpec
from blueprinting.synthesizer.bindings import InferencePhase
from blueprinting.synthesizer.dialects.transformer import (
    TransformerInferencePlanSemantic,
    TransformerInferencePlanTaskSemantic,
)
from blueprinting.synthesizer.frontend import build_transformer_inference_model_ir, inference_synthesis_session_for
from blueprinting.synthesizer.passes import PassManager, PassPipeline
from blueprinting.synthesizer.stages.distributed.passes import DistributeTransformerInferencePass
from blueprinting.synthesizer.stages.portable_plan.ir import PortablePlanIR
from blueprinting.synthesizer.stages.portable_plan.passes import PlanTransformerInferencePass
from blueprinting.system import SystemProfile
from blueprinting.workload import TransformerModelSpec


@dataclass(frozen=True)
class VidurComponentComparison:
    """One independently costed Blueprinting task and its optional Vidur peer."""

    task_name: str
    source_layer: str
    primitive: str
    estimated_seconds: float
    baseline_seconds: float | None
    baseline_match: str | None

    @property
    def comparable(self) -> bool:
        return self.baseline_seconds is not None

    @property
    def absolute_error_seconds(self) -> float | None:
        if self.baseline_seconds is None:
            return None
        return self.estimated_seconds - self.baseline_seconds

    @property
    def relative_error_percent(self) -> float | None:
        if self.baseline_seconds is None or self.baseline_seconds == 0:
            return None
        return (self.estimated_seconds - self.baseline_seconds) / self.baseline_seconds * 100

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task_name,
            "source_layer": self.source_layer,
            "primitive": self.primitive,
            "estimated_seconds": self.estimated_seconds,
            "baseline_seconds": self.baseline_seconds,
            "baseline_match": self.baseline_match,
            "absolute_error_seconds": self.absolute_error_seconds,
            "relative_error_percent": self.relative_error_percent,
            "status": "comparable" if self.comparable else "not-covered",
        }


@dataclass(frozen=True)
class VidurPhaseComparison:
    """Comparison over the intersection of Blueprinting and Vidur semantics."""

    mode: CalibrationMode
    phase: InferencePhase
    batch_size: int
    query_tokens: int
    context_tokens: int
    plan_digest: str
    hardware_revision: str
    baseline_revision: str
    estimated_block_seconds: float
    components: tuple[VidurComponentComparison, ...]

    @property
    def matched_components(self) -> tuple[VidurComponentComparison, ...]:
        return tuple(component for component in self.components if component.comparable)

    @property
    def estimated_comparable_block_seconds(self) -> float:
        return sum(component.estimated_seconds for component in self.matched_components)

    @property
    def baseline_comparable_block_seconds(self) -> float:
        return sum(component.baseline_seconds or 0.0 for component in self.matched_components)

    @property
    def excluded_estimated_block_seconds(self) -> float:
        return self.estimated_block_seconds - self.estimated_comparable_block_seconds

    @property
    def component_coverage(self) -> float:
        return len(self.matched_components) / len(self.components) if self.components else 0.0

    @property
    def comparable_subtotal_relative_error_percent(self) -> float | None:
        reference = self.baseline_comparable_block_seconds
        if reference == 0:
            return None
        return (self.estimated_comparable_block_seconds - reference) / reference * 100

    @property
    def component_absolute_percentage_errors(self) -> tuple[float, ...]:
        """Absolute component errors before aggregation, so cancellation is impossible."""

        return tuple(
            abs(error)
            for component in self.matched_components
            if (error := component.relative_error_percent) is not None
        )

    @property
    def component_mean_absolute_error_percent(self) -> float | None:
        errors = self.component_absolute_percentage_errors
        return sum(errors) / len(errors) if errors else None

    @property
    def component_max_absolute_error_percent(self) -> float | None:
        errors = self.component_absolute_percentage_errors
        return max(errors) if errors else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "phase": self.phase.value,
            "batch_size": self.batch_size,
            "query_tokens": self.query_tokens,
            "context_tokens": self.context_tokens,
            "plan_digest": self.plan_digest,
            "hardware_revision": self.hardware_revision,
            "baseline_revision": self.baseline_revision,
            "component_coverage": self.component_coverage,
            "matched_component_count": len(self.matched_components),
            "component_count": len(self.components),
            "estimated_block_seconds": self.estimated_block_seconds,
            "estimated_comparable_block_seconds": self.estimated_comparable_block_seconds,
            "baseline_comparable_block_seconds": self.baseline_comparable_block_seconds,
            "excluded_estimated_block_seconds": self.excluded_estimated_block_seconds,
            "comparable_subtotal_relative_error_percent": self.comparable_subtotal_relative_error_percent,
            "component_mean_absolute_error_percent": self.component_mean_absolute_error_percent,
            "component_max_absolute_error_percent": self.component_max_absolute_error_percent,
            "components": [component.to_dict() for component in self.components],
        }


@dataclass(frozen=True)
class VidurExperimentCase:
    """One static phase point synthesized by Blueprinting before comparison."""

    name: str
    model: TransformerModelSpec
    mapping: TransformerInferenceMappingSpec
    network_binding: NetworkTierBinding
    datatype: str
    hardware: SystemProfile
    phase: InferencePhase
    batch_size: int
    context_tokens: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Vidur experiment case name must not be empty")
        if not isinstance(self.phase, InferencePhase):
            raise TypeError("phase must be InferencePhase")
        for field_name in ("batch_size", "context_tokens"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        self.mapping.validate_model(self.model)
        if self.context_tokens > self.model.sequence_length:
            raise ValueError("context_tokens cannot exceed model sequence_length")
        if self.datatype not in {"float8", "float16", "bfloat16", "float32"}:
            raise ValueError(f"unsupported datatype: {self.datatype!r}")
        if self.hardware.datatype != self.datatype:
            raise ValueError("hardware and workload datatype must match")


@dataclass(frozen=True)
class VidurCaseReport:
    case: str
    model_digest: str
    distributed_digest: str
    portable_digest: str
    pass_checkpoints: tuple[dict[str, Any], ...]
    peak_only: VidurPhaseComparison
    system_evidence: VidurPhaseComparison

    def to_dict(self) -> dict[str, Any]:
        return {
            "case": self.case,
            "ir": {
                "model_digest": self.model_digest,
                "distributed_digest": self.distributed_digest,
                "portable_digest": self.portable_digest,
                "pass_checkpoints": list(self.pass_checkpoints),
            },
            "peak_only": self.peak_only.to_dict(),
            "system_evidence": self.system_evidence.to_dict(),
        }


@dataclass(frozen=True)
class VidurExperimentReport:
    schema: str
    baseline_revision: str
    policy: dict[str, Any]
    cases: tuple[VidurCaseReport, ...]

    @staticmethod
    def _comparable_subtotal_mean_absolute_error(
        comparisons: tuple[VidurPhaseComparison, ...],
    ) -> float | None:
        errors = tuple(
            abs(item.comparable_subtotal_relative_error_percent)
            for item in comparisons
            if item.comparable_subtotal_relative_error_percent is not None
        )
        return sum(errors) / len(errors) if errors else None

    @property
    def peak_comparable_subtotal_mean_absolute_error_percent(self) -> float | None:
        return self._comparable_subtotal_mean_absolute_error(tuple(case.peak_only for case in self.cases))

    @property
    def system_evidence_comparable_subtotal_mean_absolute_error_percent(self) -> float | None:
        return self._comparable_subtotal_mean_absolute_error(tuple(case.system_evidence for case in self.cases))

    @staticmethod
    def _component_errors(comparisons: tuple[VidurPhaseComparison, ...]) -> tuple[float, ...]:
        return tuple(error for comparison in comparisons for error in comparison.component_absolute_percentage_errors)

    @classmethod
    def _component_mean_absolute_error(cls, comparisons: tuple[VidurPhaseComparison, ...]) -> float | None:
        errors = cls._component_errors(comparisons)
        return sum(errors) / len(errors) if errors else None

    @classmethod
    def _component_max_absolute_error(cls, comparisons: tuple[VidurPhaseComparison, ...]) -> float | None:
        errors = cls._component_errors(comparisons)
        return max(errors) if errors else None

    @property
    def peak_component_mean_absolute_error_percent(self) -> float | None:
        return self._component_mean_absolute_error(tuple(case.peak_only for case in self.cases))

    @property
    def system_evidence_component_mean_absolute_error_percent(self) -> float | None:
        return self._component_mean_absolute_error(tuple(case.system_evidence for case in self.cases))

    @property
    def system_evidence_component_max_absolute_error_percent(self) -> float | None:
        return self._component_max_absolute_error(tuple(case.system_evidence for case in self.cases))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "baseline_revision": self.baseline_revision,
            "policy": self.policy,
            "summary": {
                "case_count": len(self.cases),
                "peak_comparable_subtotal_mean_absolute_error_percent": (
                    self.peak_comparable_subtotal_mean_absolute_error_percent
                ),
                "system_evidence_comparable_subtotal_mean_absolute_error_percent": (
                    self.system_evidence_comparable_subtotal_mean_absolute_error_percent
                ),
                "peak_component_mean_absolute_error_percent": self.peak_component_mean_absolute_error_percent,
                "system_evidence_component_mean_absolute_error_percent": (
                    self.system_evidence_component_mean_absolute_error_percent
                ),
                "system_evidence_component_max_absolute_error_percent": (
                    self.system_evidence_component_max_absolute_error_percent
                ),
            },
            "cases": [case.to_dict() for case in self.cases],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def compare_inference_phase_to_vidur(
    plan: PortablePlanIR,
    estimate: InferencePhaseEstimate,
    hardware: SystemProfile,
    baseline: InferenceBaseline,
) -> VidurPhaseComparison:
    """Compare an already-lowered and already-costed phase with Vidur."""

    semantic = plan.semantic
    if not isinstance(semantic, TransformerInferencePlanSemantic):
        raise TypeError("portable inference plan is missing typed Transformer semantics")
    model = semantic.model
    mapping = semantic.mapping
    datatype = semantic.datatype
    if len(plan.tasks) != len(estimate.tasks):
        raise ValueError("plan and estimate task counts differ")

    components = []
    for plan_task, task_estimate in zip(plan.tasks, estimate.tasks):
        invocation = task_estimate.invocation
        task_semantic = plan_task.semantic
        if not isinstance(task_semantic, TransformerInferencePlanTaskSemantic):
            raise TypeError("portable inference task is missing typed Transformer semantics")
        if task_semantic.name != invocation.name:
            raise ValueError("plan and estimate task order differs")
        reference = baseline.lookup(
            inference_evidence_query_for(
                invocation,
                hardware=hardware,
                mapping=mapping,
                datatype=datatype,
                model=model,
                batch_size=estimate.batch_size,
                query_tokens=estimate.query_tokens,
                context_tokens=estimate.context_tokens,
            )
        )
        components.append(
            VidurComponentComparison(
                task_name=invocation.name,
                source_layer=invocation.source_layer,
                primitive=invocation.primitive,
                estimated_seconds=task_estimate.total_seconds,
                baseline_seconds=reference.seconds if reference is not None else None,
                baseline_match=reference.match if reference is not None else None,
            )
        )
    return VidurPhaseComparison(
        mode=estimate.mode,
        phase=estimate.phase,
        batch_size=estimate.batch_size,
        query_tokens=estimate.query_tokens,
        context_tokens=estimate.context_tokens,
        plan_digest=plan.digest,
        hardware_revision=hardware.evidence_revision,
        baseline_revision=baseline.revision,
        estimated_block_seconds=estimate.block_seconds,
        components=tuple(components),
    )


def run_vidur_experiment(
    cases: tuple[VidurExperimentCase, ...],
    baseline: InferenceBaseline,
) -> VidurExperimentReport:
    """Synthesize each phase independently, then compare both analytical modes."""

    if not cases:
        raise ValueError("Vidur experiment requires at least one case")
    reports = []
    pipeline = PassPipeline.of(DistributeTransformerInferencePass(), PlanTransformerInferencePass())
    manager = PassManager()
    for case in cases:
        source = build_transformer_inference_model_ir(case.model, datatype=case.datatype)
        result = manager.require_run(
            pipeline,
            source,
            session=inference_synthesis_session_for(
                case.model,
                case.mapping,
                phase=case.phase,
                batch_size=case.batch_size,
                context_tokens=case.context_tokens,
                datatype=case.datatype,
            ),
        )
        plan = result.ir
        if not isinstance(plan, PortablePlanIR):
            raise TypeError(f"inference pipeline returned {type(plan).__name__}, expected PortablePlanIR")
        peak = estimate_inference_phase(
            plan,
            case.hardware,
            CalibrationMode.PEAK_ONLY,
            network_binding=case.network_binding,
        )
        system = estimate_inference_phase(
            plan,
            case.hardware,
            CalibrationMode.SYSTEM_EVIDENCE,
            network_binding=case.network_binding,
        )
        checkpoints = tuple(
            {
                "pass": checkpoint.record.pass_name,
                "schema": checkpoint.ir.header.schema_name,
                "digest": checkpoint.ir.digest,
            }
            for checkpoint in result.checkpoints
        )
        reports.append(
            VidurCaseReport(
                case=case.name,
                model_digest=source.digest,
                distributed_digest=result.checkpoints[0].ir.digest,
                portable_digest=plan.digest,
                pass_checkpoints=checkpoints,
                peak_only=compare_inference_phase_to_vidur(plan, peak, case.hardware, baseline),
                system_evidence=compare_inference_phase_to_vidur(plan, system, case.hardware, baseline),
            )
        )
    return VidurExperimentReport(
        schema="blueprinting.vidur-baseline-experiment.v0",
        baseline_revision=baseline.revision,
        policy={
            "baseline_role": "post-hoc-comparison-only",
            "oracle_read_during_lowering": False,
            "oracle_read_during_costing": False,
            "comparison_domain": "exact component intersection",
            "validation_claim": "drift-detection-not-accuracy-validation",
            "topology_equivalence": "not-claimed-by-raw-component-profile-alignment",
            "fit_against_case_outputs": False,
            "forbidden_inputs": [
                "Vidur component duration during lowering",
                "Vidur phase total during costing",
                "per-case correction factor",
            ],
        },
        cases=tuple(reports),
    )

"""Application service for static decoder inference planning.

This layer composes independently derived prefill and decode phase points
into one homogeneous request-cohort report.  It deliberately excludes request
arrival, queueing, continuous batching and scheduler policy; those belong to
the future serving-simulation layer.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from blueprinting.analysis import (
    CalibrationMode,
    InferenceCostProvider,
    InferencePhaseEstimate,
    estimate_inference_phase,
)
from blueprinting.synthesizer.bindings import InferencePhase
from blueprinting.synthesizer.codec import content_digest
from blueprinting.synthesizer.errors import IRVerificationError, PassExecutionError, SynthesisError
from blueprinting.synthesizer.frontend import (
    build_transformer_inference_model_ir,
    inference_synthesis_session_for,
)
from blueprinting.synthesizer.frozen import FrozenDict, freeze, thaw
from blueprinting.synthesizer.ir import ModelIR, PortablePlanIR
from blueprinting.synthesizer.lowering import DistributeTransformerInferencePass, PlanTransformerInferencePass
from blueprinting.synthesizer.passes import AnalysisStore, PassCheckpoint, PassManager, PassPipeline
from blueprinting.system import SystemProfile
from blueprinting.workload import (
    TransformerInferenceExecutionSpec,
    TransformerInferenceRequestSpec,
    TransformerModelSpec,
)

from .analysis import (
    AnalysisDiagnostic,
    DiagnosticLevel,
    IRStageReport,
    TaskReport,
    _stage_report,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class InferenceAnalysisDraft:
    """Immutable input for one homogeneous inference request cohort."""

    model_name: str
    model_data: FrozenDict
    execution_name: str
    execution_data: FrozenDict
    request_data: FrozenDict
    hardware_name: str
    hardware_data: FrozenDict
    calibration_mode: CalibrationMode = CalibrationMode.SYSTEM_EVIDENCE
    seed: int = 0

    def __post_init__(self) -> None:
        for field_name in ("model_name", "execution_name", "hardware_name"):
            if not getattr(self, field_name):
                raise ValueError(f"{field_name} must not be empty")
        for field_name in ("model_data", "execution_data", "request_data", "hardware_data"):
            value = freeze(getattr(self, field_name))
            if not isinstance(value, FrozenDict):
                raise TypeError(f"{field_name} must be a mapping")
            object.__setattr__(self, field_name, value)
        if not isinstance(self.calibration_mode, CalibrationMode):
            raise TypeError("calibration_mode must be CalibrationMode")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")

    @classmethod
    def from_mappings(
        cls,
        *,
        model_name: str,
        model_data: Mapping[str, Any],
        execution_name: str,
        execution_data: Mapping[str, Any],
        request_data: Mapping[str, Any],
        hardware_name: str,
        hardware_data: Mapping[str, Any],
        calibration_mode: CalibrationMode = CalibrationMode.SYSTEM_EVIDENCE,
        seed: int = 0,
    ) -> InferenceAnalysisDraft:
        return cls(
            model_name=model_name,
            model_data=FrozenDict(model_data),
            execution_name=execution_name,
            execution_data=FrozenDict(execution_data),
            request_data=FrozenDict(request_data),
            hardware_name=hardware_name,
            hardware_data=FrozenDict(hardware_data),
            calibration_mode=calibration_mode,
            seed=seed,
        )

    @property
    def fingerprint(self) -> str:
        return content_digest(
            FrozenDict(
                {
                    "model_name": self.model_name,
                    "model_data": self.model_data,
                    "execution_name": self.execution_name,
                    "execution_data": self.execution_data,
                    "request_data": self.request_data,
                    "hardware_name": self.hardware_name,
                    "hardware_data": self.hardware_data,
                    "calibration_mode": self.calibration_mode.value,
                    "seed": self.seed,
                }
            ),
            "blueprinting-inference-analysis-request-v1",
        )

    def normalized_execution(self) -> dict[str, Any]:
        data = thaw(self.execution_data)
        replicas = data.get("replicas", data.get("data_par", 1))
        data["replicas"] = replicas
        data["num_procs"] = data["tensor_par"] * data["pipeline_par"] * replicas
        return data


@dataclass(frozen=True)
class DecodeStepReport:
    context_tokens: int
    plan_digest: str
    session_fingerprint: str
    seconds: float
    memory_bytes: int
    evidence_revisions: FrozenDict


@dataclass(frozen=True)
class InferenceAnalysisReport:
    """Static model-phase composition with inspectable portable plans."""

    schema: str
    request_digest: str
    model_name: str
    execution_name: str
    hardware_name: str
    calibration_mode: str
    world_size: int
    analytical_memory_fits: bool
    prefill_seconds: float
    mean_decode_step_seconds: float
    first_decode_step_seconds: float
    last_decode_step_seconds: float
    model_execution_seconds: float
    static_model_tokens_per_second: float | None
    latency: FrozenDict
    memory: FrozenDict
    workload: FrozenDict
    evidence: FrozenDict
    configuration: FrozenDict
    stages: tuple[IRStageReport, ...]
    tasks: tuple[TaskReport, ...]
    decode_steps: tuple[DecodeStepReport, ...]
    limitations: tuple[str, ...]


@dataclass(frozen=True)
class InferenceAnalysisOutcome:
    request_digest: str
    diagnostics: tuple[AnalysisDiagnostic, ...]
    report: InferenceAnalysisReport | None = None

    @property
    def ok(self) -> bool:
        return self.report is not None and not any(item.level is DiagnosticLevel.ERROR for item in self.diagnostics)


@dataclass(frozen=True)
class _DerivedPhase:
    session_fingerprint: str
    plan: PortablePlanIR
    estimate: InferencePhaseEstimate
    checkpoints: tuple[PassCheckpoint, ...]


def _task_reports(plan: PortablePlanIR, estimate: InferencePhaseEstimate) -> tuple[TaskReport, ...]:
    return tuple(
        TaskReport(
            task_id=str(task.id),
            operation=str(task.operation),
            kind=task.kind.value,
            phase=str(task.workload.attributes.get("phase", "unknown")),
            engine=str(task.workload.attributes.get("engine", "unknown")),
            source_layer=str(task.workload.attributes.get("source_layer", "")),
            dependencies=tuple(str(item) for item in task.dependencies),
            concurrency_group=task.concurrency_group or "",
            operations=task.workload.operations,
            read_bytes=task.workload.read_bytes,
            write_bytes=task.workload.write_bytes,
            message_bytes=task.workload.message_bytes,
            compute_seconds=task_estimate.compute_seconds,
            memory_seconds=task_estimate.memory_seconds,
            network_seconds=task_estimate.network_seconds,
            total_seconds=task_estimate.total_seconds,
            analytical_seconds=task_estimate.analytical_seconds,
            evidence_provider=task_estimate.evidence_provider,
            evidence_revision=task_estimate.evidence_revision,
            evidence_match=task_estimate.evidence_match,
        )
        for task, task_estimate in zip(plan.tasks, estimate.tasks)
    )


class InferenceAnalysisService:
    """Derive and compose static inference phase points."""

    def __init__(
        self,
        analyses: AnalysisStore | None = None,
        *,
        cost_provider: InferenceCostProvider | None = None,
    ) -> None:
        self._manager = PassManager(analyses=analyses)
        self._pipeline = PassPipeline.of(
            DistributeTransformerInferencePass(),
            PlanTransformerInferencePass(),
        )
        self._cost_provider = cost_provider

    def analyze(self, draft: InferenceAnalysisDraft) -> InferenceAnalysisOutcome:
        try:
            return self._analyze(draft)
        except IRVerificationError as error:
            diagnostics = tuple(
                AnalysisDiagnostic(
                    code=item.code,
                    message=item.message,
                    path=item.path,
                    hint=item.hint,
                )
                for item in error.diagnostics
            )
            return InferenceAnalysisOutcome(draft.fingerprint, diagnostics)
        except PassExecutionError as error:
            return InferenceAnalysisOutcome(
                draft.fingerprint,
                (
                    AnalysisDiagnostic(
                        code="inference.lowering.pass_failed",
                        message=str(error.cause),
                        path=(error.pass_name,),
                        hint="检查 phase workload、并行整除关系与网络层级。",
                    ),
                ),
            )
        except KeyError as error:
            field = str(error.args[0]) if error.args else "<unknown>"
            return InferenceAnalysisOutcome(
                draft.fingerprint,
                (
                    AnalysisDiagnostic(
                        code="inference.configuration.missing_field",
                        message=f"推理配置缺少字段 {field!r}",
                        path=(field,),
                    ),
                ),
            )
        except (ValueError, TypeError, IndexError, ZeroDivisionError) as error:
            return InferenceAnalysisOutcome(
                draft.fingerprint,
                (
                    AnalysisDiagnostic(
                        code="inference.configuration.invalid",
                        message=str(error),
                        hint="检查请求长度、模型上下文、TP/PP 整除关系和硬件 profile。",
                    ),
                ),
            )
        except SynthesisError as error:
            return InferenceAnalysisOutcome(
                draft.fingerprint,
                (
                    AnalysisDiagnostic(
                        code="inference.analysis.synthesis_failure",
                        message=str(error),
                    ),
                ),
            )
        except Exception as error:  # pragma: no cover - defensive application boundary
            LOGGER.exception("unexpected Blueprinting inference analysis failure")
            return InferenceAnalysisOutcome(
                draft.fingerprint,
                (
                    AnalysisDiagnostic(
                        code="inference.analysis.internal_error",
                        message=f"{type(error).__name__}: {error}",
                    ),
                ),
            )

    def _derive_phase(
        self,
        source: ModelIR,
        model: TransformerModelSpec,
        execution: TransformerInferenceExecutionSpec,
        hardware: SystemProfile,
        draft: InferenceAnalysisDraft,
        *,
        phase: InferencePhase,
        batch_size: int,
        context_tokens: int,
    ) -> _DerivedPhase:
        session = replace(
            inference_synthesis_session_for(
                model,
                execution,
                phase=phase,
                batch_size=batch_size,
                context_tokens=context_tokens,
            ),
            seed=draft.seed,
        )
        pipeline = self._manager.run(self._pipeline, source, session=session)
        plan = pipeline.ir
        if not isinstance(plan, PortablePlanIR):
            raise TypeError(f"inference pipeline returned {type(plan).__name__}, expected PortablePlanIR")
        estimate = estimate_inference_phase(
            plan,
            hardware,
            draft.calibration_mode,
            cost_provider=self._cost_provider,
        )
        return _DerivedPhase(session.fingerprint, plan, estimate, pipeline.checkpoints)

    def _analyze(self, draft: InferenceAnalysisDraft) -> InferenceAnalysisOutcome:
        model_data = thaw(draft.model_data)
        execution_data = draft.normalized_execution()
        request_data = thaw(draft.request_data)
        hardware_data = thaw(draft.hardware_data)
        model = TransformerModelSpec.from_mapping(draft.model_name, model_data)
        execution = TransformerInferenceExecutionSpec.from_mapping(execution_data)
        request = TransformerInferenceRequestSpec.from_mapping(request_data)
        execution.validate_model(model)
        request.validate_model(model)
        hardware = SystemProfile.from_mapping(
            draft.hardware_name,
            hardware_data,
            datatype=execution.datatype,
        )

        frontend_started = time.perf_counter_ns()
        source = build_transformer_inference_model_ir(model, datatype=execution.datatype)
        frontend_duration = time.perf_counter_ns() - frontend_started
        prefill = self._derive_phase(
            source,
            model,
            execution,
            hardware,
            draft,
            phase=InferencePhase.PREFILL,
            batch_size=request.batch_size,
            context_tokens=request.prompt_tokens,
        )
        decode = tuple(
            self._derive_phase(
                source,
                model,
                execution,
                hardware,
                draft,
                phase=InferencePhase.DECODE,
                batch_size=request.batch_size,
                context_tokens=context_tokens,
            )
            for context_tokens in request.decode_contexts()
        )

        prefill_seconds = prefill.estimate.total_seconds
        decode_total = sum(item.estimate.total_seconds for item in decode)
        model_execution_seconds = prefill_seconds + decode_total
        mean_decode_step = decode_total / request.decode_iterations if request.decode_iterations else 0.0
        first_decode_step = decode[0].estimate.total_seconds if decode else 0.0
        last_decode_step = decode[-1].estimate.total_seconds if decode else 0.0
        phase_estimates = (prefill.estimate,) + tuple(item.estimate for item in decode)
        peak = max(phase_estimates, key=lambda item: item.memory.total)
        analytical_memory_fits = peak.memory.total <= hardware.memory.capacity_bytes

        stages = [
            _stage_report("model", "推理模型语义", "frontend-import", source, frontend_duration),
            _stage_report(
                "prefill.distributed",
                "Prefill 分布式任务",
                prefill.checkpoints[0].record.pass_name,
                prefill.checkpoints[0].ir,
                prefill.checkpoints[0].record.duration_ns,
            ),
            _stage_report(
                "prefill.portable",
                "Prefill 可移植计划",
                prefill.checkpoints[1].record.pass_name,
                prefill.checkpoints[1].ir,
                prefill.checkpoints[1].record.duration_ns,
            ),
        ]
        representative = decode[-1] if decode else None
        if representative is not None:
            stages.extend(
                (
                    _stage_report(
                        "decode.distributed",
                        "Decode 分布式任务（最终 context）",
                        representative.checkpoints[0].record.pass_name,
                        representative.checkpoints[0].ir,
                        representative.checkpoints[0].record.duration_ns,
                    ),
                    _stage_report(
                        "decode.portable",
                        "Decode 可移植计划（最终 context）",
                        representative.checkpoints[1].record.pass_name,
                        representative.checkpoints[1].ir,
                        representative.checkpoints[1].record.duration_ns,
                    ),
                )
            )

        tasks = _task_reports(prefill.plan, prefill.estimate)
        if representative is not None:
            tasks += _task_reports(representative.plan, representative.estimate)
        decode_steps = tuple(
            DecodeStepReport(
                context_tokens=item.estimate.context_tokens,
                plan_digest=item.plan.digest,
                session_fingerprint=item.session_fingerprint,
                seconds=item.estimate.total_seconds,
                memory_bytes=item.estimate.memory.total,
                evidence_revisions=item.estimate.evidence_revisions,
            )
            for item in decode
        )
        diagnostics: tuple[AnalysisDiagnostic, ...] = ()
        if not analytical_memory_fits:
            diagnostics = (
                AnalysisDiagnostic(
                    code="capacity.device_memory_exceeded",
                    message=(
                        f"每设备推理峰值预计需要 {peak.memory.total} bytes，"
                        f"超过容量 {hardware.memory.capacity_bytes} bytes"
                    ),
                    level=DiagnosticLevel.WARNING,
                    path=("memory", "total"),
                    hint="增加 TP/PP、减小 batch/context，或选择更大容量硬件。",
                ),
            )

        latency = FrozenDict(
            {
                "prefill": prefill_seconds,
                "decode_total": decode_total,
                "mean_decode_step": mean_decode_step,
                "first_decode_step": first_decode_step,
                "last_decode_step": last_decode_step,
                "model_execution": model_execution_seconds,
            }
        )
        memory = FrozenDict(
            {
                "weights": peak.memory.weights,
                "kv_cache": peak.memory.kv_cache,
                "max_kv_cache": max(item.memory.kv_cache for item in phase_estimates),
                "working_upper_bound": peak.memory.working_upper_bound,
                "max_working_upper_bound": max(item.memory.working_upper_bound for item in phase_estimates),
                "pipeline_buffers": peak.memory.pipeline_buffers,
                "total": peak.memory.total,
                "capacity": hardware.memory.capacity_bytes,
                "peak_phase": peak.phase.value,
                "peak_context_tokens": peak.context_tokens,
            }
        )
        workload = FrozenDict(
            {
                "prefill_task_count": len(prefill.plan.tasks),
                "decode_task_count": len(representative.plan.tasks) if representative else 0,
                "decode_iterations": request.decode_iterations,
                "prefill_operations_per_block": sum(task.workload.operations for task in prefill.plan.tasks),
                "final_decode_operations_per_block": (
                    sum(task.workload.operations for task in representative.plan.tasks) if representative else 0
                ),
            }
        )
        evidence_revisions: dict[str, str] = {}
        for estimate in phase_estimates:
            evidence_revisions.update(dict(estimate.evidence_revisions.items()))
        evidence = FrozenDict(
            {
                "hardware_name": hardware.name,
                "hardware_revision": hardware.evidence_revision,
                "mode": draft.calibration_mode.value,
                "cost_provider_revision": self._cost_provider.revision if self._cost_provider is not None else "none",
                "revisions": FrozenDict(evidence_revisions),
            }
        )
        configuration = FrozenDict(
            {
                "model": model_data,
                "execution": execution_data,
                "request": request_data,
                "hardware": {
                    "name": hardware.name,
                    "datatype": hardware.datatype,
                    "processing_mode": hardware.processing_mode,
                },
            }
        )
        report = InferenceAnalysisReport(
            schema="blueprinting.inference-analysis-report.v1",
            request_digest=draft.fingerprint,
            model_name=draft.model_name,
            execution_name=draft.execution_name,
            hardware_name=draft.hardware_name,
            calibration_mode=draft.calibration_mode.value,
            world_size=execution.world_size,
            analytical_memory_fits=analytical_memory_fits,
            prefill_seconds=prefill_seconds,
            mean_decode_step_seconds=mean_decode_step,
            first_decode_step_seconds=first_decode_step,
            last_decode_step_seconds=last_decode_step,
            model_execution_seconds=model_execution_seconds,
            static_model_tokens_per_second=(
                request.batch_size * request.generated_tokens / model_execution_seconds
                if model_execution_seconds > 0
                else None
            ),
            latency=latency,
            memory=memory,
            workload=workload,
            evidence=evidence,
            configuration=configuration,
            stages=tuple(stages),
            tasks=tasks,
            decode_steps=decode_steps,
            limitations=(
                "这是 homogeneous cohort 的静态 phase composition；不包含请求到达、排队、continuous batching 或 scheduler 开销。",
                "prefill/decode 时间只覆盖当前 decoder-block dialect；由于 embedding、LM head 与 sampler 未建模，不能解释为 TTFT、TPOT 或 E2E serving latency。",
                "replicas 只参与映射合法性与 world-size 记账；当前报告是单 replica cohort latency，不估算跨 replica serving capacity。",
                "当前 workload dialect 支持 dense multi-head attention 与非 gated MLP；embedding、LM head 和 sampler 尚未建模。",
                "PortablePlanIR 尚未绑定 attention implementation；working memory 使用未融合 score materialization 的保守上界。",
                "除非提供 Blueprinting cost provider，组件耗时使用共享 system profile 的解析 roofline 证据；comparison baseline 不参与该选择。",
            ),
        )
        return InferenceAnalysisOutcome(draft.fingerprint, diagnostics, report)

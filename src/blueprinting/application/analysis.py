"""Stable application boundary for interactive Blueprinting analysis.

The Streamlit workbench, CLI clients, and future agents should call this
module instead of assembling lowering passes themselves.  Expected input and
feasibility failures are returned as structured diagnostics; canonical IR and
cost-model implementation details stay behind the service boundary.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from enum import Enum
from itertools import product
from typing import TYPE_CHECKING, Any

from blueprinting.analysis import CalibrationMode, HardwareProfile, estimate_iteration
from blueprinting.synthesizer.codec import content_digest
from blueprinting.synthesizer.errors import (
    IRVerificationError,
    PassExecutionError,
    SynthesisError,
)
from blueprinting.synthesizer.frozen import FrozenDict, freeze, thaw
from blueprinting.synthesizer.ir import DistributedTaskIR, ModelIR, PortablePlanIR
from blueprinting.synthesizer.lowering import DistributeTransformerTrainingPass, PlanTransformerTrainingPass
from blueprinting.synthesizer.models import (
    TransformerExecutionSpec,
    TransformerModelSpec,
    build_transformer_model_ir,
    synthesis_session_for,
)
from blueprinting.synthesizer.passes import AnalysisStore, PassManager, PassPipeline

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from blueprinting.analysis import InferenceCostProvider

    from .inference import InferenceAnalysisDraft, InferenceAnalysisOutcome, InferenceAnalysisService


class DiagnosticLevel(Enum):
    """Presentation-neutral diagnostic severity."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class AnalysisDiagnostic:
    """A stable diagnostic that can be rendered by any client."""

    code: str
    message: str
    level: DiagnosticLevel = DiagnosticLevel.ERROR
    path: tuple[str, ...] = ()
    hint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "level": self.level.value,
            "path": list(self.path),
            "hint": self.hint,
        }


@dataclass(frozen=True)
class AnalysisDraft:
    """Immutable, client-supplied configuration before semantic validation."""

    model_name: str
    model_data: FrozenDict
    execution_name: str
    execution_data: FrozenDict
    hardware_name: str
    hardware_data: FrozenDict
    calibration_mode: CalibrationMode = CalibrationMode.SYSTEM_EVIDENCE
    seed: int = 0

    def __post_init__(self) -> None:
        for field_name in ("model_name", "execution_name", "hardware_name"):
            if not getattr(self, field_name):
                raise ValueError(f"{field_name} must not be empty")
        for field_name in ("model_data", "execution_data", "hardware_data"):
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
        hardware_name: str,
        hardware_data: Mapping[str, Any],
        calibration_mode: CalibrationMode = CalibrationMode.SYSTEM_EVIDENCE,
        seed: int = 0,
    ) -> AnalysisDraft:
        return cls(
            model_name=model_name,
            model_data=FrozenDict(model_data),
            execution_name=execution_name,
            execution_data=FrozenDict(execution_data),
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
                    "hardware_name": self.hardware_name,
                    "hardware_data": self.hardware_data,
                    "calibration_mode": self.calibration_mode.value,
                    "seed": self.seed,
                }
            ),
            "blueprinting-analysis-request-v1",
        )

    def normalized_execution(self) -> dict[str, Any]:
        """Make world size a derived fact instead of a second source of truth."""

        data = thaw(self.execution_data)
        data["num_procs"] = data["tensor_par"] * data["pipeline_par"] * data["data_par"]
        return data

    def with_parallelism(self, tensor_parallel: int, pipeline_parallel: int, data_parallel: int) -> AnalysisDraft:
        execution = thaw(self.execution_data)
        execution.update(
            {
                "tensor_par": tensor_parallel,
                "pipeline_par": pipeline_parallel,
                "data_par": data_parallel,
                "num_procs": tensor_parallel * pipeline_parallel * data_parallel,
            }
        )
        return replace(self, execution_data=FrozenDict(execution))


@dataclass(frozen=True)
class IRStageReport:
    """One inspectable boundary in the formal derivation."""

    stage: str
    label: str
    pass_name: str
    schema: str
    digest: str
    parent_digests: tuple[str, ...]
    node_count: int
    value_count: int
    duration_ns: int
    diagnostics: tuple[AnalysisDiagnostic, ...]
    snapshot_json: str

    @property
    def valid(self) -> bool:
        return not any(item.level is DiagnosticLevel.ERROR for item in self.diagnostics)


@dataclass(frozen=True)
class TaskReport:
    """UI-safe work and timing facts for one portable-plan task."""

    task_id: str
    operation: str
    kind: str
    phase: str
    engine: str
    source_layer: str
    dependencies: tuple[str, ...]
    concurrency_group: str
    operations: int
    read_bytes: int
    write_bytes: int
    message_bytes: int
    compute_seconds: float
    memory_seconds: float
    network_seconds: float
    total_seconds: float
    analytical_seconds: float = 0.0
    evidence_provider: str = ""
    evidence_revision: str = ""
    evidence_match: str = ""


@dataclass(frozen=True)
class AnalysisReport:
    """Successful, immutable result returned to an interactive client."""

    schema: str
    request_digest: str
    session_fingerprint: str
    plan_digest: str
    evidence_revision: str
    model_name: str
    execution_name: str
    hardware_name: str
    calibration_mode: str
    world_size: int
    feasible: bool
    total_seconds: float
    total_tokens_per_second: float
    tokens_per_second_per_device: float
    bottleneck: str
    latency: FrozenDict
    memory: FrozenDict
    workload: FrozenDict
    evidence: FrozenDict
    configuration: FrozenDict
    stages: tuple[IRStageReport, ...]
    tasks: tuple[TaskReport, ...]
    limitations: tuple[str, ...]


@dataclass(frozen=True)
class AnalysisOutcome:
    """Discriminated result: report on success, diagnostics on failure."""

    request_digest: str
    diagnostics: tuple[AnalysisDiagnostic, ...]
    report: AnalysisReport | None = None

    @property
    def ok(self) -> bool:
        return self.report is not None and not any(item.level is DiagnosticLevel.ERROR for item in self.diagnostics)


@dataclass(frozen=True)
class SweepRequest:
    """Bounded strategy-space exploration request."""

    base: AnalysisDraft
    tensor_parallel: tuple[int, ...]
    pipeline_parallel: tuple[int, ...]
    data_parallel: tuple[int, ...]
    max_candidates: int = 128

    def __post_init__(self) -> None:
        for field_name in ("tensor_parallel", "pipeline_parallel", "data_parallel"):
            values = tuple(dict.fromkeys(getattr(self, field_name)))
            if not values or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values
            ):
                raise ValueError(f"{field_name} must contain positive integers")
            object.__setattr__(self, field_name, values)
        if self.max_candidates <= 0:
            raise ValueError("max_candidates must be positive")
        if self.candidate_count > self.max_candidates:
            raise ValueError(f"strategy sweep has {self.candidate_count} candidates; limit is {self.max_candidates}")

    @property
    def candidate_count(self) -> int:
        return len(self.tensor_parallel) * len(self.pipeline_parallel) * len(self.data_parallel)

    @property
    def fingerprint(self) -> str:
        return content_digest(
            FrozenDict(
                {
                    "base": self.base.fingerprint,
                    "tensor_parallel": self.tensor_parallel,
                    "pipeline_parallel": self.pipeline_parallel,
                    "data_parallel": self.data_parallel,
                }
            ),
            "blueprinting-sweep-request-v1",
        )


@dataclass(frozen=True)
class SweepCase:
    tensor_parallel: int
    pipeline_parallel: int
    data_parallel: int
    world_size: int
    status: str
    feasible: bool
    pareto: bool
    total_seconds: float | None
    memory_bytes: int | None
    tokens_per_second_per_device: float | None
    bottleneck: str | None
    request_digest: str
    diagnostics: tuple[AnalysisDiagnostic, ...]


@dataclass(frozen=True)
class SweepReport:
    schema: str
    request_digest: str
    cases: tuple[SweepCase, ...]

    @property
    def succeeded_count(self) -> int:
        return sum(case.status == "success" for case in self.cases)

    @property
    def feasible_count(self) -> int:
        return sum(case.feasible for case in self.cases)


def _diagnostics_from_verification(ir: ModelIR | DistributedTaskIR | PortablePlanIR) -> tuple[AnalysisDiagnostic, ...]:
    return tuple(
        AnalysisDiagnostic(
            code=item.code,
            message=item.message,
            level=DiagnosticLevel(item.severity.value),
            path=item.path,
            hint=item.hint,
        )
        for item in ir.verify().diagnostics
    )


def _stage_report(
    stage: str,
    label: str,
    pass_name: str,
    ir: ModelIR | DistributedTaskIR | PortablePlanIR,
    duration_ns: int,
) -> IRStageReport:
    if isinstance(ir, ModelIR):
        node_count, value_count = len(ir.operations), len(ir.values)
    elif isinstance(ir, DistributedTaskIR):
        node_count, value_count = len(ir.tasks), len(ir.values)
    else:
        node_count, value_count = len(ir.tasks), len(ir.buffers)
    return IRStageReport(
        stage=stage,
        label=label,
        pass_name=pass_name,
        schema=f"{ir.header.schema_name}@{ir.header.schema_version}",
        digest=ir.digest,
        parent_digests=ir.header.parent_digests,
        node_count=node_count,
        value_count=value_count,
        duration_ns=duration_ns,
        diagnostics=_diagnostics_from_verification(ir),
        snapshot_json=ir.to_json(),
    )


class BlueprintingService:
    """Single supported orchestration entry point for Blueprinting clients."""

    def __init__(
        self,
        analyses: AnalysisStore | None = None,
        *,
        inference_cost_provider: InferenceCostProvider | None = None,
    ) -> None:
        from .inference import InferenceAnalysisService

        self._manager = PassManager(analyses=analyses)
        self._pipeline = PassPipeline.of(
            DistributeTransformerTrainingPass(),
            PlanTransformerTrainingPass(),
        )
        self._inference: InferenceAnalysisService = InferenceAnalysisService(
            analyses,
            cost_provider=inference_cost_provider,
        )

    def analyze_inference(self, draft: InferenceAnalysisDraft) -> InferenceAnalysisOutcome:
        """Run the canonical static inference path through the same service boundary."""

        return self._inference.analyze(draft)

    def analyze(self, draft: AnalysisDraft) -> AnalysisOutcome:
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
            return AnalysisOutcome(draft.fingerprint, diagnostics)
        except PassExecutionError as error:
            diagnostic = AnalysisDiagnostic(
                code="lowering.pass_failed",
                message=str(error.cause),
                path=(error.pass_name,),
                hint="检查该 lowering 阶段所需的模型、策略和网络约束。",
            )
            return AnalysisOutcome(draft.fingerprint, (diagnostic,))
        except KeyError as error:
            field = str(error.args[0]) if error.args else "<unknown>"
            diagnostic = AnalysisDiagnostic(
                code="configuration.missing_field",
                message=f"配置缺少字段 {field!r}",
                path=(field,),
                hint="选择完整的预设，或补齐该字段后重试。",
            )
            return AnalysisOutcome(draft.fingerprint, (diagnostic,))
        except (ValueError, TypeError, IndexError, ZeroDivisionError) as error:
            diagnostic = AnalysisDiagnostic(
                code="configuration.invalid",
                message=str(error),
                hint="检查模型维度、批量整除关系、并行度和网络层级。",
            )
            return AnalysisOutcome(draft.fingerprint, (diagnostic,))
        except SynthesisError as error:
            diagnostic = AnalysisDiagnostic(
                code="analysis.synthesis_failure",
                message=str(error),
                hint="查看 IR 推导页中的阶段信息和 digest。",
            )
            return AnalysisOutcome(draft.fingerprint, (diagnostic,))
        except Exception as error:  # pragma: no cover - defensive application boundary
            LOGGER.exception("unexpected Blueprinting analysis failure")
            diagnostic = AnalysisDiagnostic(
                code="analysis.internal_error",
                message=f"{type(error).__name__}: {error}",
                hint="这是未分类的实现错误，请携带 request digest 报告。",
            )
            return AnalysisOutcome(draft.fingerprint, (diagnostic,))

    def _analyze(self, draft: AnalysisDraft) -> AnalysisOutcome:
        model_data = thaw(draft.model_data)
        execution_data = draft.normalized_execution()
        hardware_data = thaw(draft.hardware_data)

        model = TransformerModelSpec.from_mapping(draft.model_name, model_data)
        execution = TransformerExecutionSpec.from_mapping(execution_data)
        hardware = HardwareProfile.from_mapping(
            draft.hardware_name,
            hardware_data,
            datatype=execution.datatype,
        )

        frontend_started = time.perf_counter_ns()
        source = build_transformer_model_ir(model, datatype=execution.datatype)
        frontend_duration = time.perf_counter_ns() - frontend_started
        session = replace(synthesis_session_for(model, execution), seed=draft.seed)
        pipeline = self._manager.run(self._pipeline, source, session=session)
        plan = pipeline.ir
        if not isinstance(plan, PortablePlanIR):
            raise TypeError(f"analysis pipeline returned {type(plan).__name__}, expected PortablePlanIR")
        estimate = estimate_iteration(plan, hardware, draft.calibration_mode)

        stages = [_stage_report("model", "模型语义", "frontend-import", source, frontend_duration)]
        stage_metadata = (
            ("distributed", "分布式任务", pipeline.checkpoints[0]),
            ("portable", "可移植计划", pipeline.checkpoints[1]),
        )
        for stage, label, checkpoint in stage_metadata:
            stages.append(
                _stage_report(
                    stage,
                    label,
                    checkpoint.record.pass_name,
                    checkpoint.ir,
                    checkpoint.record.duration_ns,
                )
            )

        latency = FrozenDict(
            {
                "forward": estimate.forward,
                "backward": estimate.backward,
                "optimizer": estimate.optimizer,
                "recompute": estimate.recompute,
                "tensor_parallel": estimate.tensor_parallel,
                "pipeline_parallel": estimate.pipeline_parallel,
                "data_parallel": estimate.data_parallel,
                "recommunication": estimate.recommunication,
                "pipeline_bubble": estimate.pipeline_bubble,
            }
        )
        memory = FrozenDict(
            {
                "weights": estimate.memory.weights,
                "activations": estimate.memory.activations,
                "activation_checkpoints": estimate.memory.activation_checkpoints,
                "weight_gradients": estimate.memory.weight_gradients,
                "activation_gradients": estimate.memory.activation_gradients,
                "optimizer": estimate.memory.optimizer,
                "total": estimate.memory.total,
                "capacity": hardware.memory.capacity_bytes,
            }
        )
        tasks = tuple(
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
            )
            for task, task_estimate in zip(plan.tasks, estimate.block.tasks)
        )
        workload = FrozenDict(
            {
                "task_count": len(tasks),
                "compute_task_count": sum(task.kind == "compute" for task in tasks),
                "collective_task_count": sum(task.kind == "collective" for task in tasks),
                "operations": sum(task.operations for task in tasks),
                "read_bytes": sum(task.read_bytes for task in tasks),
                "write_bytes": sum(task.write_bytes for task in tasks),
                "message_bytes": sum(task.message_bytes for task in tasks),
            }
        )
        bottleneck = max(latency.items(), key=lambda item: item[1])[0]
        total_tokens = model.sequence_length * execution.global_batch_size
        feasible = estimate.memory.total <= hardware.memory.capacity_bytes
        diagnostics: tuple[AnalysisDiagnostic, ...] = ()
        if not feasible:
            diagnostics = (
                AnalysisDiagnostic(
                    code="capacity.device_memory_exceeded",
                    message=(
                        f"每设备预计需要 {estimate.memory.total} bytes，超过容量 {hardware.memory.capacity_bytes} bytes"
                    ),
                    level=DiagnosticLevel.WARNING,
                    path=("memory", "total"),
                    hint="增加并行度、启用重计算或选择更大容量的硬件。",
                ),
            )

        configuration = FrozenDict(
            {
                "model": model_data,
                "execution": execution_data,
                "hardware": {
                    "name": hardware.name,
                    "datatype": hardware.datatype,
                    "processing_mode": hardware.processing_mode,
                },
            }
        )
        evidence = FrozenDict(
            {
                "hardware_name": hardware.name,
                "revision": hardware.evidence_revision,
                "mode": draft.calibration_mode.value,
                "matrix_peak_ops_per_second": hardware.matrix.peak_operations_per_second,
                "vector_peak_ops_per_second": hardware.vector.peak_operations_per_second,
                "memory_peak_bytes_per_second": hardware.memory.peak_bytes_per_second,
                "memory_capacity_bytes": hardware.memory.capacity_bytes,
                "network_tiers": len(hardware.networks),
            }
        )
        report = AnalysisReport(
            schema="blueprinting.analysis-report.v1",
            request_digest=draft.fingerprint,
            session_fingerprint=session.fingerprint,
            plan_digest=plan.digest,
            evidence_revision=hardware.evidence_revision,
            model_name=draft.model_name,
            execution_name=draft.execution_name,
            hardware_name=draft.hardware_name,
            calibration_mode=draft.calibration_mode.value,
            world_size=execution.world_size,
            feasible=feasible,
            total_seconds=estimate.total,
            total_tokens_per_second=total_tokens / estimate.total,
            tokens_per_second_per_device=total_tokens / estimate.total / execution.world_size,
            bottleneck=bottleneck,
            latency=latency,
            memory=memory,
            workload=workload,
            evidence=evidence,
            configuration=configuration,
            stages=tuple(stages),
            tasks=tasks,
            limitations=(
                "当前结果止于 PortablePlanIR，尚未进行具体硬件 placement、queue scheduling 和 programme emission。",
                "当前训练调度采用解析式 1F1B/interleaved 组合，不是事件级 Timeline 仿真。",
                "TP/DP overlap 与 offload 策略尚未进入这条生产分析路径。",
            ),
        )
        return AnalysisOutcome(draft.fingerprint, diagnostics, report)

    def sweep(
        self,
        request: SweepRequest,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> SweepReport:
        cases = []
        candidates = tuple(product(request.tensor_parallel, request.pipeline_parallel, request.data_parallel))
        for index, (tp, pp, dp) in enumerate(candidates, start=1):
            outcome = self.analyze(request.base.with_parallelism(tp, pp, dp))
            if outcome.report is None:
                case = SweepCase(
                    tensor_parallel=tp,
                    pipeline_parallel=pp,
                    data_parallel=dp,
                    world_size=tp * pp * dp,
                    status="invalid",
                    feasible=False,
                    pareto=False,
                    total_seconds=None,
                    memory_bytes=None,
                    tokens_per_second_per_device=None,
                    bottleneck=None,
                    request_digest=outcome.request_digest,
                    diagnostics=outcome.diagnostics,
                )
            else:
                report = outcome.report
                case = SweepCase(
                    tensor_parallel=tp,
                    pipeline_parallel=pp,
                    data_parallel=dp,
                    world_size=report.world_size,
                    status="success",
                    feasible=report.feasible,
                    pareto=False,
                    total_seconds=report.total_seconds,
                    memory_bytes=report.memory["total"],
                    tokens_per_second_per_device=report.tokens_per_second_per_device,
                    bottleneck=report.bottleneck,
                    request_digest=report.request_digest,
                    diagnostics=outcome.diagnostics,
                )
            cases.append(case)
            if on_progress is not None:
                on_progress(index, len(candidates))

        feasible = tuple(case for case in cases if case.status == "success" and case.feasible)
        pareto_digests = {
            case.request_digest
            for case in feasible
            if not any(
                other.request_digest != case.request_digest
                and other.total_seconds <= case.total_seconds
                and other.memory_bytes <= case.memory_bytes
                and (other.total_seconds < case.total_seconds or other.memory_bytes < case.memory_bytes)
                for other in feasible
            )
        }
        cases = [replace(case, pareto=case.request_digest in pareto_digests) for case in cases]
        return SweepReport(
            schema="blueprinting.strategy-sweep.v1",
            request_digest=request.fingerprint,
            cases=tuple(cases),
        )

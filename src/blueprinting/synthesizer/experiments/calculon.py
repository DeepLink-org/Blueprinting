"""Calculon comparison without reference-dependent alignment knobs.

The experiment uses Calculon in two roles only:

* an oracle for independently checking derived work and schedule metrics;
* a source of historical SeqSel paper values for held-out validation.

Calculon results are never read while constructing IR, workload facts, or the
hardware profile.  The calibrated estimate consumes only the same target-wide
system evidence curves for every case.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from calculon.llm import Llm
from calculon.system import System

from ...analysis.cost_model import (
    CalibrationMode,
    HardwareProfile,
    IterationEstimate,
    estimate_iteration,
)
from ...analysis.transformer_workload import EngineKind, PrimitiveInvocation, TrainingPhase
from ..ir import PortablePlanIR
from ..lowering import DistributeTransformerTrainingPass, PlanTransformerTrainingPass
from ..models import (
    TransformerExecutionSpec,
    TransformerModelSpec,
    build_transformer_model_ir,
    synthesis_session_for,
)
from ..passes import PassManager, PassPipeline

SEQSEL_TABLE5_SECONDS = {
    "megatron-22B": {"full": 1.42, "seqsel": 1.10},
    "gpt3-175B": {"full": 18.13, "seqsel": 13.75},
    "turing-530B": {"full": 49.05, "seqsel": 37.83},
    "megatron-1T": {"full": 94.42, "seqsel": 71.49},
}


@dataclass(frozen=True)
class CalculonCase:
    name: str
    model_path: Path
    execution_path: Path
    system_path: Path
    paper_seconds: float | None = None


@dataclass(frozen=True)
class MetricComparison:
    blueprinting: float
    reference: float

    @property
    def relative_error_percent(self) -> float:
        if self.reference == 0:
            return 0.0 if self.blueprinting == 0 else float("inf")
        return (self.blueprinting - self.reference) / self.reference * 100

    def to_dict(self) -> dict[str, float]:
        return {
            "blueprinting": self.blueprinting,
            "reference": self.reference,
            "relative_error_percent": self.relative_error_percent,
        }


@dataclass(frozen=True)
class CalculonCaseReport:
    case: str
    model_digest: str
    distributed_digest: str
    portable_digest: str
    pass_checkpoints: tuple[dict[str, Any], ...]
    workload: dict[str, MetricComparison]
    peak_only: IterationEstimate
    calibrated: IterationEstimate
    calculon_stats: dict[str, Any]
    paper_seconds: float | None

    @property
    def calculon_total_seconds(self) -> float:
        return self.calculon_stats["total_time"]

    @property
    def peak_error_percent(self) -> float:
        return (self.peak_only.total - self.calculon_total_seconds) / self.calculon_total_seconds * 100

    @property
    def calibrated_error_percent(self) -> float:
        return (self.calibrated.total - self.calculon_total_seconds) / self.calculon_total_seconds * 100

    @property
    def paper_error_percent(self) -> float | None:
        if self.paper_seconds is None:
            return None
        return (self.calibrated.total - self.paper_seconds) / self.paper_seconds * 100

    def to_dict(self) -> dict[str, Any]:
        return {
            "case": self.case,
            "ir": {
                "model_digest": self.model_digest,
                "distributed_digest": self.distributed_digest,
                "portable_digest": self.portable_digest,
                "pass_checkpoints": list(self.pass_checkpoints),
            },
            "workload_audit": {name: comparison.to_dict() for name, comparison in self.workload.items()},
            "timing_seconds": {
                "peak_only": self.peak_only.total,
                "system_evidence": self.calibrated.total,
                "calculon": self.calculon_total_seconds,
                "paper": self.paper_seconds,
            },
            "timing_error_percent": {
                "peak_only_vs_calculon": self.peak_error_percent,
                "system_evidence_vs_calculon": self.calibrated_error_percent,
                "system_evidence_vs_paper": self.paper_error_percent,
            },
            "estimated_breakdown_seconds": {
                "forward": self.calibrated.forward,
                "backward": self.calibrated.backward,
                "optimizer": self.calibrated.optimizer,
                "recompute": self.calibrated.recompute,
                "tensor_parallel": self.calibrated.tensor_parallel,
                "pipeline_parallel": self.calibrated.pipeline_parallel,
                "data_parallel": self.calibrated.data_parallel,
                "recommunication": self.calibrated.recommunication,
                "pipeline_bubble": self.calibrated.pipeline_bubble,
            },
            "calculon_breakdown_seconds": {
                "forward": self.calculon_stats["fw_time"],
                "backward": self.calculon_stats["bw_time"],
                "optimizer": self.calculon_stats["optim_step_time"],
                "recompute": self.calculon_stats["recompute_time"],
                "tensor_parallel": self.calculon_stats["tp_comm_exposed_time"],
                "pipeline_parallel": self.calculon_stats["pp_comm_exposed_time"],
                "data_parallel": self.calculon_stats["dp_comm_exposed_time"],
                "recommunication": self.calculon_stats["recomm_exposed_time"],
                "pipeline_bubble": self.calculon_stats["bubble_time"],
            },
            "memory_bytes": {
                "estimated": self.calibrated.memory.total,
                "calculon": self.calculon_stats["proc_mem_tier1_cap_req"],
                "relative_error_percent": (
                    (self.calibrated.memory.total - self.calculon_stats["proc_mem_tier1_cap_req"])
                    / self.calculon_stats["proc_mem_tier1_cap_req"]
                    * 100
                ),
            },
            "recompute_counter_audit": {
                "derived_explicit_operations": _phase_operations_from_estimate(
                    self.calibrated, TrainingPhase.RECOMPUTE, local_only=True
                ),
                "calculon_block_re_flops": self.calculon_stats["block_re_flops"],
                "note": (
                    "Calculon accumulates a running forward prefix in block_re_flops; "
                    "block_re_time sums selected operations and is the comparable metric."
                ),
            },
        }


@dataclass(frozen=True)
class CalculonExperimentReport:
    schema: str
    hardware_name: str
    evidence_revision: str
    calibration_policy: dict[str, Any]
    cases: tuple[CalculonCaseReport, ...]

    @property
    def peak_mean_absolute_error_percent(self) -> float:
        return sum(abs(case.peak_error_percent) for case in self.cases) / len(self.cases)

    @property
    def calibrated_mean_absolute_error_percent(self) -> float:
        return sum(abs(case.calibrated_error_percent) for case in self.cases) / len(self.cases)

    @property
    def calibrated_max_absolute_error_percent(self) -> float:
        return max(abs(case.calibrated_error_percent) for case in self.cases)

    @property
    def workload_max_absolute_error_percent(self) -> float:
        return max(abs(metric.relative_error_percent) for case in self.cases for metric in case.workload.values())

    @property
    def paper_mean_absolute_error_percent(self) -> float | None:
        errors = tuple(abs(case.paper_error_percent) for case in self.cases if case.paper_error_percent is not None)
        return sum(errors) / len(errors) if errors else None

    @property
    def paper_max_absolute_error_percent(self) -> float | None:
        errors = tuple(abs(case.paper_error_percent) for case in self.cases if case.paper_error_percent is not None)
        return max(errors) if errors else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "hardware": {
                "name": self.hardware_name,
                "evidence_revision": self.evidence_revision,
            },
            "calibration_policy": self.calibration_policy,
            "summary": {
                "case_count": len(self.cases),
                "peak_only_mean_absolute_error_percent": self.peak_mean_absolute_error_percent,
                "system_evidence_mean_absolute_error_percent": self.calibrated_mean_absolute_error_percent,
                "system_evidence_max_absolute_error_percent": self.calibrated_max_absolute_error_percent,
                "workload_max_absolute_error_percent": self.workload_max_absolute_error_percent,
                "system_evidence_vs_paper_mean_absolute_error_percent": self.paper_mean_absolute_error_percent,
                "system_evidence_vs_paper_max_absolute_error_percent": self.paper_max_absolute_error_percent,
            },
            "cases": [case.to_dict() for case in self.cases],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def discover_seqsel_tab5_cases(data_root: Path) -> tuple[CalculonCase, ...]:
    cases = []
    for model_name, modes in SEQSEL_TABLE5_SECONDS.items():
        for mode, paper_seconds in modes.items():
            cases.append(
                CalculonCase(
                    name=f"seqsel-tab5/{model_name}/{mode}",
                    model_path=data_root / "models" / f"{model_name}.json",
                    execution_path=data_root / "validation" / "seqsel" / "tab5" / f"{model_name}_{mode}.json",
                    system_path=data_root / "systems" / "a100_80g.json",
                    paper_seconds=paper_seconds,
                )
            )
    return tuple(cases)


def _run_calculon(
    model_data: dict[str, Any],
    execution_data: dict[str, Any],
    system_data: dict[str, Any],
) -> dict[str, Any]:
    logger = logging.getLogger("blueprinting.synthesizer.experiments.calculon")
    application = Llm.Application(model_data)
    execution_fields = {field: execution_data[field] for field in Llm.Execution.fields()}
    execution = Llm.Execution.from_json(execution_fields)
    system = System(system_data)
    model = Llm(application, logger)
    model.compile(system, execution)
    model.run(system)
    return model.get_stats_json(False)


def _derive_plan(
    model: TransformerModelSpec,
    execution: TransformerExecutionSpec,
) -> tuple[PortablePlanIR, tuple[dict[str, Any], ...], str, str]:
    source = build_transformer_model_ir(model)
    session = synthesis_session_for(model, execution)
    result = PassManager().run(
        PassPipeline.of(DistributeTransformerTrainingPass(), PlanTransformerTrainingPass()),
        source,
        session=session,
    )
    checkpoints = tuple(
        {
            "pass": checkpoint.record.pass_name,
            "schema": checkpoint.ir.header.schema_name,
            "digest": checkpoint.ir.digest,
        }
        for checkpoint in result.checkpoints
    )
    distributed_digest = result.checkpoints[0].ir.digest
    return result.ir, checkpoints, source.digest, distributed_digest


def _phase_operations_from_estimate(
    estimate: IterationEstimate,
    phase: TrainingPhase,
    *,
    local_only: bool,
) -> int:
    return sum(
        task.invocation.work.operations
        for task in estimate.block.tasks
        if task.invocation.phase is phase and (not local_only or task.invocation.engine is not EngineKind.COLLECTIVE)
    )


def _phase_work(plan: PortablePlanIR, phase: TrainingPhase) -> tuple[int, int, int]:
    operations = 0
    memory_bytes = 0
    message_bytes = 0
    for task in plan.tasks:
        invocation = task.attributes.get("invocation")
        if not isinstance(invocation, PrimitiveInvocation) or invocation.phase is not phase:
            continue
        operations += invocation.work.operations
        memory_bytes += invocation.work.memory_bytes
        message_bytes += invocation.work.message_bytes
    return operations, memory_bytes, message_bytes


def _workload_audit(plan: PortablePlanIR, calculon: dict[str, Any]) -> dict[str, MetricComparison]:
    forward_ops, forward_memory, forward_messages = _phase_work(plan, TrainingPhase.FORWARD)
    agrad_ops, agrad_memory, backward_messages = _phase_work(plan, TrainingPhase.ACTIVATION_GRADIENT)
    wgrad_ops, wgrad_memory, _ = _phase_work(plan, TrainingPhase.WEIGHT_GRADIENT)
    optimizer_ops, optimizer_memory, _ = _phase_work(plan, TrainingPhase.OPTIMIZER)
    _, _, recomm_messages = _phase_work(plan, TrainingPhase.RECOMMUNICATION)
    memory = plan.attributes["block_memory"]
    return {
        "block_forward_operations": MetricComparison(forward_ops, calculon["block_fw_flops"]),
        "block_forward_memory_bytes": MetricComparison(forward_memory, calculon["block_fw_mem_accessed"]),
        "block_activation_gradient_operations": MetricComparison(agrad_ops, calculon["block_agrad_flops"]),
        "block_activation_gradient_memory_bytes": MetricComparison(agrad_memory, calculon["block_agrad_mem_accessed"]),
        "block_weight_gradient_operations": MetricComparison(wgrad_ops, calculon["block_wgrad_flops"]),
        "block_weight_gradient_memory_bytes": MetricComparison(wgrad_memory, calculon["block_wgrad_mem_accessed"]),
        "block_optimizer_operations": MetricComparison(optimizer_ops, calculon["block_optim_flops"]),
        "block_optimizer_memory_bytes": MetricComparison(optimizer_memory, calculon["block_optim_mem_accessed"]),
        "block_forward_tp_message_bytes": MetricComparison(forward_messages, calculon["baseblock_fw_tp_size"]),
        "block_backward_tp_message_bytes": MetricComparison(backward_messages, calculon["baseblock_bw_tp_size"]),
        "block_recommunication_message_bytes": MetricComparison(recomm_messages, calculon["baseblock_recomm_size"]),
        "block_weight_bytes": MetricComparison(memory.weights, calculon["block_weight_space"]),
        "block_activation_working_bytes": MetricComparison(
            memory.activation_working, calculon["block_act_working_space"]
        ),
        "block_activation_storage_bytes": MetricComparison(
            memory.activation_storage, calculon["block_act_storage_space"]
        ),
        "block_activation_checkpoint_bytes": MetricComparison(
            memory.activation_checkpoint, calculon["block_act_checkpoint_size"]
        ),
        "block_weight_gradient_bytes": MetricComparison(memory.weight_gradients, calculon["block_weight_grad_space"]),
        "block_weight_gradient_unsharded_bytes": MetricComparison(
            memory.weight_gradients_unsharded, calculon["block_weight_grad_space_no_sharding"]
        ),
        "block_activation_gradient_bytes": MetricComparison(
            memory.activation_gradients, calculon["block_act_grad_space"]
        ),
        "block_optimizer_bytes": MetricComparison(memory.optimizer, calculon["block_optimizer_space"]),
    }


def run_calculon_experiment(cases: tuple[CalculonCase, ...]) -> CalculonExperimentReport:
    if not cases:
        raise ValueError("Calculon experiment requires at least one case")
    reports = []
    hardware_name = cases[0].system_path.stem
    evidence_revision = ""
    for case in cases:
        model_data = _read_json(case.model_path)
        execution_data = _read_json(case.execution_path)
        system_data = _read_json(case.system_path)
        model = TransformerModelSpec.from_mapping(case.model_path.stem, model_data)
        execution = TransformerExecutionSpec.from_mapping(execution_data)
        plan, checkpoints, model_digest, distributed_digest = _derive_plan(model, execution)
        hardware = HardwareProfile.from_mapping(case.system_path.stem, system_data, datatype=execution.datatype)
        if not evidence_revision:
            evidence_revision = hardware.evidence_revision
        elif evidence_revision != hardware.evidence_revision:
            raise ValueError("one experiment report must use one hardware evidence revision")
        calculon_stats = _run_calculon(model_data, execution_data, system_data)
        reports.append(
            CalculonCaseReport(
                case=case.name,
                model_digest=model_digest,
                distributed_digest=distributed_digest,
                portable_digest=plan.digest,
                pass_checkpoints=checkpoints,
                workload=_workload_audit(plan, calculon_stats),
                peak_only=estimate_iteration(plan, hardware, CalibrationMode.PEAK_ONLY),
                calibrated=estimate_iteration(plan, hardware, CalibrationMode.SYSTEM_EVIDENCE),
                calculon_stats=calculon_stats,
                paper_seconds=case.paper_seconds,
            )
        )
    return CalculonExperimentReport(
        schema="blueprinting.calculon-calibration-experiment.v2",
        hardware_name=hardware_name,
        evidence_revision=evidence_revision,
        calibration_policy={
            "scope": "target-and-datatype",
            "evidence": [
                "matrix operation-size efficiency curve",
                "vector operation-size efficiency curve",
                "memory transfer-size efficiency curve",
                "network bandwidth efficiency and collective volume model",
            ],
            "shared_across_cases": True,
            "fit_against_case_outputs": False,
            "forbidden_inputs": [
                "model name",
                "Calculon duration",
                "paper duration",
                "per-case correction factor",
                "agrad/wgrad reference ratio",
            ],
        },
        cases=tuple(reports),
    )

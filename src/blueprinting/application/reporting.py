"""Presentation-neutral reports shared by Blueprinting application services."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from blueprinting.schema.frozen import FrozenDict
from blueprinting.synthesizer.stages.common import CanonicalIRMixin
from blueprinting.synthesizer.stages.distributed.ir import DistributedTaskIR
from blueprinting.synthesizer.stages.model.ir import ModelIR
from blueprinting.synthesizer.stages.portable_plan.ir import PortablePlanIR


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
    evidence_source_revision: str = ""
    evidence_record_ids: tuple[str, ...] = ()
    evidence_match: str = ""
    evidence_method: str = ""
    evidence_uncertainty: FrozenDict = field(default_factory=FrozenDict)
    evidence_assumptions: tuple[str, ...] = ()


def _diagnostics_from_verification(
    ir: ModelIR | DistributedTaskIR | PortablePlanIR,
) -> tuple[AnalysisDiagnostic, ...]:
    return tuple(
        AnalysisDiagnostic(
            code=item.code,
            message=item.message,
            level=DiagnosticLevel(item.severity.value),
            path=item.path,
            hint=item.hint,
        )
        for item in ir.diagnostics().diagnostics
    )


def stage_report(
    stage: str,
    label: str,
    pass_name: str,
    ir: CanonicalIRMixin,
    duration_ns: int,
) -> IRStageReport:
    """Build an inspectable report for one verified derivation boundary."""

    if isinstance(ir, ModelIR):
        node_count, value_count = len(ir.operations), len(ir.values)
    elif isinstance(ir, DistributedTaskIR):
        node_count, value_count = len(ir.tasks), len(ir.values)
    elif isinstance(ir, PortablePlanIR):
        node_count, value_count = len(ir.tasks), len(ir.buffers)
    else:
        raise TypeError(f"application stage reports do not support {type(ir).__name__}")
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


__all__ = [
    "AnalysisDiagnostic",
    "DiagnosticLevel",
    "IRStageReport",
    "TaskReport",
    "stage_report",
]

"""Typed binding objects for progressive formal synthesis."""

from __future__ import annotations

from dataclasses import field, replace
from enum import Enum
from typing import Any

from typing_extensions import assert_never

from blueprinting.schema.authoring import (
    NonEmptyText,
    NonNegativeInt,
    PositiveInt,
    adt,
    enum,
    record,
    seal_adt,
    variant,
)
from blueprinting.schema.codec import content_digest
from blueprinting.schema.frozen import FrozenDict

from .axes import BindingAxis
from .errors import BindingError, MissingBindingError
from .semantics import EMPTY_SEMANTIC, BindingSemantic


def _reject_reserved_attributes(attributes: FrozenDict, reserved: frozenset[str], field_name: str) -> None:
    def walk(value: Any, path: tuple[str, ...]) -> None:
        if isinstance(value, FrozenDict):
            for key, item in value.items():
                location = path + (key,)
                if key.lower() in reserved:
                    raise BindingError(
                        f"{field_name}.{'.'.join(location)} is a typed semantic field and cannot be an attribute"
                    )
                walk(item, location)
        elif isinstance(value, (tuple, frozenset)):
            for index, item in enumerate(value):
                walk(item, path + (str(index),))

    walk(attributes, ())


_WORKLOAD_RESERVED_ATTRIBUTES = frozenset(
    {
        "workload_spec",
        "phase",
        "inference_phase",
        "batch_size",
        "global_batch_size",
        "microbatch_size",
        "micro_batches",
        "sequence_length",
        "prompt_tokens",
        "generated_tokens",
        "query_tokens",
        "context_tokens",
        "datatype",
    }
)
_STRATEGY_RESERVED_ATTRIBUTES = frozenset(
    {
        "mapping_spec",
        "inference_mapping_spec",
        "parallelism",
        "tensor_parallel",
        "tensor_par",
        "pipeline_parallel",
        "pipeline_par",
        "data_parallel",
        "data_par",
        "replicas",
        "recompute",
        "recompute_policy",
        "activation_recompute",
        "pipeline_schedule",
        "pipeline_interleaving",
        "optimizer_sharding",
        "tensor_parallel_communication",
        "tensor_par_comm_type",
        "world_size",
        "num_procs",
        "fused_activation",
        "sequence_parallel_all_gather_redo",
    }
)


@enum("blueprinting.binding.inference-phase")
class InferencePhase(Enum):
    PREFILL = "prefill"
    DECODE = "decode"


@adt(wire="blueprinting.binding.workload-mode")
class WorkloadMode:
    """Closed workload specialization; inference always carries its phase."""


@variant("training")
class TrainingWorkload(WorkloadMode):
    pass


@variant("inference")
class InferenceWorkload(WorkloadMode):
    phase: InferencePhase


WorkloadModeVariant = TrainingWorkload | InferenceWorkload
seal_adt(WorkloadMode, WorkloadModeVariant)


@record("blueprinting.binding.workload")
class WorkloadBinding:
    mode: WorkloadModeVariant
    semantic: BindingSemantic = EMPTY_SEMANTIC
    attributes: FrozenDict = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        _reject_reserved_attributes(self.attributes, _WORKLOAD_RESERVED_ATTRIBUTES, "workload.attributes")

    @property
    def inference_phase(self) -> InferencePhase | None:
        match self.mode:
            case TrainingWorkload():
                return None
            case InferenceWorkload(phase):
                return phase
        assert_never(self.mode)

    @property
    def fingerprint(self) -> str:
        return content_digest(self, "workload-binding")


@record("blueprinting.binding.strategy")
class StrategyBinding:
    semantic: BindingSemantic = EMPTY_SEMANTIC
    attributes: FrozenDict = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        _reject_reserved_attributes(self.attributes, _STRATEGY_RESERVED_ATTRIBUTES, "strategy.attributes")

    @property
    def fingerprint(self) -> str:
        return content_digest(self, "strategy-binding")


@record("blueprinting.binding.target-requirements")
class TargetRequirements:
    device_count_range: tuple[PositiveInt, PositiveInt] = (1, 2**31 - 1)
    minimum_memory_bytes: NonNegativeInt = 0
    required_dtypes: frozenset[NonEmptyText] = frozenset()
    required_collectives: frozenset[NonEmptyText] = frozenset()
    required_capabilities: frozenset[NonEmptyText] = frozenset()
    attributes: FrozenDict = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        lower, upper = self.device_count_range
        if upper < lower:
            raise BindingError("device_count_range bounds must be ordered")


@record("blueprinting.binding.target")
class TargetProfile:
    name: NonEmptyText
    architecture: NonEmptyText
    architecture_revision: NonEmptyText
    runtime_stack: NonEmptyText
    runtime_revision: NonEmptyText
    target_abi: NonEmptyText
    supported_dtypes: frozenset[NonEmptyText] = frozenset()
    supported_operations: frozenset[NonEmptyText] = frozenset()
    memory_spaces: frozenset[NonEmptyText] = frozenset()
    execution_engines: frozenset[NonEmptyText] = frozenset()
    supported_collectives: frozenset[NonEmptyText] = frozenset()
    capabilities: frozenset[NonEmptyText] = frozenset()
    kernel_library: NonEmptyText = "none"
    collective_library: NonEmptyText = "none"
    attributes: FrozenDict = field(default_factory=FrozenDict)

    @property
    def fingerprint(self) -> str:
        return content_digest(self, "target-profile")

    def satisfies(self, requirements: TargetRequirements) -> bool:
        return (
            requirements.required_dtypes.issubset(self.supported_dtypes)
            and requirements.required_collectives.issubset(self.supported_collectives)
            and requirements.required_capabilities.issubset(self.capabilities)
        )


@record("blueprinting.binding.deployment")
class DeploymentProfile:
    name: NonEmptyText
    device_count: PositiveInt
    topology: FrozenDict
    environment_revision: NonEmptyText
    available_memory_bytes: tuple[NonNegativeInt, ...] = ()
    reserved_resources: FrozenDict = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        if len(self.available_memory_bytes) not in (0, 1, self.device_count):
            raise BindingError("available_memory_bytes must be empty, scalar-like, or per-device")

    @property
    def fingerprint(self) -> str:
        return content_digest(self, "deployment-profile")

    def satisfies(self, requirements: TargetRequirements) -> bool:
        lower, upper = requirements.device_count_range
        if not lower <= self.device_count <= upper:
            return False
        if requirements.minimum_memory_bytes == 0 or not self.available_memory_bytes:
            return True
        memory = self.available_memory_bytes
        if len(memory) == 1:
            memory = memory * self.device_count
        return all(item >= requirements.minimum_memory_bytes for item in memory)


@record("blueprinting.binding.calibration")
class CalibrationBinding:
    evidence_revision: NonEmptyText
    cost_model_revision: NonEmptyText

    @property
    def fingerprint(self) -> str:
        return content_digest(self, "calibration-binding")


BindingValue = WorkloadBinding | StrategyBinding | TargetProfile | DeploymentProfile | CalibrationBinding


@record("blueprinting.binding.set")
class BindingSet:
    workload: WorkloadBinding | None = None
    strategy: StrategyBinding | None = None
    target: TargetProfile | None = None
    deployment: DeploymentProfile | None = None
    calibration: CalibrationBinding | None = None

    def get(self, axis: BindingAxis) -> BindingValue | None:
        if axis is BindingAxis.WORKLOAD:
            return self.workload
        if axis is BindingAxis.STRATEGY:
            return self.strategy
        if axis is BindingAxis.TARGET:
            return self.target
        if axis is BindingAxis.DEPLOYMENT:
            return self.deployment
        if axis is BindingAxis.CALIBRATION:
            return self.calibration
        raise TypeError(f"unsupported binding axis: {axis!r}")

    def has(self, axis: BindingAxis) -> bool:
        return self.get(axis) is not None

    def require(self, *axes: BindingAxis) -> None:
        missing = tuple(axis for axis in axes if not self.has(axis))
        if missing:
            names = ", ".join(axis.value for axis in missing)
            raise MissingBindingError(f"missing required synthesis bindings: {names}")

    def with_binding(self, binding: BindingValue) -> BindingSet:
        if isinstance(binding, WorkloadBinding):
            return replace(self, workload=binding)
        if isinstance(binding, StrategyBinding):
            return replace(self, strategy=binding)
        if isinstance(binding, TargetProfile):
            return replace(self, target=binding)
        if isinstance(binding, DeploymentProfile):
            return replace(self, deployment=binding)
        if isinstance(binding, CalibrationBinding):
            return replace(self, calibration=binding)
        raise TypeError(f"unsupported binding type: {type(binding).__name__}")

    @property
    def fingerprint(self) -> str:
        return content_digest(self, "binding-set")

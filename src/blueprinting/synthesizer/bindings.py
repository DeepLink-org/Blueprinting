"""Typed binding objects for progressive formal synthesis."""

from __future__ import annotations

from dataclasses import field, replace
from enum import Enum
from typing import Any

from typing_extensions import assert_never

from blueprinting.schema.authoring import adt, record, require_adt_variant, seal_adt, variant
from blueprinting.schema.codec import content_digest, enum_type
from blueprinting.schema.frozen import FrozenDict, freeze

from .axes import BindingAxis
from .errors import BindingError, MissingBindingError
from .semantics import EMPTY_SEMANTIC, BindingSemantic


def _frozen_map(value: Any) -> FrozenDict:
    frozen = freeze(value)
    if not isinstance(frozen, FrozenDict):
        raise TypeError("expected a mapping")
    return frozen


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


def _string_set(value: Any, field_name: str) -> frozenset[str]:
    if isinstance(value, (str, bytes)):
        raise BindingError(f"{field_name} must be an iterable of strings, not one string")
    try:
        result = frozenset(value)
    except TypeError as error:
        raise BindingError(f"{field_name} must be an iterable of strings") from error
    if any(not isinstance(item, str) or not item for item in result):
        raise BindingError(f"{field_name} must contain non-empty strings")
    return result


@enum_type("blueprinting.binding.inference-phase")
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

    def __post_init__(self) -> None:
        if not isinstance(self.phase, InferencePhase):
            raise BindingError("inference workload phase must be InferencePhase")


WorkloadModeVariant = TrainingWorkload | InferenceWorkload
seal_adt(WorkloadMode, WorkloadModeVariant)


@record("blueprinting.binding.workload")
class WorkloadBinding:
    mode: WorkloadModeVariant
    semantic: BindingSemantic = EMPTY_SEMANTIC
    attributes: FrozenDict = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        require_adt_variant(self.mode, WorkloadMode, "workload mode")
        if not isinstance(self.semantic, BindingSemantic):
            raise BindingError("workload semantic must be a BindingSemantic")
        attributes = _frozen_map(self.attributes)
        _reject_reserved_attributes(attributes, _WORKLOAD_RESERVED_ATTRIBUTES, "workload.attributes")
        object.__setattr__(self, "attributes", attributes)

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
        if not isinstance(self.semantic, BindingSemantic):
            raise BindingError("strategy semantic must be a BindingSemantic")
        attributes = _frozen_map(self.attributes)
        _reject_reserved_attributes(attributes, _STRATEGY_RESERVED_ATTRIBUTES, "strategy.attributes")
        object.__setattr__(self, "attributes", attributes)

    @property
    def fingerprint(self) -> str:
        return content_digest(self, "strategy-binding")


@record("blueprinting.binding.target-requirements")
class TargetRequirements:
    device_count_range: tuple[int, int] = (1, 2**31 - 1)
    minimum_memory_bytes: int = 0
    required_dtypes: frozenset[str] = frozenset()
    required_collectives: frozenset[str] = frozenset()
    required_capabilities: frozenset[str] = frozenset()
    attributes: FrozenDict = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        device_range = tuple(self.device_count_range)
        if len(device_range) != 2:
            raise BindingError("device_count_range must contain exactly two bounds")
        lower, upper = device_range
        if any(isinstance(value, bool) or not isinstance(value, int) for value in device_range):
            raise BindingError("device_count_range bounds must be integers")
        if lower <= 0 or upper < lower:
            raise BindingError("device_count_range must be positive and ordered")
        if (
            isinstance(self.minimum_memory_bytes, bool)
            or not isinstance(self.minimum_memory_bytes, int)
            or self.minimum_memory_bytes < 0
        ):
            raise BindingError("minimum_memory_bytes must not be negative")
        object.__setattr__(self, "device_count_range", device_range)
        object.__setattr__(self, "required_dtypes", _string_set(self.required_dtypes, "required_dtypes"))
        object.__setattr__(
            self,
            "required_collectives",
            _string_set(self.required_collectives, "required_collectives"),
        )
        object.__setattr__(
            self,
            "required_capabilities",
            _string_set(self.required_capabilities, "required_capabilities"),
        )
        object.__setattr__(self, "attributes", _frozen_map(self.attributes))


@record("blueprinting.binding.target")
class TargetProfile:
    name: str
    architecture: str
    architecture_revision: str
    runtime_stack: str
    runtime_revision: str
    target_abi: str
    supported_dtypes: frozenset[str] = frozenset()
    supported_operations: frozenset[str] = frozenset()
    memory_spaces: frozenset[str] = frozenset()
    execution_engines: frozenset[str] = frozenset()
    supported_collectives: frozenset[str] = frozenset()
    capabilities: frozenset[str] = frozenset()
    kernel_library: str = "none"
    collective_library: str = "none"
    attributes: FrozenDict = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        required = (
            "name",
            "architecture",
            "architecture_revision",
            "runtime_stack",
            "runtime_revision",
            "target_abi",
        )
        if any(not isinstance(getattr(self, name), str) or not getattr(self, name) for name in required):
            raise BindingError("target identity fields must not be empty")
        for name in (
            "supported_dtypes",
            "supported_operations",
            "memory_spaces",
            "execution_engines",
            "supported_collectives",
            "capabilities",
        ):
            object.__setattr__(self, name, _string_set(getattr(self, name), name))
        if any(not isinstance(value, str) or not value for value in (self.kernel_library, self.collective_library)):
            raise BindingError("target library identities must be non-empty strings")
        object.__setattr__(self, "attributes", _frozen_map(self.attributes))

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
    name: str
    device_count: int
    topology: FrozenDict
    environment_revision: str
    available_memory_bytes: tuple[int, ...] = ()
    reserved_resources: FrozenDict = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or not isinstance(self.environment_revision, str)
            or not self.name
            or not self.environment_revision
        ):
            raise BindingError("deployment identity fields must not be empty")
        if isinstance(self.device_count, bool) or not isinstance(self.device_count, int) or self.device_count <= 0:
            raise BindingError("device_count must be a positive integer")
        memory = tuple(self.available_memory_bytes)
        if len(memory) not in (0, 1, self.device_count):
            raise BindingError("available_memory_bytes must be empty, scalar-like, or per-device")
        if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in memory):
            raise BindingError("available memory must not be negative")
        object.__setattr__(self, "available_memory_bytes", memory)
        object.__setattr__(self, "topology", _frozen_map(self.topology))
        object.__setattr__(self, "reserved_resources", _frozen_map(self.reserved_resources))

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
    evidence_revision: str
    cost_model_revision: str

    def __post_init__(self) -> None:
        if any(not isinstance(value, str) or not value for value in (self.evidence_revision, self.cost_model_revision)):
            raise BindingError("calibration revisions must not be empty")

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

    def __post_init__(self) -> None:
        expected = {
            "workload": WorkloadBinding,
            "strategy": StrategyBinding,
            "target": TargetProfile,
            "deployment": DeploymentProfile,
            "calibration": CalibrationBinding,
        }
        for name, expected_type in expected.items():
            value = getattr(self, name)
            if value is not None and not isinstance(value, expected_type):
                raise BindingError(f"{name} binding must be {expected_type.__name__}")

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

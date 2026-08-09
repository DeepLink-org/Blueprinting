"""Typed binding objects for progressive formal synthesis."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

from blueprinting.schema.codec import content_digest, enum_type, record_type
from blueprinting.schema.frozen import FrozenDict, freeze

from .axes import BindingAxis
from .errors import BindingError, MissingBindingError
from .expr import Scalar, ScalarExpr, Symbol, free_symbols


def _frozen_map(value: Any) -> FrozenDict:
    frozen = freeze(value)
    if not isinstance(frozen, FrozenDict):
        raise TypeError("expected a mapping")
    return frozen


def _positive(value: Scalar, field_name: str) -> None:
    if not isinstance(value, (int, float, Symbol, ScalarExpr)):
        raise BindingError(f"{field_name} must be numeric or symbolic")
    if isinstance(value, bool) or (
        isinstance(value, (int, float)) and ((isinstance(value, float) and not math.isfinite(value)) or value <= 0)
    ):
        raise BindingError(f"{field_name} must be finite and greater than zero")
    if any(symbol.axis is not BindingAxis.WORKLOAD for symbol in free_symbols(value)):
        raise BindingError(f"{field_name} may only reference workload symbols")


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


@enum_type("compiler.workload_mode")
class WorkloadMode(Enum):
    TRAINING = "training"
    INFERENCE = "inference"


@enum_type("compiler.inference_phase")
class InferencePhase(Enum):
    PREFILL = "prefill"
    DECODE = "decode"


@record_type("compiler.binding.workload")
@dataclass(frozen=True)
class WorkloadBinding:
    mode: WorkloadMode
    batch_size: Scalar = 1
    sequence_length: Scalar = 1
    micro_batches: Scalar = 1
    inference_phase: InferencePhase | None = None
    attributes: FrozenDict = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        if not isinstance(self.mode, WorkloadMode):
            raise BindingError("mode must be a WorkloadMode")
        _positive(self.batch_size, "batch_size")
        _positive(self.sequence_length, "sequence_length")
        _positive(self.micro_batches, "micro_batches")
        if self.mode is WorkloadMode.TRAINING and self.inference_phase is not None:
            raise BindingError("training workload cannot define an inference phase")
        object.__setattr__(self, "attributes", _frozen_map(self.attributes))

    @property
    def fingerprint(self) -> str:
        return content_digest(self, "workload-binding")


@record_type("compiler.binding.strategy")
@dataclass(frozen=True)
class StrategyBinding:
    tensor_parallel: int = 1
    pipeline_parallel: int = 1
    data_parallel: int = 1
    recompute_policy: str = "none"
    pipeline_policy: str = "none"
    attributes: FrozenDict = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        for name in ("tensor_parallel", "pipeline_parallel", "data_parallel"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise BindingError(f"{name} must be a positive integer")
        if not isinstance(self.recompute_policy, str) or not isinstance(self.pipeline_policy, str):
            raise BindingError("strategy policies must be strings")
        if not self.recompute_policy or not self.pipeline_policy:
            raise BindingError("strategy policies must not be empty")
        object.__setattr__(self, "attributes", _frozen_map(self.attributes))

    @property
    def world_size(self) -> int:
        return self.tensor_parallel * self.pipeline_parallel * self.data_parallel

    @property
    def fingerprint(self) -> str:
        return content_digest(self, "strategy-binding")


@record_type("compiler.target_requirements")
@dataclass(frozen=True)
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


@record_type("compiler.binding.target", field_aliases={"compiler_abi": "target_abi"})
@dataclass(frozen=True)
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


@record_type("compiler.binding.deployment")
@dataclass(frozen=True)
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


@record_type("compiler.binding.calibration")
@dataclass(frozen=True)
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


@record_type("compiler.binding_set")
@dataclass(frozen=True)
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
        return getattr(self, axis.value)

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

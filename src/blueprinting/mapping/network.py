"""Late-bound communication placement for analytical system evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from blueprinting.schema.authoring import record


def _tier(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _aliased_tier(data: Mapping[str, Any], canonical: str, legacy: str) -> int:
    if canonical in data and legacy in data and data[canonical] != data[legacy]:
        raise ValueError(f"{canonical} conflicts with legacy alias {legacy}")
    return _tier(data.get(canonical, data.get(legacy, 0)), canonical)


@record("blueprinting.mapping.network-tier-binding")
class NetworkTierBinding:
    """Map logical parallel domains to ordered tiers of one bound system.

    This binding is intentionally absent from target-neutral workload and
    portable-plan digests. Cost projection supplies it together with a concrete
    :class:`~blueprinting.system.SystemProfile`.
    """

    tensor_parallel: int = 0
    pipeline_parallel: int = 0
    data_parallel: int = 0

    def __post_init__(self) -> None:
        for name in ("tensor_parallel", "pipeline_parallel", "data_parallel"):
            _tier(getattr(self, name), name)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> NetworkTierBinding:
        return cls(
            tensor_parallel=_aliased_tier(data, "tensor_parallel_network", "tensor_par_net"),
            pipeline_parallel=_aliased_tier(data, "pipeline_parallel_network", "pipeline_par_net"),
            data_parallel=_aliased_tier(data, "data_parallel_network", "data_par_net"),
        )

    def validate_capacity(self, tier_count: int) -> None:
        if isinstance(tier_count, bool) or not isinstance(tier_count, int) or tier_count < 0:
            raise ValueError("tier_count must be a non-negative integer")
        for name in ("tensor_parallel", "pipeline_parallel", "data_parallel"):
            if getattr(self, name) >= tier_count:
                raise ValueError(f"{name} network tier is not defined by the bound system")

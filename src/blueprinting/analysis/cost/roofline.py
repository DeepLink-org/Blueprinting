"""Analytical compute/memory roofline and collective communication provider."""

from __future__ import annotations

from ...synthesizer.codec import content_digest
from ...synthesizer.frozen import FrozenDict
from ...system import SystemProfile
from ..cost_model import CalibrationMode
from .protocol import (
    CostEstimate,
    CostProvider,
    CostQuery,
    CostSubject,
    CostSupport,
    EstimateMatch,
    EstimateMethod,
    EstimateUncertainty,
    InvalidCostEvidenceError,
    SupportStatus,
)


class RooflineCostProvider(CostProvider):
    """Cost portable work with explicit peak/evidence rates.

    Local operator latency is ``max(compute, memory)`` by default.  The legacy
    profile's ``no_overlap`` behavior remains available as an explicit option;
    it is never selected by an implicit correction factor.
    """

    def __init__(
        self,
        hardware: SystemProfile,
        *,
        mode: CalibrationMode = CalibrationMode.SYSTEM_EVIDENCE,
        processing_mode: str = "roofline",
    ) -> None:
        if not isinstance(hardware, SystemProfile):
            raise TypeError("hardware must be SystemProfile")
        if not isinstance(mode, CalibrationMode):
            raise TypeError("mode must be CalibrationMode")
        if processing_mode == "profile":
            processing_mode = hardware.processing_mode
        if processing_mode not in {"roofline", "no_overlap"}:
            raise ValueError("processing_mode must be roofline, no_overlap, or profile")
        self._hardware = hardware
        self._mode = mode
        self._processing_mode = processing_mode
        self._name = f"roofline:{hardware.name}"
        self._revision = content_digest(
            FrozenDict(
                {
                    "provider": "blueprinting-roofline-v1",
                    "hardware_revision": hardware.evidence_revision,
                    "calibration_mode": mode.value,
                    "processing_mode": processing_mode,
                }
            ),
            "cost-provider",
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def revision(self) -> str:
        return self._revision

    @property
    def hardware(self) -> SystemProfile:
        return self._hardware

    def supports(self, query: CostQuery) -> CostSupport:
        if query.hardware != self._hardware.name:
            return CostSupport.unavailable("query targets a different system profile")
        if query.datatype != self._hardware.datatype:
            return CostSupport.unavailable("query datatype is not covered by this system profile")
        if query.hardware_revision and query.hardware_revision != self._hardware.evidence_revision:
            return CostSupport.unavailable("query requires a different hardware evidence revision")
        if query.subject is CostSubject.OPERATOR:
            if query.engine not in {"matrix", "vector"}:
                return CostSupport.unavailable(
                    "operator roofline requires engine='matrix' or engine='vector'", "engine"
                )
            return CostSupport.available("compute/memory roofline is defined")
        if query.network_tier >= len(self._hardware.networks):
            return CostSupport.unavailable("system profile does not define the requested network tier")
        network = self._hardware.networks[query.network_tier]
        if query.operation not in network.operations:
            return CostSupport.unavailable("network tier does not define the requested communication operation")
        return CostSupport.available("analytical collective model is defined")

    def estimate(self, query: CostQuery) -> CostEstimate:
        support = self.supports(query)
        if support.status is not SupportStatus.AVAILABLE:
            raise InvalidCostEvidenceError(f"roofline provider cannot estimate query: {support.reason}")

        compute_seconds = 0.0
        memory_seconds = 0.0
        network_seconds = 0.0
        bottleneck = "none"
        assumptions: tuple[str, ...]
        if query.subject is CostSubject.OPERATOR:
            processor = self._hardware.matrix if query.engine == "matrix" else self._hardware.vector
            if query.operations:
                compute_seconds = query.operations / processor.throughput(
                    query.operations,
                    apply_efficiency=self._mode is CalibrationMode.SYSTEM_EVIDENCE,
                )
            transferred_bytes = query.read_bytes + query.write_bytes
            if transferred_bytes:
                memory_seconds = transferred_bytes / self._hardware.memory.throughput(
                    transferred_bytes,
                    apply_efficiency=self._mode is CalibrationMode.SYSTEM_EVIDENCE,
                )
            if self._processing_mode == "roofline":
                seconds = max(compute_seconds, memory_seconds)
                bottleneck = "compute" if compute_seconds >= memory_seconds else "memory"
                assumptions = ("compute and memory service overlap perfectly at the roofline bound",)
            else:
                seconds = compute_seconds + memory_seconds
                bottleneck = "serialized-compute-memory"
                assumptions = ("compute and memory service are serialized",)
        else:
            network = self._hardware.networks[query.network_tier]
            network_seconds = network.time(
                query.operation,
                query.message_bytes,
                query.participants,
                apply_efficiency=self._mode is CalibrationMode.SYSTEM_EVIDENCE,
            )
            seconds = network_seconds
            bottleneck = "network"
            assumptions = ("collective cost follows the selected network tier's volume and latency model",)

        return CostEstimate(
            seconds=seconds,
            provider=self.name,
            provider_revision=self.revision,
            source_revision=self._hardware.evidence_revision,
            method=EstimateMethod.ANALYTICAL,
            match=EstimateMatch.ANALYTICAL,
            uncertainty=EstimateUncertainty(),
            validity_domain=FrozenDict(
                {
                    "hardware": self._hardware.name,
                    "datatype": self._hardware.datatype,
                    "calibration_mode": self._mode.value,
                    "processing_mode": self._processing_mode,
                }
            ),
            components=FrozenDict(
                {
                    "compute_seconds": compute_seconds,
                    "memory_seconds": memory_seconds,
                    "network_seconds": network_seconds,
                    "arithmetic_intensity": (
                        query.operations / (query.read_bytes + query.write_bytes)
                        if query.read_bytes + query.write_bytes
                        else 0.0
                    ),
                    "bottleneck": bottleneck,
                }
            ),
            assumptions=assumptions,
        )

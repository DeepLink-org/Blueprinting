"""Aggregate chip and interconnect descriptions into one system profile."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from blueprinting.schema.codec import content_digest, record_type
from blueprinting.schema.frozen import FrozenDict

from .chip import EfficiencyCurve, EfficiencyPoint, MemoryProfile, ProcessorProfile
from .interconnect import NetworkOperationProfile, NetworkProfile


@record_type("blueprinting.system.profile")
@dataclass(frozen=True)
class SystemProfile:
    """One accelerator system used for analytical evaluation.

    Matrix/vector engines and memory describe chip-local resources; networks
    describe ordered interconnect tiers. ``evidence_revision`` identifies the
    exact imported system evidence snapshot.
    """

    name: str
    datatype: str
    matrix: ProcessorProfile
    vector: ProcessorProfile
    memory: MemoryProfile
    processing_mode: str
    networks: tuple[NetworkProfile, ...]
    evidence_revision: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("system profile name must not be empty")
        if not isinstance(self.datatype, str) or not self.datatype:
            raise ValueError("system profile datatype must not be empty")
        if not isinstance(self.matrix, ProcessorProfile) or not isinstance(self.vector, ProcessorProfile):
            raise TypeError("matrix and vector must be ProcessorProfile")
        if not isinstance(self.memory, MemoryProfile):
            raise TypeError("memory must be MemoryProfile")
        if self.processing_mode not in {"roofline", "no_overlap"}:
            raise ValueError("processing_mode must be roofline or no_overlap")
        networks = tuple(self.networks)
        if any(not isinstance(network, NetworkProfile) for network in networks):
            raise TypeError("networks must contain NetworkProfile values")
        object.__setattr__(self, "networks", networks)
        if not isinstance(self.evidence_revision, str) or not self.evidence_revision:
            raise ValueError("evidence_revision must not be empty")

    @classmethod
    def from_mapping(
        cls,
        name: str,
        data: Mapping[str, Any],
        *,
        datatype: str,
    ) -> SystemProfile:
        """Import the retained Calculon-compatible system profile schema."""

        def processor(section: str) -> ProcessorProfile:
            item = data[section][datatype]
            curve = EfficiencyCurve(
                tuple(
                    EfficiencyPoint(int(giga_operations * 1e9), efficiency)
                    for giga_operations, efficiency in item["gflops_efficiency"]
                )
            )
            return ProcessorProfile(item["tflops"] * 1e12, curve)

        memory_data = data["mem1"]
        memory_curve = EfficiencyCurve(
            tuple(
                EfficiencyPoint(int(megabytes * 1e6), efficiency)
                for megabytes, efficiency in memory_data["MB_efficiency"]
            )
        )
        networks = []
        for network in data["networks"]:
            operations = {
                operation: NetworkOperationProfile(multiplier, 0 if offset is None else offset)
                for operation, (multiplier, offset) in network["ops"].items()
            }
            networks.append(
                NetworkProfile(
                    peak_bytes_per_second=network["bandwidth"] * 1e9,
                    efficiency=network["efficiency"],
                    latency_seconds=network["latency"],
                    participant_capacity=network["size"],
                    operations=FrozenDict(operations),
                )
            )
        revision = content_digest(FrozenDict(dict(data)), f"hardware-profile:{name}:{datatype}")
        return cls(
            name=name,
            datatype=datatype,
            matrix=processor("matrix"),
            vector=processor("vector"),
            memory=MemoryProfile(
                int(memory_data["GiB"] * 1024**3),
                memory_data["GBps"] * 1e9,
                memory_curve,
            ),
            processing_mode=data["processing_mode"],
            networks=tuple(networks),
            evidence_revision=revision,
        )

    def processing_time(self, compute_seconds: float, memory_seconds: float) -> float:
        if self.processing_mode == "roofline":
            return max(compute_seconds, memory_seconds)
        return compute_seconds + memory_seconds

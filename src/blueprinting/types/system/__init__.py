"""System module for blueprinting."""

from typing import List

import hyperparameter as hp

from .memory import Memory
from .network import Network
from .processor import Processor

__all__ = ["System", "Memory", "Processor", "Network"]


class System:
    """Hardware system configuration.

    Examples
    --------
    >>> import json
    >>> import hyperparameter as hp
    >>> sys_cfg = json.load(open("system.json"))
    >>> with hp.scope(sys=sys_cfg):
    ...     system = System()
    ...     system.proc_mode
    'roofline'
    """

    TypeSizes = {
        "float8": 1,
        "float16": 2,
        "float32": 4,
        "bfloat16": 2,
    }

    @staticmethod
    def supported_datatypes() -> List[str]:
        """Return list of supported data types."""
        return list(System.TypeSizes.keys())

    def __init__(self, cfg=None) -> None:
        if cfg is None:
            cfg = hp.scope()

        self.matrix = Processor("matrix")
        self.vector = Processor("vector")
        self.datatype = None

        self.mem1 = Memory("mem1")
        self.mem2 = Memory("mem2")

        self.proc_mode = cfg.processing_mode | "roofline"
        assert self.proc_mode in ["roofline", "no_overlap"]

        # Networks is a list of dicts, create each with its own scope
        networks_cfg = cfg.networks | []
        self.networks = []
        for net_dict in networks_cfg:
            with hp.scope(**net_dict):
                self.networks.append(Network())

    @property
    def num_networks(self) -> int:
        """Number of networks."""
        return len(self.networks)

    def get_network(self, tier: int) -> Network:
        """Get network by tier."""
        assert tier < len(self.networks), f"Bad network tier ID: {tier}"
        return self.networks[tier]

    def set_datatype(self, datatype: str) -> None:
        """Set the data type for processing."""
        assert datatype in System.TypeSizes, f"Unsupported data type: {datatype}"
        self.datatype = datatype

    def get_matrix_throughput(self, flops: int) -> float:
        """Get matrix processor throughput."""
        return self.matrix.throughput(self.datatype, flops)

    def get_vector_throughput(self, flops: int) -> float:
        """Get vector processor throughput."""
        return self.vector.throughput(self.datatype, flops)

    def get_mem1_throughput(self, size: int) -> float:
        """Get primary memory throughput."""
        return self.mem1.throughput(size)

    def get_mem2_throughput(self, size: int) -> float:
        """Get secondary memory throughput."""
        return self.mem2.throughput(size)

    def compute_offload_time(self, size: int) -> float:
        """Compute time to offload data to secondary memory."""
        return size / self.mem2.throughput(size)

    def get_processing_time(self, flops_time: float, mem_time: float) -> float:
        """Get processing time based on processing mode."""
        if self.proc_mode == "roofline":
            return max(flops_time, mem_time)
        elif self.proc_mode == "no_overlap":
            return flops_time + mem_time
        return flops_time + mem_time

"""Network configuration for blueprinting."""

from dataclasses import dataclass

import hyperparameter as hp

__all__ = ["Network", "NetworkOp"]


@dataclass
class NetworkOp:
    """Network operation configuration."""

    scalar: float
    offset: float


class Network:
    """Network configuration.

    Examples
    --------
    >>> import hyperparameter as hp
    >>> with hp.scope(bandwidth=100, efficiency=0.9, size=8, latency=1e-6):
    ...     net = Network()
    ...     net.bandwidth
    100000000000.0
    """

    COLLECTIVES = {"reduce_scatter", "all_gather", "all_reduce"}
    NET_OPS = {"p2p", "reduce_scatter", "all_gather", "all_reduce"}

    def __init__(self) -> None:
        """Initialize Network from current hp.scope."""
        cfg = hp.scope()

        self._bandwidth = (cfg.bandwidth | 0) * 1e9  # GB/s -> B/s
        self._efficiency = cfg.efficiency | 1.0
        assert 0 < self._efficiency <= 1.0
        self._size = cfg.size | 0
        self._latency = cfg.latency | 0
        self._must_be_filled = cfg.must_be_filled | False
        self._processor_usage = cfg.processor_usage | 0.0
        assert 0.0 <= self._processor_usage < 1.0

        # Get ops configuration - extract op names from flat keys
        self._ops = {}
        all_keys = cfg.keys()
        ops_keys = [k for k in all_keys if k.startswith("ops.")]
        op_names = {k.split(".")[1] for k in ops_keys if len(k.split(".")) > 1}

        for op in op_names:
            op_cfg = getattr(cfg.ops, op)
            # op_cfg is a list [scalar, offset]
            params = op_cfg | [1.0, None]
            if isinstance(params, (list, tuple)):
                scalar, offset = params
            else:
                scalar = 1.0
                offset = None

            assert op in self.NET_OPS, f"Invalid network op: {op}"
            assert scalar > 0.0, f"Invalid network scalar for {op}: {scalar}"

            if op in self.COLLECTIVES:
                assert offset is not None, f"Must give offset for {op}"
                self._ops[op] = NetworkOp(scalar, offset)
            else:
                assert offset is None, f"Can't give offset for {op}"
                self._ops[op] = NetworkOp(scalar, 0)

    @property
    def bandwidth(self) -> float:
        """Bandwidth in bytes/second."""
        return self._bandwidth

    @property
    def size(self) -> int:
        """Network size (number of nodes)."""
        return self._size

    @property
    def must_be_filled(self) -> bool:
        """Whether network must be fully utilized."""
        return self._must_be_filled

    @property
    def processor_usage(self) -> float:
        """Processor usage during network operations."""
        return self._processor_usage

    def time(self, op: str, op_size: int, comm_size: int) -> float:
        """Compute time for a network operation.

        Args:
            op: Operation name (p2p, reduce_scatter, all_gather, all_reduce)
            op_size: Operation size in bytes
            comm_size: Number of participants in operation

        Returns:
            Time needed for operation in seconds
        """
        if op not in self.COLLECTIVES:
            assert comm_size == 2
        else:
            assert comm_size >= 2
        assert op in self.NET_OPS
        assert op_size >= 0

        net_op = self._ops[op]

        # Scale the op_size by the scalar
        scaled_size = op_size * net_op.scalar

        # Scale the op_size by the op offset
        chunk_size = scaled_size / comm_size
        total_size = scaled_size + chunk_size * net_op.offset

        # Calculate time based on raw bandwidth, efficiency, and latency
        return self._latency + total_size / (self._bandwidth * self._efficiency)

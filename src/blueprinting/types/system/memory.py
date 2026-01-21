"""Memory configuration for blueprinting."""

import hyperparameter as hp

__all__ = ["Memory"]


class Memory:
    """Memory configuration.

    Examples
    --------
    >>> import hyperparameter as hp
    >>> with hp.scope(**{"mem1.GiB": 80, "mem1.GBps": 2000, "mem1.MB_efficiency": [(100, 0.9)]}):
    ...     mem = Memory("mem1")
    ...     mem.capacity
    85899345920
    """

    def __init__(self, prefix: str) -> None:
        """Initialize Memory.

        Args:
            prefix: The configuration prefix (e.g., "mem1", "mem2")
        """
        cfg = getattr(hp.scope(), prefix)

        self.capacity = (cfg.GiB | 0) * 1024**3
        self.bandwidth = (cfg.GBps | 0) * 1e9
        self._efficiency = []
        for mbytes, eff in (cfg.MB_efficiency | []):
            bytes = mbytes * 1e6
            assert 0 < eff <= 1.0
            self._efficiency.append((bytes, eff))

    def efficiency(self, op_bytes: int) -> float:
        """Get efficiency for given operation size."""
        for bytes, eff in self._efficiency:
            if op_bytes >= bytes:
                return eff
        raise ValueError(f"OP bytes {op_bytes} wasn't covered by efficiency curve")

    def throughput(self, op_bytes: int) -> float:
        """Get throughput for given operation size."""
        return self.bandwidth * self.efficiency(op_bytes)

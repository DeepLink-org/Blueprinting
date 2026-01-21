"""Processor configuration for blueprinting."""

import hyperparameter as hp

__all__ = ["Processor"]


class Processor:
    """Processor configuration with support for multiple data types.

    Examples
    --------
    >>> import hyperparameter as hp
    >>> with hp.scope(**{"matrix.float16.tflops": 312, "matrix.float16.gflops_efficiency": [(100, 0.9)]}):
    ...     proc = Processor("matrix")
    ...     proc.flops("float16")
    312000000000000.0
    """

    def __init__(self, prefix: str) -> None:
        """Initialize Processor.

        Args:
            prefix: The configuration prefix (e.g., "matrix", "vector")
        """
        cfg = hp.scope()
        self._datatypes = {}

        # Get keys that match this prefix (e.g., "matrix.float16.tflops")
        all_keys = cfg.keys()
        prefix_dot = f"{prefix}."

        # Extract unique data types under this prefix
        # "matrix.float16.tflops" -> "float16"
        dtypes = set()
        for k in all_keys:
            if k.startswith(prefix_dot):
                parts = k[len(prefix_dot) :].split(".")
                if parts:
                    dtypes.add(parts[0])

        # Build configuration for each data type
        for dtype in dtypes:
            dt_cfg = getattr(getattr(cfg, prefix), dtype)
            tflops = dt_cfg.tflops | 0
            gflops_eff = dt_cfg.gflops_efficiency | []

            self._datatypes[dtype] = {"flops": tflops * 1e12, "efficiency": []}
            last = None
            for gflops, eff in gflops_eff:
                flops = gflops * 1e9
                assert 0 < eff <= 1.0
                if last:
                    assert flops < last
                last = flops
                self._datatypes[dtype]["efficiency"].append((flops, eff))

    def flops(self, datatype: str) -> float:
        """Get peak flops for given data type."""
        return self._datatypes[datatype]["flops"]

    def efficiency(self, datatype: str, op_flops: int) -> float:
        """Get efficiency for given data type and operation flops."""
        for flops, eff in self._datatypes[datatype]["efficiency"]:
            if op_flops >= flops:
                return eff
        raise ValueError(f"{op_flops} wasn't covered in {datatype} efficiency curve")

    def throughput(self, datatype: str, op_flops: int) -> float:
        """Get throughput for given data type and operation flops."""
        if datatype not in self._datatypes:
            raise ValueError(f"Unsupported data type: {datatype}")
        return self.flops(datatype) * self.efficiency(datatype, op_flops)

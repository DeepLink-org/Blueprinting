"""System configuration module - Compatible with Calculon config format.

This module provides a unified way to load and use hardware system configurations,
supporting both Calculon's JSON format and simpler dictionary configs.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


class EfficiencyCurve:
    """Efficiency curve that maps operation size to efficiency factor.
    
    Calculon uses efficiency curves to model how hardware utilization
    varies with operation size (smaller operations are less efficient).
    """
    
    def __init__(self, curve: List[Tuple[float, float]], unit_scale: float = 1.0):
        """Initialize efficiency curve.
        
        Args:
            curve: List of (threshold, efficiency) tuples, sorted by threshold descending.
                   threshold is in base units (after scaling), efficiency is 0-1.
            unit_scale: Scale factor to convert config units to base units.
                       e.g., 1e9 for GFLOPs -> FLOPs, 1e6 for MB -> bytes
        """
        # Sort by threshold descending
        self._curve = sorted(curve, key=lambda x: x[0], reverse=True)
        self._unit_scale = unit_scale
    
    def efficiency(self, value: float) -> float:
        """Get efficiency for a given operation size.
        
        Args:
            value: Operation size in base units (FLOPs, bytes, etc.)
            
        Returns:
            Efficiency factor (0-1)
        """
        for threshold, eff in self._curve:
            if value >= threshold:
                return eff
        # Return last (smallest threshold) efficiency if nothing matched
        return self._curve[-1][1] if self._curve else 1.0
    
    def throughput(self, peak: float, value: float) -> float:
        """Get effective throughput for a given operation size.
        
        Args:
            peak: Peak throughput (FLOPs/s, bytes/s, etc.)
            value: Operation size in base units
            
        Returns:
            Effective throughput = peak * efficiency(value)
        """
        return peak * self.efficiency(value)


class ProcessorConfig:
    """Processor (compute) configuration."""
    
    def __init__(self, cfg: Dict):
        """Initialize from Calculon processor config.
        
        Args:
            cfg: Processor config dict with 'matrix' and 'vector' sections
        """
        self._datatypes = {}
        
        # Parse matrix config (for matrix operations like matmul)
        if 'matrix' in cfg:
            for dtype, dtype_cfg in cfg['matrix'].items():
                self._datatypes[f"matrix_{dtype}"] = {
                    "peak_tflops": dtype_cfg.get('tflops', 0),
                    "efficiency": EfficiencyCurve(
                        [(gflops * 1e9, eff) for gflops, eff in dtype_cfg.get('gflops_efficiency', [[0, 1.0]])],
                        unit_scale=1e9
                    )
                }
        
        # Parse vector config (for element-wise operations)
        if 'vector' in cfg:
            for dtype, dtype_cfg in cfg['vector'].items():
                self._datatypes[f"vector_{dtype}"] = {
                    "peak_tflops": dtype_cfg.get('tflops', 0),
                    "efficiency": EfficiencyCurve(
                        [(gflops * 1e9, eff) for gflops, eff in dtype_cfg.get('gflops_efficiency', [[0, 1.0]])],
                        unit_scale=1e9
                    )
                }
    
    def get_peak_tflops(self, engine: str = "matrix", dtype: str = "float16") -> float:
        """Get peak TFLOPs for given engine and dtype."""
        key = f"{engine}_{dtype}"
        return self._datatypes.get(key, {}).get("peak_tflops", 0)
    
    def get_throughput(self, flops: float, engine: str = "matrix", dtype: str = "float16") -> float:
        """Get effective throughput in FLOPs/s for given operation size.
        
        Args:
            flops: Operation FLOPs
            engine: "matrix" or "vector"
            dtype: Data type (e.g., "float16", "float8")
            
        Returns:
            Effective throughput in FLOPs/s
        """
        key = f"{engine}_{dtype}"
        if key not in self._datatypes:
            # Fallback to peak without efficiency
            return self.get_peak_tflops(engine, dtype) * 1e12
        
        peak = self._datatypes[key]["peak_tflops"] * 1e12  # TFLOPs -> FLOPs/s
        efficiency_curve = self._datatypes[key]["efficiency"]
        return efficiency_curve.throughput(peak, flops)
    
    def get_efficiency(self, flops: float, engine: str = "matrix", dtype: str = "float16") -> float:
        """Get efficiency factor for given operation size."""
        key = f"{engine}_{dtype}"
        if key not in self._datatypes:
            return 1.0
        return self._datatypes[key]["efficiency"].efficiency(flops)


class MemoryConfig:
    """Memory configuration."""
    
    def __init__(self, cfg: Dict):
        """Initialize from Calculon memory config.
        
        Args:
            cfg: Memory config dict with 'GiB', 'GBps', 'MB_efficiency'
        """
        self._capacity_bytes = cfg.get('GiB', 0) * 1024**3
        self._bandwidth_bps = cfg.get('GBps', 0) * 1e9
        
        # Parse efficiency curve
        mb_efficiency = cfg.get('MB_efficiency', [[0, 1.0]])
        self._efficiency = EfficiencyCurve(
            [(mb * 1e6, eff) for mb, eff in mb_efficiency],
            unit_scale=1e6
        )
    
    @property
    def capacity(self) -> float:
        """Memory capacity in bytes."""
        return self._capacity_bytes
    
    @property
    def bandwidth(self) -> float:
        """Peak bandwidth in bytes/s."""
        return self._bandwidth_bps
    
    def get_throughput(self, access_bytes: float) -> float:
        """Get effective bandwidth for given access size.
        
        Args:
            access_bytes: Memory access size in bytes
            
        Returns:
            Effective bandwidth in bytes/s
        """
        return self._efficiency.throughput(self._bandwidth_bps, access_bytes)
    
    def get_efficiency(self, access_bytes: float) -> float:
        """Get efficiency factor for given access size."""
        return self._efficiency.efficiency(access_bytes)


class NetworkConfig:
    """Network configuration."""
    
    def __init__(self, cfg: Dict):
        """Initialize from Calculon network config.
        
        Args:
            cfg: Network config dict with 'bandwidth', 'efficiency', 'size', etc.
        """
        self._bandwidth_gbps = cfg.get('bandwidth', 0)
        self._efficiency = cfg.get('efficiency', 1.0)
        self._size = cfg.get('size', 8)
        self._latency = cfg.get('latency', 0)
        self._processor_usage = cfg.get('processor_usage', 0)
        self._must_be_filled = cfg.get('must_be_filled', False)
        
        # Operation-specific configs
        self._ops = cfg.get('ops', {})
    
    @property
    def bandwidth(self) -> float:
        """Peak bandwidth in bytes/s."""
        return self._bandwidth_gbps * 1e9
    
    @property
    def efficiency(self) -> float:
        """Network efficiency factor."""
        return self._efficiency
    
    @property
    def size(self) -> int:
        """Network size (max number of peers)."""
        return self._size
    
    @property
    def latency(self) -> float:
        """Network latency in seconds."""
        return self._latency
    
    @property
    def processor_usage(self) -> float:
        """Fraction of processor used during communication."""
        return self._processor_usage
    
    def get_throughput(self, comm_bytes: float = None) -> float:
        """Get effective bandwidth.
        
        For now, uses fixed efficiency. Could be extended to
        model bandwidth vs message size.
        """
        return self.bandwidth * self._efficiency
    
    def time(self, op_type: str, comm_bytes: float, num_peers: int) -> float:
        """Compute communication time.
        
        Args:
            op_type: "all_reduce", "all_gather", "reduce_scatter", "p2p"
            comm_bytes: Total bytes to communicate
            num_peers: Number of participating peers
            
        Returns:
            Communication time in seconds
        """
        if num_peers <= 1:
            return 0.0
        
        eff_bandwidth = self.get_throughput(comm_bytes)
        
        # Prefer Calculon-style op scaling if config provides it
        op_cfg = self._ops.get(op_type)
        if op_cfg is not None and isinstance(op_cfg, (list, tuple)) and len(op_cfg) == 2:
            scalar, offset = op_cfg
            op_size = comm_bytes * scalar
            if offset is not None and num_peers > 0:
                chunk_size = op_size / num_peers
                op_size += chunk_size * offset
            return self._latency + op_size / eff_bandwidth
        
        # Fallback: ring algorithm factor
        if op_type in ("all_reduce",):
            factor = 2 * (num_peers - 1) / num_peers
        elif op_type in ("all_gather", "reduce_scatter"):
            factor = (num_peers - 1) / num_peers
        else:
            factor = 1.0
        
        return (comm_bytes * factor / eff_bandwidth) + self._latency


class SystemConfig:
    """Unified system configuration compatible with Calculon format.
    
    Can be initialized from:
    1. Calculon JSON config file path
    2. Calculon config dict
    3. Simple config dict (for backward compatibility)
    """
    
    def __init__(
        self,
        config: Union[str, Path, Dict],
        datatype: str = "float16"
    ):
        """Initialize system config.
        
        Args:
            config: Either:
                - Path to Calculon JSON config file
                - Calculon config dict
                - Simple config dict with peak_tflops, memory_bandwidth_gbps, etc.
            datatype: Default data type for compute operations
        """
        self.datatype = datatype
        
        # Load config
        if isinstance(config, (str, Path)):
            with open(config) as f:
                cfg = json.load(f)
        else:
            cfg = config
        
        # Detect config format
        if 'matrix' in cfg or 'vector' in cfg:
            # Calculon format
            self._init_from_calculon(cfg)
        else:
            # Simple format (backward compatibility)
            self._init_from_simple(cfg)
    
    def _init_from_calculon(self, cfg: Dict):
        """Initialize from Calculon config format."""
        self.processor = ProcessorConfig(cfg)
        self.memory = MemoryConfig(cfg.get('mem1', {}))
        self.memory2 = MemoryConfig(cfg.get('mem2', {})) if 'mem2' in cfg else None
        
        # Parse networks
        self.networks = []
        for net_cfg in cfg.get('networks', []):
            self.networks.append(NetworkConfig(net_cfg))
        
        # Processing mode
        self.processing_mode = cfg.get('processing_mode', 'roofline')
        
        # Store raw config
        self._raw_config = cfg
    
    def _init_from_simple(self, cfg: Dict):
        """Initialize from simple config format (backward compatibility)."""
        # Create pseudo-Calculon config
        peak_tflops = cfg.get('peak_tflops', 312)
        mem_bw_gbps = cfg.get('memory_bandwidth_gbps', 2000)
        mem_cap_gb = cfg.get('memory_capacity_gb', 80)
        net_bw_gbps = cfg.get('network_bandwidth_gbps', 400)
        
        # Build Calculon-like config
        calc_cfg = {
            'matrix': {
                'float16': {
                    'tflops': peak_tflops,
                    'gflops_efficiency': [[0, 1.0]]  # Fixed efficiency
                }
            },
            'vector': {
                'float16': {
                    'tflops': peak_tflops / 8,  # Vector typically slower
                    'gflops_efficiency': [[0, 1.0]]
                }
            },
            'mem1': {
                'GiB': mem_cap_gb,
                'GBps': mem_bw_gbps,
                'MB_efficiency': [[0, 1.0]]  # Fixed efficiency
            },
            'networks': [{
                'bandwidth': net_bw_gbps,
                'efficiency': cfg.get('network_efficiency', 0.9),
                'size': cfg.get('network_size', 8),
                'latency': 0,
                'processor_usage': 0
            }],
            'processing_mode': cfg.get('processing_mode', 'roofline')
        }
        
        self._init_from_calculon(calc_cfg)
    
    # Convenience properties
    @property
    def peak_tflops(self) -> float:
        """Peak matrix TFLOPs for default dtype."""
        return self.processor.get_peak_tflops("matrix", self.datatype)
    
    @property
    def memory_bandwidth(self) -> float:
        """Peak memory bandwidth in bytes/s."""
        return self.memory.bandwidth
    
    @property
    def memory_capacity(self) -> float:
        """Memory capacity in bytes."""
        return self.memory.capacity
    
    @property
    def network_bandwidth(self) -> float:
        """Primary network bandwidth in bytes/s."""
        return self.networks[0].bandwidth if self.networks else 0
    
    @property
    def network_efficiency(self) -> float:
        """Primary network efficiency."""
        return self.networks[0].efficiency if self.networks else 1.0
    
    # Throughput methods (with dynamic efficiency)
    def get_compute_throughput(self, flops: float, engine: str = "matrix") -> float:
        """Get effective compute throughput for given operation.
        
        Args:
            flops: Operation FLOPs
            engine: "matrix" for matmul, "vector" for element-wise
            
        Returns:
            Effective throughput in FLOPs/s
        """
        return self.processor.get_throughput(flops, engine, self.datatype)
    
    def get_memory_throughput(self, access_bytes: float) -> float:
        """Get effective memory bandwidth for given access.
        
        Args:
            access_bytes: Memory access size in bytes
            
        Returns:
            Effective bandwidth in bytes/s
        """
        return self.memory.get_throughput(access_bytes)
    
    def get_network_throughput(self, comm_bytes: float = None, tier: int = 0) -> float:
        """Get effective network bandwidth.
        
        Args:
            comm_bytes: Communication size in bytes (for potential size-based efficiency)
            tier: Network tier (0 = primary, usually NVLink)
            
        Returns:
            Effective bandwidth in bytes/s
        """
        if tier < len(self.networks):
            return self.networks[tier].get_throughput(comm_bytes)
        return 0
    
    def get_processing_time(self, flops_time: float, mem_time: float) -> float:
        """Get processing time using configured processing mode.
        
        Args:
            flops_time: Compute time (FLOPs / throughput)
            mem_time: Memory access time (bytes / throughput)
            
        Returns:
            Processing time based on mode:
            - "roofline": max(flops_time, mem_time)
            - "no_overlap": flops_time + mem_time
        """
        if self.processing_mode == 'roofline':
            return max(flops_time, mem_time)
        else:  # no_overlap
            return flops_time + mem_time
    
    def compute_op_time(
        self,
        flops: float,
        memory_bytes: float,
        engine: str = "matrix"
    ) -> Tuple[float, str, float]:
        """Compute operation time with dynamic efficiency.
        
        Args:
            flops: Operation FLOPs
            memory_bytes: Memory access in bytes
            engine: "matrix" or "vector"
            
        Returns:
            Tuple of (time, bound, arithmetic_intensity):
            - time: Execution time in seconds
            - bound: "compute" or "memory"
            - arithmetic_intensity: FLOPs / bytes
        """
        compute_throughput = self.get_compute_throughput(flops, engine)
        memory_throughput = self.get_memory_throughput(memory_bytes)
        
        flops_time = flops / compute_throughput if compute_throughput > 0 else 0
        mem_time = memory_bytes / memory_throughput if memory_throughput > 0 else 0
        
        time = self.get_processing_time(flops_time, mem_time)
        bound = "compute" if flops_time >= mem_time else "memory"
        ai = flops / memory_bytes if memory_bytes > 0 else float('inf')
        
        return time, bound, ai
    
    def compute_comm_time(
        self,
        comm_bytes: float,
        op_type: str = "all_reduce",
        num_peers: int = 8,
        tier: int = 0
    ) -> float:
        """Compute communication time.
        
        Args:
            comm_bytes: Bytes to communicate
            op_type: "all_reduce", "all_gather", "reduce_scatter", "p2p"
            num_peers: Number of participating devices
            tier: Network tier
            
        Returns:
            Communication time in seconds
        """
        if tier < len(self.networks):
            return self.networks[tier].time(op_type, comm_bytes, num_peers)
        return 0
    
    # Efficiency getters (for debugging/analysis)
    def get_compute_efficiency(self, flops: float, engine: str = "matrix") -> float:
        """Get compute efficiency for given operation size."""
        return self.processor.get_efficiency(flops, engine, self.datatype)
    
    def get_memory_efficiency(self, access_bytes: float) -> float:
        """Get memory efficiency for given access size."""
        return self.memory.get_efficiency(access_bytes)
    
    def to_dict(self) -> Dict:
        """Export config as simple dict (for backward compatibility)."""
        return {
            "peak_tflops": self.peak_tflops,
            "memory_bandwidth_gbps": self.memory_bandwidth / 1e9,
            "memory_capacity_gb": self.memory_capacity / 1e9,
            "network_bandwidth_gbps": self.network_bandwidth / 1e9,
            "processing_mode": self.processing_mode,
        }


def load_system_config(config_path: Union[str, Path]) -> SystemConfig:
    """Load system configuration from file.
    
    Args:
        config_path: Path to Calculon JSON config file
        
    Returns:
        SystemConfig instance
    """
    return SystemConfig(config_path)

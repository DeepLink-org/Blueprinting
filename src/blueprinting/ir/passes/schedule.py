"""SchedulePass - Generate execution schedule from GraphIR.

This pass transforms GraphIR into ScheduleIR by:
1. Topologically sorting operations
2. Computing execution times based on system capabilities
3. Handling overlap between compute and communication
4. Tracking tensor lifetimes for memory analysis
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

from sympy import Expr, Symbol

from ..graph import GraphIR, OpNode
from ..schedule import ScheduleIR, ScheduledOp, TensorLifetime
from ..system import SystemConfig
from .base import Pass


from ..symmax import sym_max, _to_float, eval_lazy


def _to_python_float(x):
    """Convert to Python float, handling SymPy Float and int."""
    return _to_float(x)


def lazy_max(a, b):
    """Max that works with both numeric and symbolic values.
    
    For numeric values, returns Python float (fast).
    For symbolic, returns SymMax that evaluates correctly when substituted.
    """
    return sym_max(a, b)


class SchedulePass(Pass):
    """Pass to generate execution schedule from graph.
    
    This pass computes timing information and creates a ScheduleIR
    that can be used to analyze memory usage and execution time.
    
    The timing model supports two modes (compatible with Calculon):
    - "roofline": time = max(compute_time, memory_time)
    - "no_overlap": time = compute_time + memory_time
    
    Efficiency factors are loaded from system config (dynamic, size-dependent)
    or can be overridden with fixed values via calibration dict.
    """
    
    def __init__(
        self,
        system_config: Optional[Union[Dict, str, Path, SystemConfig]] = None,
        strategy: str = "sequential",  # "sequential", "1f1b", "interleaved"
        overlap_compute_comm: bool = True,
        processing_mode: Optional[str] = None,  # Override config's processing_mode
        calibration: Optional[Dict[str, float]] = None,  # Override dynamic efficiency
        training: bool = True,  # Whether to include backward pass
    ):
        """Initialize SchedulePass.
        
        Args:
            system_config: Hardware system configuration, can be:
                - Path to Calculon JSON config file (str or Path)
                - Calculon config dict (with 'matrix', 'mem1', etc.)
                - Simple config dict (peak_tflops, memory_bandwidth_gbps, etc.)
                - SystemConfig instance
            strategy: Scheduling strategy for pipeline parallelism
            overlap_compute_comm: Whether compute and communication can overlap
            processing_mode: Override config's processing_mode ("roofline" or "no_overlap")
            calibration: Override dynamic efficiency with fixed factors:
                - compute_efficiency: Fixed compute efficiency (0-1)
                - memory_efficiency: Fixed memory efficiency (0-1)
                - network_efficiency: Fixed network efficiency (0-1)
            training: Whether to include backward pass FLOPs/memory
        """
        self.strategy = strategy
        self.overlap_compute_comm = overlap_compute_comm
        self.training = training
        self.calibration = calibration  # None means use dynamic efficiency from config
        
        # Load system config
        if system_config is None:
            self.sys = SystemConfig({})  # Default config
        elif isinstance(system_config, SystemConfig):
            self.sys = system_config
        elif isinstance(system_config, (str, Path)):
            self.sys = SystemConfig(system_config)
        else:
            self.sys = SystemConfig(system_config)
        
        # Override processing mode if specified
        if processing_mode is not None:
            self.processing_mode = processing_mode
        else:
            self.processing_mode = self.sys.processing_mode
        
        # Store for metadata
        self.peak_tflops = self.sys.peak_tflops
        self.memory_bandwidth = self.sys.memory_bandwidth
        self.network_bandwidth = self.sys.network_bandwidth
        self.memory_capacity = self.sys.memory_capacity
    
    def run(self, ir: GraphIR) -> ScheduleIR:
        """Execute the schedule pass."""
        schedule = ScheduleIR(
            num_devices=ir.metadata.get("pp", 1),
            num_stages=ir.metadata.get("pp", 1),
        )
        
        # Get topological order
        topo_order = ir.topological_sort()
        
        # Track current time per device
        device_time: Dict[int, Union[float, Expr]] = {}
        
        # Track scheduled ops by ID for quick lookup (O(1) instead of O(n))
        scheduled_ops_index: Dict[str, ScheduledOp] = {}
        
        # Track tensor lifetimes
        tensor_last_use: Dict[str, str] = {}  # tensor_id -> last_using_op_id
        
        # First pass: determine tensor lifetimes
        for node_id in topo_order:
            node = ir.get_node(node_id)
            if node:
                for inp in node.inputs:
                    tensor_last_use[inp] = node_id
        
        # Second pass: schedule operations
        for node_id in topo_order:
            node = ir.get_node(node_id)
            if node is None:
                continue
            
            device = node.device or 0
            
            # Compute operation duration
            duration = self._compute_duration(node)
            
            # Determine start time based on dependencies
            start_time = self._compute_start_time(node, ir, device_time, scheduled_ops_index)
            
            # Determine stream
            stream = "comm" if node.op_type in ("AllReduce", "AllGather", "ReduceScatter") else "compute"
            
            # Create scheduled op with workload attributes
            attrs = node.attrs.copy()
            # Copy workload attributes from OpNode to attrs for EvaluatePass
            if node.flops_fw is not None:
                attrs["flops_fw"] = node.flops_fw
            if node.flops_bw is not None:
                attrs["flops_bw"] = node.flops_bw
            if node.weight_bytes is not None:
                attrs["weight_bytes"] = node.weight_bytes
            if node.activation_bytes is not None:
                attrs["activation_bytes"] = node.activation_bytes
            if node.memory_fw is not None:
                attrs["memory_fw"] = node.memory_fw
            if node.memory_bw is not None:
                attrs["memory_bw"] = node.memory_bw
            
            scheduled_op = ScheduledOp(
                op_id=node_id,
                op_type=node.op_type,
                device=device,
                stream=stream,
                start=start_time,
                duration=duration,
                attrs=attrs,
            )
            
            # Track tensor allocations
            for output_id in node.outputs:
                tensor = ir.get_tensor(output_id)
                if tensor:
                    lifetime = TensorLifetime(
                        tensor_id=output_id,
                        alloc_time=start_time,
                        free_time=None,  # Will be set when last use is scheduled
                        size=tensor.nbytes,
                    )
                    scheduled_op.tensors_alloc.append(lifetime)
                    schedule.tensors[output_id] = lifetime
            
            # Mark tensor frees for inputs that won't be used again
            for inp in node.inputs:
                if tensor_last_use.get(inp) == node_id:
                    scheduled_op.tensors_free.append(inp)
                    # Update the tensor's free time
                    if inp in schedule.tensors:
                        end_time = start_time + duration
                        schedule.tensors[inp].free_time = end_time
            
            schedule.add_op(scheduled_op)
            scheduled_ops_index[node_id] = scheduled_op  # Index for quick lookup
            
            # Update device time (use lazy_max to avoid SymPy's expensive simplification)
            end_time = start_time + duration
            if device not in device_time:
                device_time[device] = end_time
            else:
                device_time[device] = lazy_max(device_time[device], end_time)
        
        # Store metadata
        schedule.metadata["strategy"] = self.strategy
        schedule.metadata["processing_mode"] = self.processing_mode
        schedule.metadata["peak_tflops"] = self.peak_tflops
        schedule.metadata["memory_bandwidth"] = self.memory_bandwidth
        schedule.metadata["network_bandwidth"] = self.network_bandwidth
        schedule.metadata["memory_capacity"] = self.memory_capacity
        
        return schedule
    
    def _compute_duration(self, node: OpNode) -> Union[float, Expr]:
        """Compute the duration of an operation using Calculon-compatible timing model.
        
        Uses dynamic efficiency factors from system config (size-dependent),
        or fixed calibration factors if provided.
        
        Processing mode:
        - "roofline": time = max(compute_time, memory_time)
        - "no_overlap": time = compute_time + memory_time
        
        IMPORTANT: For training mode, FW and BW are computed separately with their
        own efficiency lookups, then summed. This matches Calculon's behavior where
        small ops (e.g., 13 GFLOPs) get lower efficiency (60%) than large ops.
        """
        # Get FLOPs and memory access
        flops_fw = node.flops_fw or 0
        memory_fw = node.memory_fw or 0
        
        # Get agrad/wgrad separately if available, otherwise fallback to bw
        flops_agrad = node.flops_agrad or 0 if self.training else 0
        flops_wgrad = node.flops_wgrad or 0 if self.training else 0
        memory_agrad = node.memory_agrad or 0 if self.training else 0
        memory_wgrad = node.memory_wgrad or 0 if self.training else 0
        
        # Fallback: if agrad/wgrad not set, use bw / 2
        flops_bw = node.flops_bw or 0 if self.training else 0
        memory_bw = node.memory_bw or 0 if self.training else 0
        
        if flops_agrad == 0 and flops_bw > 0:
            flops_agrad = flops_bw / 2
            flops_wgrad = flops_bw / 2
        if memory_agrad == 0 and memory_bw > 0:
            memory_agrad = memory_bw / 2
            memory_wgrad = memory_bw / 2
        
        # Communication bytes: forward and backward are separate events in timeline
        comm_bytes = node.comm_bytes_fw or 0
        
        # Handle communication ops
        if node.op_type in ("AllReduce", "AllGather", "ReduceScatter"):
            comm_num = _to_python_float(comm_bytes)
            if comm_num is not None:
                num_peers = node.attrs.get("tp", 8)
                return self.sys.compute_comm_time(
                    comm_num,
                    op_type=node.op_type.lower(),
                    num_peers=num_peers,
                    tier=0
                )
            else:
                eff_net_bw = self.sys.get_network_throughput()
                return comm_bytes / eff_net_bw if eff_net_bw > 0 else 0
        
        # Check if we can do pure numeric computation
        flops_fw_num = _to_python_float(flops_fw)
        flops_bw_num = _to_python_float(flops_bw)
        memory_fw_num = _to_python_float(memory_fw)
        memory_bw_num = _to_python_float(memory_bw)
        
        all_numeric = all(x is not None for x in [flops_fw_num, memory_fw_num])
        if self.training:
            all_numeric = all_numeric and all(x is not None for x in [flops_bw_num, memory_bw_num])
        
        if all_numeric:
            # Pure numeric path - compute FW and BW separately with their own efficiency
            op_type = node.op_type
            
            if self.calibration:
                # Use fixed calibration factors
                compute_eff = self.calibration.get("compute_efficiency", 1.0)
                memory_eff = self.calibration.get("memory_efficiency", 1.0)
                eff_peak_flops = self.sys.peak_tflops * 1e12 * compute_eff
                eff_mem_bw = self.sys.memory_bandwidth * memory_eff
                
                flops_time_fw = flops_fw_num / eff_peak_flops if eff_peak_flops > 0 else 0.0
                memory_time_fw = memory_fw_num / eff_mem_bw if eff_mem_bw > 0 else 0.0
                
                if self.training:
                    flops_time_bw = flops_bw_num / eff_peak_flops if eff_peak_flops > 0 else 0.0
                    memory_time_bw = memory_bw_num / eff_mem_bw if eff_mem_bw > 0 else 0.0
                else:
                    flops_time_bw = 0.0
                    memory_time_bw = 0.0
            else:
                # Use dynamic efficiency - lookup separately for FW, AGrad, and WGrad
                # This is critical: Calculon looks up efficiency per-stage
                # A 13 GFLOPs FW op gets 60%, not 90% from combined 40 GFLOPs
                engine = "matrix" if op_type in ("Linear", "Attention") else "vector"
                
                # Get agrad/wgrad flops separately for efficiency lookup
                flops_agrad_num = _to_python_float(flops_agrad) or (flops_bw_num / 2 if flops_bw_num else 0)
                flops_wgrad_num = _to_python_float(flops_wgrad) or (flops_bw_num / 2 if flops_bw_num else 0)
                
                # FW throughput based on FW FLOPs
                compute_throughput_fw = self.sys.get_compute_throughput(flops_fw_num, engine) if flops_fw_num > 0 else self.sys.peak_tflops * 1e12
                memory_throughput_fw = self.sys.get_memory_throughput(memory_fw_num) if memory_fw_num > 0 else self.sys.memory_bandwidth
                
                flops_time_fw = flops_fw_num / compute_throughput_fw if compute_throughput_fw > 0 else 0.0
                memory_time_fw = memory_fw_num / memory_throughput_fw if memory_throughput_fw > 0 else 0.0
                
                if self.training and (flops_agrad_num > 0 or flops_wgrad_num > 0):
                    # AGrad: dX = dY @ W^T - lookup efficiency using agrad FLOPs
                    compute_throughput_agrad = self.sys.get_compute_throughput(flops_agrad_num, engine) if flops_agrad_num > 0 else self.sys.peak_tflops * 1e12
                    flops_time_agrad = flops_agrad_num / compute_throughput_agrad if compute_throughput_agrad > 0 else 0.0
                    
                    # WGrad: dW = X^T @ dY - lookup efficiency using wgrad FLOPs
                    compute_throughput_wgrad = self.sys.get_compute_throughput(flops_wgrad_num, engine) if flops_wgrad_num > 0 else self.sys.peak_tflops * 1e12
                    flops_time_wgrad = flops_wgrad_num / compute_throughput_wgrad if compute_throughput_wgrad > 0 else 0.0
                    
                    # Total BW FLOPs time = AGrad + WGrad
                    flops_time_bw = flops_time_agrad + flops_time_wgrad
                    
                    # BW Memory: use total memory_bw (agrad and wgrad share some data access like dY)
                    # This matches Calculon's approach where BW memory is computed as a whole
                    memory_throughput_bw = self.sys.get_memory_throughput(memory_bw_num) if memory_bw_num > 0 else self.sys.memory_bandwidth
                    memory_time_bw = memory_bw_num / memory_throughput_bw if memory_throughput_bw > 0 else 0.0
                    
                    # Store breakdown for analysis
                    node.attrs["flops_time_agrad"] = flops_time_agrad
                    node.attrs["flops_time_wgrad"] = flops_time_wgrad
                else:
                    flops_time_bw = 0.0
                    memory_time_bw = 0.0
            
            # Combine FW and BW times
            flops_time = flops_time_fw + flops_time_bw
            memory_time = memory_time_fw + memory_time_bw
            memory_num = (memory_fw_num or 0) + (memory_bw_num or 0)
            flops_num = (flops_fw_num or 0) + (flops_bw_num or 0)
            
            # Apply processing mode
            if self.processing_mode == "roofline":
                duration = max(flops_time, memory_time)
                # Store which bound dominated for analysis
                node.attrs["bound"] = "compute" if flops_time >= memory_time else "memory"
            else:  # no_overlap
                duration = flops_time + memory_time
                node.attrs["bound"] = "no_overlap"
            
            node.attrs["arithmetic_intensity"] = flops_num / memory_num if memory_num > 0 else float('inf')
            node.attrs["flops_time"] = flops_time
            node.attrs["memory_time"] = memory_time
            return duration
        else:
            # Symbolic case: use peak values (no dynamic efficiency for symbolic)
            # Combine FW and BW for symbolic
            flops = flops_fw + flops_bw
            memory = memory_fw + memory_bw
            
            if self.calibration:
                compute_eff = self.calibration.get("compute_efficiency", 1.0)
                memory_eff = self.calibration.get("memory_efficiency", 1.0)
            else:
                # Use typical efficiency for small ops (matches Calculon's ~60%)
                compute_eff = self.sys.get_compute_efficiency(15e9)  # ~15 GFLOPs typical
                memory_eff = self.sys.get_memory_efficiency(30e6)    # ~30 MB typical
            
            eff_peak_flops = self.sys.peak_tflops * 1e12 * compute_eff
            eff_mem_bw = self.sys.memory_bandwidth * memory_eff
            
            flops_time = flops / eff_peak_flops
            memory_time = memory / eff_mem_bw
            
            if self.processing_mode == "roofline":
                return lazy_max(flops_time, memory_time)
            else:
                return flops_time + memory_time
    
    def _compute_start_time(
        self,
        node: OpNode,
        ir: GraphIR,
        device_time: Dict[int, Union[float, Expr]],
        scheduled_ops_index: Dict[str, ScheduledOp],
    ) -> Union[float, Expr]:
        """Compute the start time of an operation based on dependencies."""
        device = node.device or 0
        
        # Start time is the max of:
        # 1. Current device time
        # 2. End time of all predecessors
        
        start = device_time.get(device, 0)
        
        # Check predecessor end times using index for O(1) lookup
        predecessors = ir.get_predecessors(node.id)
        for pred_id in predecessors:
            sched_op = scheduled_ops_index.get(pred_id)
            if sched_op:
                pred_end = sched_op.end
                start = lazy_max(start, pred_end)
        
        return start

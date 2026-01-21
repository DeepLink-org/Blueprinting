"""SchedulePass - Generate execution schedule from hierarchical GraphIR.

This pass transforms hierarchical GraphIR into hierarchical ScheduleIR by:
1. Traversing the module/block/op tree
2. Computing execution times based on system capabilities
3. Assigning operations to stages and devices
4. Tracking tensor lifetimes for memory analysis
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Union

from sympy import Expr, Symbol

from ..graph import GraphIR, OpNode, BlockNode, ModuleNode
from ..schedule import ScheduleIR, ScheduledOp, StageSchedule, DeviceSchedule, TensorLifetime
from ..system import SystemConfig
from .base import Pass
from ..symmax import sym_max, _to_float


def _to_python_float(x):
    """Convert to Python float, handling SymPy Float and int."""
    return _to_float(x)


def lazy_max(a, b):
    """Max that works with both numeric and symbolic values."""
    return sym_max(a, b)


class SchedulePass(Pass):
    """Pass to generate execution schedule from hierarchical graph.
    
    This pass computes timing information and creates a ScheduleIR
    that represents the execution-side view:
    
    Stage → Device → ScheduledOp
    
    Each op gets an event_seq for timeline ordering.
    """
    
    name = "SchedulePass"
    
    def __init__(
        self,
        system_config: Optional[Union[Dict, str, Path, SystemConfig]] = None,
        strategy: str = "sequential",
        overlap_compute_comm: bool = True,
        processing_mode: Optional[str] = None,
        calibration: Optional[Dict[str, float]] = None,
        training: bool = True,
    ):
        """Initialize SchedulePass.
        
        Args:
            system_config: Hardware system configuration
            strategy: Scheduling strategy for pipeline parallelism
            overlap_compute_comm: Whether compute and communication can overlap
            processing_mode: "roofline" or "no_overlap"
            calibration: Override dynamic efficiency with fixed factors
            training: Whether to include backward pass
        """
        self.strategy = strategy
        self.overlap_compute_comm = overlap_compute_comm
        self.training = training
        self.calibration = calibration
        
        if system_config is None:
            self.sys = SystemConfig({})
        elif isinstance(system_config, SystemConfig):
            self.sys = system_config
        elif isinstance(system_config, (str, Path)):
            self.sys = SystemConfig(system_config)
        else:
            self.sys = SystemConfig(system_config)
        
        if processing_mode is not None:
            self.processing_mode = processing_mode
        else:
            self.processing_mode = self.sys.processing_mode
        
        self.peak_tflops = self.sys.peak_tflops
        self.memory_bandwidth = self.sys.memory_bandwidth
        self.network_bandwidth = self.sys.network_bandwidth
        self.memory_capacity = self.sys.memory_capacity
    
    def run(self, ir: GraphIR) -> ScheduleIR:
        """Execute the schedule pass."""
        pp = ir.metadata.get("pp", 1)
        tp = ir.metadata.get("tp", 1)
        dp = ir.metadata.get("dp", 1)
        
        schedule = ScheduleIR(num_devices=pp * tp * dp)
        schedule.metadata.update(ir.metadata)
        
        device_time: Dict[tuple, Union[float, Expr]] = {}
        
        if ir.root:
            self._schedule_module(ir.root, schedule, device_time, ir)
        
        self._compute_metrics(schedule, ir)
        
        schedule.metadata["strategy"] = self.strategy
        schedule.metadata["processing_mode"] = self.processing_mode
        schedule.metadata["peak_tflops"] = self.peak_tflops
        schedule.metadata["memory_bandwidth"] = self.memory_bandwidth
        schedule.metadata["network_bandwidth"] = self.network_bandwidth
        schedule.metadata["memory_capacity"] = self.memory_capacity
        schedule.metadata["training"] = self.training
        
        return schedule
    
    def _schedule_module(self, module: ModuleNode, schedule: ScheduleIR,
                         device_time: Dict, ir: GraphIR) -> None:
        """Schedule all blocks in a module."""
        for block in module.children:
            self._schedule_block(block, schedule, device_time, ir, path=[module.name])
    
    def _schedule_block(self, block: BlockNode, schedule: ScheduleIR,
                        device_time: Dict, ir: GraphIR, path: List[str]) -> None:
        """Schedule all ops/sub-blocks in a block."""
        current_path = path + [block.name]
        device = block.device or 0
        stage = device
        
        for child in block.children:
            if isinstance(child, OpNode):
                self._schedule_op(child, schedule, device_time, ir, 
                                   stage=stage, device=device, path=current_path)
            elif isinstance(child, BlockNode):
                if child.device is None:
                    child.device = device
                self._schedule_block(child, schedule, device_time, ir, current_path)
    
    def _schedule_op(self, op: OpNode, schedule: ScheduleIR,
                     device_time: Dict, ir: GraphIR,
                     stage: int, device: int, path: List[str]) -> None:
        """Schedule a single operation."""
        op_path = ".".join(path + [op.name])
        
        duration = self._compute_duration(op)
        
        time_key = (stage, device)
        start_time = device_time.get(time_key, 0)
        
        stream = "comm" if op.op_type in ("AllReduce", "AllGather", "ReduceScatter") else "compute"
        
        sched_op = ScheduledOp(
            name=op.name,
            op_path=op_path,
            op_type=op.op_type,
            device=device,
            stage=stage,
            stream=stream,
            start=start_time,
            duration=duration,
            flops=op.flops,
            memory_bytes=op.memory_fw,
            comm_bytes=op.comm_bytes,
            attrs=self._collect_op_attrs(op),
        )
        
        schedule.add_op(sched_op, stage, device)
        
        for output_name in op.outputs:
            tensor = ir.get_tensor(output_name)
            if tensor:
                lifetime = TensorLifetime(
                    tensor_id=output_name,
                    alloc_time=start_time,
                    free_time=None,
                    size=tensor.nbytes,
                )
                sched_op.tensors_alloc.append(lifetime)
                schedule.tensors[output_name] = lifetime
        
        end_time = start_time + duration
        device_time[time_key] = lazy_max(device_time.get(time_key, 0), end_time)
    
    def _collect_op_attrs(self, op: OpNode) -> Dict:
        """Collect op attributes for the scheduled op."""
        attrs = op.attrs.copy()
        
        for attr in ['flops_fw', 'flops_bw', 'flops_agrad', 'flops_wgrad',
                     'weight_bytes', 'activation_bytes', 'memory_fw', 'memory_bw',
                     'comm_bytes_fw', 'comm_bytes_bw']:
            val = getattr(op, attr, None)
            if val is not None:
                attrs[attr] = val
        
        if op.shard:
            attrs['shard'] = op.shard
        
        return attrs
    
    def _compute_duration(self, op: OpNode) -> Union[float, Expr]:
        """Compute the duration of an operation."""
        flops_fw = op.flops_fw or 0
        memory_fw = op.memory_fw or 0
        
        flops_bw = op.flops_bw or 0 if self.training else 0
        memory_bw = op.memory_bw or 0 if self.training else 0
        
        comm_bytes = op.comm_bytes_fw or 0
        
        if op.op_type in ("AllReduce", "AllGather", "ReduceScatter"):
            comm_num = _to_python_float(comm_bytes)
            if comm_num is not None:
                num_peers = op.attrs.get("tp", 8)
                return self.sys.compute_comm_time(
                    comm_num,
                    op_type=op.op_type.lower(),
                    num_peers=num_peers,
                    tier=0
                )
            else:
                eff_net_bw = self.sys.get_network_throughput()
                return comm_bytes / eff_net_bw if eff_net_bw > 0 else 0
        
        flops_fw_num = _to_python_float(flops_fw)
        flops_bw_num = _to_python_float(flops_bw)
        memory_fw_num = _to_python_float(memory_fw)
        memory_bw_num = _to_python_float(memory_bw)
        
        all_numeric = flops_fw_num is not None and memory_fw_num is not None
        if self.training:
            all_numeric = all_numeric and flops_bw_num is not None and memory_bw_num is not None
        
        if all_numeric:
            return self._compute_duration_numeric(
                op, flops_fw_num, flops_bw_num, memory_fw_num, memory_bw_num
            )
        else:
            return self._compute_duration_symbolic(flops_fw, flops_bw, memory_fw, memory_bw)
    
    def _compute_duration_numeric(self, op: OpNode,
                                   flops_fw: float, flops_bw: float,
                                   memory_fw: float, memory_bw: float) -> float:
        """Compute duration for numeric values."""
        op_type = op.op_type
        
        flops_agrad_num = _to_python_float(op.flops_agrad) or (flops_bw / 2 if flops_bw else 0)
        flops_wgrad_num = _to_python_float(op.flops_wgrad) or (flops_bw / 2 if flops_bw else 0)
        
        if self.calibration:
            compute_eff = self.calibration.get("compute_efficiency", 1.0)
            memory_eff = self.calibration.get("memory_efficiency", 1.0)
            eff_peak_flops = self.sys.peak_tflops * 1e12 * compute_eff
            eff_mem_bw = self.sys.memory_bandwidth * memory_eff
            
            flops_time_fw = flops_fw / eff_peak_flops if eff_peak_flops > 0 else 0.0
            memory_time_fw = memory_fw / eff_mem_bw if eff_mem_bw > 0 else 0.0
            
            if self.training:
                flops_time_bw = flops_bw / eff_peak_flops if eff_peak_flops > 0 else 0.0
                memory_time_bw = memory_bw / eff_mem_bw if eff_mem_bw > 0 else 0.0
            else:
                flops_time_bw = 0.0
                memory_time_bw = 0.0
        else:
            engine = "matrix" if op_type in ("Linear", "Attention") else "vector"
            
            compute_throughput_fw = self.sys.get_compute_throughput(flops_fw, engine) if flops_fw > 0 else self.sys.peak_tflops * 1e12
            memory_throughput_fw = self.sys.get_memory_throughput(memory_fw) if memory_fw > 0 else self.sys.memory_bandwidth
            
            flops_time_fw = flops_fw / compute_throughput_fw if compute_throughput_fw > 0 else 0.0
            memory_time_fw = memory_fw / memory_throughput_fw if memory_throughput_fw > 0 else 0.0
            
            if self.training and (flops_agrad_num > 0 or flops_wgrad_num > 0):
                compute_throughput_agrad = self.sys.get_compute_throughput(flops_agrad_num, engine) if flops_agrad_num > 0 else self.sys.peak_tflops * 1e12
                flops_time_agrad = flops_agrad_num / compute_throughput_agrad if compute_throughput_agrad > 0 else 0.0
                
                compute_throughput_wgrad = self.sys.get_compute_throughput(flops_wgrad_num, engine) if flops_wgrad_num > 0 else self.sys.peak_tflops * 1e12
                flops_time_wgrad = flops_wgrad_num / compute_throughput_wgrad if compute_throughput_wgrad > 0 else 0.0
                
                flops_time_bw = flops_time_agrad + flops_time_wgrad
                
                memory_throughput_bw = self.sys.get_memory_throughput(memory_bw) if memory_bw > 0 else self.sys.memory_bandwidth
                memory_time_bw = memory_bw / memory_throughput_bw if memory_throughput_bw > 0 else 0.0
                
                op.attrs["flops_time_agrad"] = flops_time_agrad
                op.attrs["flops_time_wgrad"] = flops_time_wgrad
            else:
                flops_time_bw = 0.0
                memory_time_bw = 0.0
        
        flops_time = flops_time_fw + flops_time_bw
        memory_time = memory_time_fw + memory_time_bw
        
        if self.processing_mode == "roofline":
            duration = max(flops_time, memory_time)
            op.attrs["bound"] = "compute" if flops_time >= memory_time else "memory"
        else:
            duration = flops_time + memory_time
            op.attrs["bound"] = "no_overlap"
        
        total_flops = flops_fw + flops_bw
        total_mem = memory_fw + memory_bw
        op.attrs["arithmetic_intensity"] = total_flops / total_mem if total_mem > 0 else float('inf')
        op.attrs["flops_time"] = flops_time
        op.attrs["memory_time"] = memory_time
        
        return duration
    
    def _compute_duration_symbolic(self, flops_fw, flops_bw, memory_fw, memory_bw):
        """Compute duration for symbolic values."""
        flops = flops_fw + flops_bw
        memory = memory_fw + memory_bw
        
        if self.calibration:
            compute_eff = self.calibration.get("compute_efficiency", 1.0)
            memory_eff = self.calibration.get("memory_efficiency", 1.0)
        else:
            compute_eff = self.sys.get_compute_efficiency(15e9)
            memory_eff = self.sys.get_memory_efficiency(30e6)
        
        eff_peak_flops = self.sys.peak_tflops * 1e12 * compute_eff
        eff_mem_bw = self.sys.memory_bandwidth * memory_eff
        
        flops_time = flops / eff_peak_flops
        memory_time = memory / eff_mem_bw
        
        if self.processing_mode == "roofline":
            return lazy_max(flops_time, memory_time)
        else:
            return flops_time + memory_time
    
    def _compute_metrics(self, schedule: ScheduleIR, ir: GraphIR) -> None:
        """Compute memory and time metrics for comparison."""
        device_weight_bytes = 0
        device_activation_bytes = 0
        device_flops = 0
        
        layer_activations = {}
        layer_activations_sum = {}
        layer_weights = {}
        layer_flops = {}
        layer_times = {}
        other_activations = 0
        
        per_mb_compute_time = 0
        per_mb_comm_time = 0
        
        for op in schedule.iter_ops():
            if op.stage != 0:
                continue
            
            attrs = op.attrs
            
            weight_bytes = attrs.get("weight_bytes", 0)
            if isinstance(weight_bytes, Expr):
                device_weight_bytes = device_weight_bytes + weight_bytes
            else:
                device_weight_bytes += weight_bytes or 0
            
            activation_bytes = attrs.get("activation_bytes", 0)
            if isinstance(activation_bytes, Expr):
                activation_bytes = 0
            
            match = re.search(r"layer(\d+)", op.op_path or op.name)
            if match:
                layer_idx = int(match.group(1))
                if layer_idx not in layer_activations:
                    layer_activations[layer_idx] = 0
                    layer_activations_sum[layer_idx] = 0
                    layer_weights[layer_idx] = 0
                    layer_flops[layer_idx] = {"fw": 0, "agrad": 0, "wgrad": 0}
                    layer_times[layer_idx] = {"compute": 0, "comm": 0}
                
                layer_activations_sum[layer_idx] += activation_bytes or 0
                layer_activations[layer_idx] = max(layer_activations[layer_idx], activation_bytes or 0)
                
                weight_val = _to_python_float(weight_bytes) or 0
                layer_weights[layer_idx] += weight_val
                
                flops_fw = _to_python_float(attrs.get("flops_fw", 0)) or 0
                flops_agrad = _to_python_float(attrs.get("flops_agrad", 0)) or 0
                flops_wgrad = _to_python_float(attrs.get("flops_wgrad", 0)) or 0
                flops_bw = _to_python_float(attrs.get("flops_bw", 0)) or 0
                
                if flops_agrad == 0 and flops_wgrad == 0 and flops_bw > 0:
                    flops_agrad = flops_bw / 2
                    flops_wgrad = flops_bw / 2
                
                layer_flops[layer_idx]["fw"] += flops_fw
                layer_flops[layer_idx]["agrad"] += flops_agrad
                layer_flops[layer_idx]["wgrad"] += flops_wgrad
                
                duration = op.duration
                if isinstance(duration, (int, float)):
                    if op.stream in ("comm", "nccl"):
                        layer_times[layer_idx]["comm"] += duration
                    else:
                        layer_times[layer_idx]["compute"] += duration
            else:
                other_activations += activation_bytes or 0
            
            flops_fw = attrs.get("flops_fw", 0)
            flops_bw = attrs.get("flops_bw", 0)
            if isinstance(flops_fw, Expr):
                device_flops = device_flops + flops_fw
            else:
                device_flops += flops_fw or 0
            if isinstance(flops_bw, Expr):
                device_flops = device_flops + flops_bw
            else:
                device_flops += flops_bw or 0
            
            if op.op_type not in ("OptimizerStep", "RecomputeStep"):
                duration = op.duration
                if isinstance(duration, (int, float)):
                    if op.stream in ("comm", "nccl"):
                        per_mb_comm_time += duration
                    else:
                        per_mb_compute_time += duration
        
        if layer_activations:
            first_layer_idx = min(layer_activations.keys())
            device_activation_bytes = layer_activations[first_layer_idx] + other_activations
        else:
            first_layer_idx = None
            device_activation_bytes = other_activations
        
        first_layer_weight_bytes = layer_weights.get(first_layer_idx, 0) if first_layer_idx is not None else 0
        first_layer_activation_bytes = layer_activations.get(first_layer_idx, 0) if first_layer_idx is not None else 0
        
        layer_fw_time = 0
        layer_agrad_time = 0
        layer_wgrad_time = 0
        layer_compute_time = 0
        layer_comm_time = 0
        
        if first_layer_idx is not None:
            layer_compute_time = layer_times[first_layer_idx]["compute"]
            layer_comm_time = layer_times[first_layer_idx]["comm"]
            flops_fw = layer_flops[first_layer_idx]["fw"]
            flops_agrad = layer_flops[first_layer_idx]["agrad"]
            flops_wgrad = layer_flops[first_layer_idx]["wgrad"]
            flops_total = flops_fw + flops_agrad + flops_wgrad
            if flops_total > 0:
                layer_fw_time = layer_compute_time * (flops_fw / flops_total)
                layer_agrad_time = layer_compute_time * (flops_agrad / flops_total)
                layer_wgrad_time = layer_compute_time * (flops_wgrad / flops_total)
        
        schedule.metadata["total_weight_bytes"] = device_weight_bytes
        schedule.metadata["total_activation_bytes"] = device_activation_bytes
        schedule.metadata["total_flops"] = device_flops
        
        schedule.metadata["comparison_base"] = {
            "schema": "comparison_v1",
            "units": {"memory": "bytes", "time": "seconds"},
            "per_gpu": {
                "weights_fp16_bytes": device_weight_bytes,
                "activation_block_bytes": device_activation_bytes,
                "total_flops": device_flops,
            },
            "per_mb": {
                "compute_time": per_mb_compute_time,
                "comm_time": per_mb_comm_time,
                "total_time": per_mb_compute_time + per_mb_comm_time,
            },
            "activation_basis": "single_layer_peak",
            "layer": {
                "index": first_layer_idx,
                "weights_bytes": first_layer_weight_bytes,
                "activations_bytes": first_layer_activation_bytes,
                "time": {
                    "forward": layer_fw_time,
                    "agrad": layer_agrad_time,
                    "wgrad": layer_wgrad_time,
                    "compute": layer_compute_time,
                    "comm": layer_comm_time,
                    "total": layer_compute_time + layer_comm_time,
                },
            },
            "layer_activations": layer_activations,
            "layer_activations_sum": layer_activations_sum,
            "layer_weights": layer_weights,
            "layer_flops": layer_flops,
        }

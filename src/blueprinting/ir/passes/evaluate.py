"""EvaluatePass - Evaluate ScheduleIR or TimelineIR to produce SimulationResult.

This pass substitutes symbolic values and computes final metrics
like peak memory, end-to-end time, and MFU.
"""

from typing import Any, Dict, Optional, Union

from sympy import Expr, Symbol

from ..schedule import ScheduleIR
from ..timeline import TimelineIR
from ..result import SimulationResult, MemoryBreakdown, TimeBreakdown
from ..symmax import SymMax, eval_lazy, _to_float
from .base import Pass


class EvaluatePass(Pass):
    """Pass to evaluate schedule/timeline and produce simulation results.
    
    This pass:
    1. Substitutes symbolic values with concrete numbers
    2. Computes peak memory usage (precise with TimelineIR)
    3. Computes end-to-end execution time
    4. Calculates derived metrics (MFU, throughput)
    
    Supports both ScheduleIR and TimelineIR as input:
    - ScheduleIR: Basic evaluation with conservative estimates
    - TimelineIR: Precise memory tracking and overlap analysis
    """
    
    def __init__(
        self,
        subs: Optional[Dict[Union[str, Symbol], Union[int, float]]] = None,
        training: bool = True,
        optimizer: str = "adam",  # "adam", "sgd", etc.
    ):
        """Initialize EvaluatePass.
        
        Args:
            subs: Symbol substitutions (e.g., {"B": 4, "S": 2048, "H": 4096})
            training: Whether this is a training workload
            optimizer: Optimizer type (affects memory for optimizer states)
        """
        self.subs = subs or {}
        self.training = training
        self.optimizer = optimizer
        
        # Normalize subs to use Symbol keys
        self._subs_dict = {}
        for k, v in self.subs.items():
            if isinstance(k, str):
                self._subs_dict[Symbol(k)] = v
            else:
                self._subs_dict[k] = v
    
    def run(self, ir: Union[ScheduleIR, TimelineIR]) -> SimulationResult:
        """Execute the evaluate pass.
        
        Args:
            ir: Either ScheduleIR or TimelineIR
            
        Returns:
            SimulationResult with computed metrics
        """
        if isinstance(ir, TimelineIR):
            return self._run_timeline(ir)
        else:
            return self._run_schedule(ir)
    
    def _run_timeline(self, timeline: TimelineIR) -> SimulationResult:
        """Evaluate TimelineIR for precise metrics."""
        # Compute makespan (end-to-end time)
        e2e_time = self._eval_expr(timeline.makespan())
        
        # Compute peak memory (precise from event stream)
        peak_memory = self._eval_expr(timeline.peak_memory(device=0))
        
        # Compute time breakdown from timeline
        compute_time = self._eval_expr(timeline.compute_time(device=0))
        comm_time = self._eval_expr(timeline.comm_time(device=0))
        bubble_time = max(0, e2e_time - compute_time - comm_time)
        
        time_breakdown = TimeBreakdown(
            forward=compute_time * 0.4,  # Estimate: 40% forward
            backward=compute_time * 0.6,  # Estimate: 60% backward
            optimizer=0,
            communication=comm_time,
            bubble=bubble_time,
        )
        
        # Memory breakdown from metadata (computed by SchedulePass)
        # total_weight_bytes is FP16 (from WorkloadPass with dtype_bytes=2)
        total_weight_bytes_fp16 = self._eval_expr(timeline.metadata.get("total_weight_bytes", 0))
        total_activation_bytes = self._eval_expr(timeline.metadata.get("total_activation_bytes", 0))
        is_training = timeline.metadata.get("training", self.training)
        
        # Convert to FP32 master weights (to match Calculon's weight_space)
        total_weight_bytes_fp32 = total_weight_bytes_fp16 * 2
        
        # Calculate gradients (FP16, same as compute weights)
        gradients = total_weight_bytes_fp16 if is_training else 0
        
        # Calculate optimizer states
        optimizer_states = 0
        if is_training:
            if self.optimizer == "adam":
                # Adam optimizer states: m (FP32) + v (FP32) = 2 * weight_params * 4 bytes
                # weight_params = total_weight_bytes_fp16 / 2
                # optimizer_states = weight_params * 4 * 2 = total_weight_bytes_fp16 * 4
                optimizer_states = total_weight_bytes_fp16 * 4  # m + v in FP32
            elif self.optimizer == "sgd":
                optimizer_states = 0
        
        memory_breakdown = MemoryBreakdown(
            weights=total_weight_bytes_fp32,  # FP32 master weights (matches Calculon)
            activations=total_activation_bytes if total_activation_bytes > 0 else peak_memory,
            gradients=gradients,  # FP16 gradients
            optimizer_states=optimizer_states,  # m + v in FP32
        )
        
        # Compute total FLOPs from metadata
        total_flops = timeline.metadata.get("total_flops", 0)
        achieved_flops = total_flops / e2e_time if e2e_time > 0 else 0
        
        # Overlap analysis
        overlap_ratio = timeline.overlap_ratio(device=0)
        
        result = SimulationResult(
            peak_memory=peak_memory,
            e2e_time=e2e_time,
            memory_breakdown=memory_breakdown,
            time_breakdown=time_breakdown,
            total_flops=total_flops,
            achieved_flops=achieved_flops,
            config={
                "peak_tflops": timeline.metadata.get("peak_tflops", 0),
                "memory_capacity": timeline.metadata.get("memory_capacity", 0),
                "training": self.training,
                "optimizer": self.optimizer,
                "overlap_ratio": overlap_ratio,
                "subs": {str(k): v for k, v in self._subs_dict.items()},
            },
        )
        
        comparison_metrics = self._build_comparison_metrics(
            timeline,
            memory_breakdown=memory_breakdown,
            time_breakdown=time_breakdown,
            peak_memory=peak_memory,
            e2e_time=e2e_time,
        )
        timeline.metadata["comparison_metrics"] = comparison_metrics
        result.config["comparison_metrics"] = comparison_metrics
        
        # Add warnings
        capacity = timeline.metadata.get("memory_capacity", float("inf"))
        if peak_memory > capacity:
            result.warnings.append(
                f"Peak memory ({peak_memory/1e9:.2f} GB) exceeds capacity ({capacity/1e9:.2f} GB)"
            )
        
        return result
    
    def _run_schedule(self, schedule: ScheduleIR) -> SimulationResult:
        """Evaluate ScheduleIR (original implementation)."""
        # Compute makespan (end-to-end time)
        e2e_time = self._eval_expr(schedule.makespan())
        
        # Compute peak memory
        peak_memory = self._eval_expr(schedule.peak_memory(device=0))
        
        # Compute memory breakdown
        memory_breakdown = self._compute_memory_breakdown(schedule)
        
        # Compute time breakdown
        time_breakdown = self._compute_time_breakdown(schedule)
        
        # Compute total FLOPs
        total_flops = self._compute_total_flops(schedule)
        
        # Compute achieved FLOPs (FLOPs / time)
        achieved_flops = total_flops / e2e_time if e2e_time > 0 else 0
        
        # Build result
        result = SimulationResult(
            peak_memory=peak_memory,
            e2e_time=e2e_time,
            memory_breakdown=memory_breakdown,
            time_breakdown=time_breakdown,
            total_flops=total_flops,
            achieved_flops=achieved_flops,
            config={
                "peak_tflops": schedule.metadata.get("peak_tflops", 0),
                "memory_capacity": schedule.metadata.get("memory_capacity", 0),
                "training": self.training,
                "optimizer": self.optimizer,
                "subs": {str(k): v for k, v in self._subs_dict.items()},
            },
        )
        
        comparison_metrics = self._build_comparison_metrics(
            schedule,
            memory_breakdown=memory_breakdown,
            time_breakdown=time_breakdown,
            peak_memory=peak_memory,
            e2e_time=e2e_time,
        )
        schedule.metadata["comparison_metrics"] = comparison_metrics
        result.config["comparison_metrics"] = comparison_metrics
        
        # Add warnings if memory exceeds capacity
        capacity = schedule.metadata.get("memory_capacity", float("inf"))
        if peak_memory > capacity:
            result.warnings.append(
                f"Peak memory ({peak_memory/1e9:.2f} GB) exceeds capacity ({capacity/1e9:.2f} GB)"
            )
        
        return result
    
    def _eval_expr(self, expr: Union[int, float, Expr, SymMax]) -> float:
        """Evaluate a symbolic expression with substitutions.
        
        Handles:
        - Python int/float: return as-is
        - SymMax and other lazy expressions: call eval() with substitutions
        - SymPy Expr: substitute and convert to float
        """
        if isinstance(expr, (int, float)):
            return float(expr)
        
        # Handle SymMax and other lazy expressions
        result = eval_lazy(expr, self._subs_dict)
        
        # Try to convert result to float
        num = _to_float(result)
        if num is not None:
            return num
        
        # Still symbolic after substitution - return 0 as fallback
        return 0.0
    
    def _build_comparison_metrics(
        self,
        ir: Union[ScheduleIR, TimelineIR],
        memory_breakdown: Optional[MemoryBreakdown],
        time_breakdown: Optional[TimeBreakdown],
        peak_memory: float,
        e2e_time: float,
    ) -> Dict[str, Any]:
        """Build a normalized metrics dict for IR/Calculon comparisons."""
        base = ir.metadata.get("comparison_base", {})
        per_gpu_base = base.get("per_gpu", {})
        
        weights_fp16 = self._eval_expr(
            per_gpu_base.get("weights_fp16_bytes", ir.metadata.get("total_weight_bytes", 0))
        )
        weights_fp32 = weights_fp16 * 2
        weights_basis = "fp16"
        
        activation_block = self._eval_expr(per_gpu_base.get("activation_block_bytes", 0))
        activation_peak = peak_memory
        
        gradients_raw = weights_fp16 if ir.metadata.get("training", self.training) else 0
        optimizer_raw = memory_breakdown.optimizer_states if memory_breakdown else 0
        
        optimizer_sharding = ir.metadata.get("optimizer_sharding", False) or ir.metadata.get("zero", 0) > 0
        gradients = 0
        optimizer_states = 0 if optimizer_sharding else optimizer_raw
        
        activations = activation_block if activation_block > 0 else (
            memory_breakdown.activations if memory_breakdown else activation_peak
        )
        
        weights_display = weights_fp16 if weights_basis == "fp16" else weights_fp32
        total_memory = weights_display + activations + optimizer_states
        
        forward = time_breakdown.forward if time_breakdown else 0
        backward = time_breakdown.backward if time_breakdown else 0
        optimizer_time = time_breakdown.optimizer if time_breakdown else 0
        comm = time_breakdown.communication if time_breakdown else 0
        bubble = time_breakdown.bubble if time_breakdown else 0
        
        # Prefer optimizer metadata if available (more explicit than time_breakdown)
        if "optimizer_time" in ir.metadata:
            optimizer_time = self._eval_expr(ir.metadata.get("optimizer_time", optimizer_time))
        
        recompute_time = 0
        if "recompute_time" in ir.metadata:
            recompute_time = self._eval_expr(ir.metadata.get("recompute_time", 0))
        
        per_mb = base.get("per_mb", {})
        per_mb_compute = self._eval_expr(per_mb.get("compute_time", 0))
        per_mb_comm = self._eval_expr(per_mb.get("comm_time", 0))
        per_mb_total = self._eval_expr(per_mb.get("total_time", per_mb_compute + per_mb_comm))
        num_microbatches = ir.metadata.get("num_microbatches", 1)
        pp = ir.metadata.get("pp", getattr(ir, "num_stages", 1))
        
        if num_microbatches and per_mb_total:
            pipeline_span = num_microbatches + max(pp - 1, 0)
            bubble = per_mb_total * max(pp - 1, 0)
            forward = 0
            backward = 0
            comm = 0
            e2e_time = per_mb_total * pipeline_span + optimizer_time + recompute_time
        
        comparison_metrics = {
            "schema": "comparison_v1",
            "units": {
                "memory": "bytes",
                "time": "seconds",
            },
            "basis": {
                "weights": weights_basis,
                "activations": base.get("activation_basis", "unknown"),
                "gradients": "fp16",
                "optimizer_states": "fp32",
                "optimizer_sharding": optimizer_sharding,
            },
            "per_gpu": {
                "memory": {
                    "weights_fp16_bytes": weights_fp16,
                    "weights_fp32_bytes": weights_fp32,
                    "activations_bytes": activations,
                    "activations_block_bytes": activation_block,
                    "activations_peak_bytes": activation_peak,
                    "gradients_bytes": gradients,
                    "optimizer_bytes": optimizer_states,
                    "gradients_raw_bytes": gradients_raw,
                    "optimizer_raw_bytes": optimizer_raw,
                    "total_bytes": total_memory,
                },
                "time": {
                    "iteration_time": e2e_time,
                    "forward_time": forward,
                    "backward_time": backward,
                    "optimizer_time": optimizer_time,
                    "communication_time": comm,
                    "bubble_time": bubble,
                },
            },
            "block": {
                "memory": {
                    "weights_bytes": base.get("layer", {}).get("weights_bytes", 0),
                    "activations_bytes": base.get("layer", {}).get("activations_bytes", 0),
                    # Per-layer optimizer (before sharding, for comparison with Calculon)
                    "optimizer_bytes": base.get("layer", {}).get("optimizer_bytes", 0),
                },
                "time": {
                    "forward_time": base.get("layer", {}).get("time", {}).get("forward", 0),
                    "agrad_time": base.get("layer", {}).get("time", {}).get("agrad", 0),
                    "wgrad_time": base.get("layer", {}).get("time", {}).get("wgrad", 0),
                    "compute_time": base.get("layer", {}).get("time", {}).get("compute", 0),
                    "comm_time": base.get("layer", {}).get("time", {}).get("comm", 0),
                    "total_time": base.get("layer", {}).get("time", {}).get("total", 0),
                },
            },
        }
        
        return comparison_metrics
    
    def _compute_memory_breakdown(self, schedule: ScheduleIR) -> MemoryBreakdown:
        """Compute detailed memory breakdown."""
        weights = 0
        activations = 0
        gradients = 0
        optimizer_states = 0
        
        # Sum up memory from scheduled ops
        for op in schedule.ops:
            # Weight memory
            weight_bytes = op.attrs.get("weight_bytes", 0)
            if isinstance(weight_bytes, Expr):
                weight_bytes = self._eval_expr(weight_bytes)
            weights += weight_bytes
            
            # Activation memory from tensor allocations
            for tensor in op.tensors_alloc:
                size = self._eval_expr(tensor.size) if isinstance(tensor.size, Expr) else tensor.size
                activations += size
        
        # For training, add gradient and optimizer state memory
        if self.training:
            gradients = weights  # Same size as weights
            
            if self.optimizer == "adam":
                # Adam: 2 moment buffers per parameter (in FP32)
                # Plus master copy of weights if using mixed precision
                optimizer_states = weights * 2 * 2  # m, v in FP32 (2x)
                optimizer_states += weights * 2  # Master weights in FP32
            elif self.optimizer == "sgd":
                optimizer_states = 0  # SGD has no state
        
        return MemoryBreakdown(
            weights=weights,
            activations=activations,
            gradients=gradients,
            optimizer_states=optimizer_states,
        )
    
    def _compute_time_breakdown(self, schedule: ScheduleIR) -> TimeBreakdown:
        """Compute detailed time breakdown."""
        forward = 0
        backward = 0
        optimizer_time = 0
        communication = 0
        
        for op in schedule.ops:
            duration = self._eval_expr(op.duration)
            
            if op.stream in ("comm", "nccl"):
                communication += duration
            elif "backward" in op.op_id.lower() or "_bw" in op.op_id.lower():
                backward += duration
            elif "optim" in op.op_id.lower():
                optimizer_time += duration
            else:
                forward += duration
        
        # Compute bubble time for pipeline parallelism
        bubble = 0
        if schedule.num_stages > 1:
            # Simplified bubble calculation
            # Real bubble depends on schedule strategy (1F1B, etc.)
            total_compute = forward + backward
            num_microbatches = self._subs_dict.get(Symbol("num_microbatches"), 1)
            if isinstance(num_microbatches, Expr):
                num_microbatches = self._eval_expr(num_microbatches)
            
            # 1F1B bubble ratio: (PP - 1) / num_microbatches
            if num_microbatches > 0:
                bubble_ratio = (schedule.num_stages - 1) / num_microbatches
                bubble = total_compute * bubble_ratio
        
        return TimeBreakdown(
            forward=forward,
            backward=backward,
            optimizer=optimizer_time,
            communication=communication,
            bubble=bubble,
        )
    
    def _compute_total_flops(self, schedule: ScheduleIR) -> float:
        """Compute total FLOPs from schedule."""
        total = 0
        
        for op in schedule.ops:
            # Get FLOPs from attrs
            flops_fw = op.attrs.get("flops_fw", 0)
            flops_bw = op.attrs.get("flops_bw", 0)
            
            if isinstance(flops_fw, Expr):
                flops_fw = self._eval_expr(flops_fw)
            if isinstance(flops_bw, Expr):
                flops_bw = self._eval_expr(flops_bw)
            
            total += flops_fw + flops_bw
        
        return total

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
        
        # Memory breakdown from timeline (simplified)
        memory_breakdown = MemoryBreakdown(
            weights=0,
            activations=peak_memory,
            gradients=0,
            optimizer_states=0,
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

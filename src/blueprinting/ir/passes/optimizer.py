"""OptimizerPass - Add optimizer step time modeling.

This pass adds optimizer operations to the schedule, modeling the time
for optimizer steps (e.g., Adam update rule) and gradient checkpointing
recomputation.

Key components modeled:
1. Optimizer step: For each parameter, compute m, v updates and weight update
   - Time dominated by memory bandwidth (read m, v, g, w; write m, v, w)
2. Gradient checkpointing: Recompute activations during backward pass
   - Time ≈ forward time for checkpointed layers
"""

from dataclasses import dataclass
from typing import Dict, Optional, Union

from sympy import Expr, Symbol

from ..schedule import ScheduleIR, ScheduledOp
from .base import Pass


@dataclass
class OptimizerConfig:
    """Configuration for optimizer modeling.
    
    Attributes:
        optimizer_type: Type of optimizer ('adam', 'sgd', 'adamw')
        dtype_bytes: Bytes per element for model parameters
        master_weights: Whether using FP32 master weights
        gradient_checkpointing: Whether gradient checkpointing is enabled
        recompute_mode: Recomputation mode ('full', 'attn_only', 'none')
            - 'full': Recompute all activations (checkpoint_ratio = 1.0)
            - 'attn_only': Only recompute attention (checkpoint_ratio ≈ 0.2)
            - 'none': No recomputation (checkpoint_ratio = 0.0)
        checkpoint_ratio: Fraction of forward time to recompute
        num_microbatches: Number of microbatches per iteration
    """
    optimizer_type: str = "adam"
    dtype_bytes: int = 2  # FP16
    master_weights: bool = True
    gradient_checkpointing: bool = False
    recompute_mode: str = "attn_only"  # 'full', 'attn_only', 'none'
    checkpoint_ratio: float = 0.2  # For attn_only mode
    num_microbatches: int = 1  # Number of microbatches
    
    def __post_init__(self):
        """Set checkpoint_ratio based on recompute_mode if not explicitly set."""
        if self.gradient_checkpointing:
            if self.recompute_mode == "full":
                self.checkpoint_ratio = 1.0
            elif self.recompute_mode == "attn_only":
                # Attention is roughly 20% of forward compute time
                self.checkpoint_ratio = 0.2
            elif self.recompute_mode == "none":
                self.checkpoint_ratio = 0.0
    
    def get_optimizer_memory_multiplier(self) -> float:
        """Get memory multiplier for optimizer states.
        
        Returns bytes of optimizer state per byte of model weight.
        """
        if self.optimizer_type in ("adam", "adamw"):
            # Adam: m (momentum), v (variance), master weights
            # m, v in same dtype as computation (FP16 or FP32)
            # master weights in FP32
            if self.master_weights:
                # FP32 master + FP32 m + FP32 v = 12 bytes per param
                return 12 / self.dtype_bytes
            else:
                # m + v in FP16 = 4 bytes per param
                return 4 / self.dtype_bytes
        elif self.optimizer_type == "sgd":
            # SGD with momentum: just m
            if self.master_weights:
                return 8 / self.dtype_bytes  # FP32 master + FP32 m
            else:
                return 2 / self.dtype_bytes  # FP16 m
        else:
            return 0
    
    def get_optimizer_flops_per_param(self) -> int:
        """Get FLOPs per parameter for optimizer update."""
        if self.optimizer_type in ("adam", "adamw"):
            # Adam update: ~10 ops per param
            # m = beta1 * m + (1-beta1) * g   : 3 ops
            # v = beta2 * v + (1-beta2) * g^2 : 4 ops
            # w = w - lr * m / (sqrt(v) + eps) : 3 ops
            return 10
        elif self.optimizer_type == "sgd":
            # SGD with momentum: ~3 ops
            return 3
        else:
            return 0
    
    def get_optimizer_memory_access_per_param(self) -> int:
        """Get memory access bytes per parameter for optimizer update.
        
        Following Calculon's model: memory access = optimizer state size.
        This is a simplified model that matches real-world behavior because:
        1. Optimizer update is streaming (element-wise)
        2. Memory access patterns allow efficient caching
        """
        if self.optimizer_type in ("adam", "adamw"):
            # Optimizer state: m (FP32) + v (FP32) = 8 bytes per param
            # Plus FP32 master weights if used: 4 bytes
            if self.master_weights:
                return 8 + 4  # 12 bytes (m + v + master_weight)
            else:
                return 8  # 8 bytes (m + v)
        elif self.optimizer_type == "sgd":
            # SGD with momentum: m only
            if self.master_weights:
                return 4 + 4  # 8 bytes (m + master_weight)
            else:
                return 4  # 4 bytes (m)
        else:
            return 0


class OptimizerPass(Pass):
    """Pass to add optimizer step time to schedule.
    
    This pass modifies the ScheduleIR to account for:
    1. Optimizer update time (dominated by memory bandwidth)
    2. Gradient checkpointing recomputation time
    """
    
    def __init__(
        self,
        optimizer_config: Optional[OptimizerConfig] = None,
        system_config: Optional[Dict] = None,
    ):
        """Initialize OptimizerPass.
        
        Args:
            optimizer_config: Optimizer configuration
            system_config: Hardware system configuration
        """
        self.config = optimizer_config or OptimizerConfig()
        self.system_config = system_config or {}
        
        # System parameters
        self.memory_bandwidth = self.system_config.get("memory_bandwidth_gbps", 3072) * 1e9
        self.peak_tflops = self.system_config.get("peak_tflops", 1000) * 1e12
    
    def run(self, ir: ScheduleIR) -> ScheduleIR:
        """Execute the optimizer pass.
        
        Key insight: 
        1. Optimizer runs ONCE after all microbatches complete.
           It processes each layer's weights exactly once.
        2. Recompute happens during EVERY backward pass (per microbatch)
           when gradient checkpointing is enabled.
        
        Correct calculation:
        - optimizer_time = sum(unique_layer_weight_bytes) × cost_per_byte  (once)
        - recompute_time = forward_time_per_layer × num_layers × num_microbatches
        """
        # Collect unique weight bytes per layer (by op_id without _FW/_BW suffix)
        unique_layer_weights: Dict[str, float] = {}
        forward_time_per_mb = 0  # Forward time for ONE microbatch
        total_ops_time = 0
        max_end_time = 0
        num_microbatches = 1
        
        # Count microbatches from op_ids
        import re
        mb_pattern = re.compile(r'_mb(\d+)')
        seen_mbs = set()
        
        # ScheduleIR.ops is a list of ScheduledOp
        for op in ir.ops:
            # Count microbatches
            mb_match = mb_pattern.search(op.op_id)
            if mb_match:
                seen_mbs.add(int(mb_match.group(1)))
            
            # Extract layer name (remove _FW, _BW, _mb0, _mb1 suffixes)
            op_id = op.op_id
            base_layer = op_id
            for suffix in ("_FW", "_BW", "_wgrad", "_agrad"):
                if suffix in base_layer:
                    base_layer = base_layer.split(suffix)[0]
                    break
            base_layer = mb_pattern.sub('', base_layer)
            
            # Only count weight bytes (avoid double counting)
            weight_bytes = op.attrs.get("weight_bytes", 0)
            if isinstance(weight_bytes, (int, float)) and weight_bytes > 0:
                if base_layer not in unique_layer_weights:
                    unique_layer_weights[base_layer] = weight_bytes
                else:
                    unique_layer_weights[base_layer] = max(unique_layer_weights[base_layer], weight_bytes)
            
            # Track compute time for all non-communication ops
            if op.op_type not in ("AllReduce", "AllGather", "ReduceScatter", "OptimizerStep", "RecomputeStep"):
                duration = op.duration
                if isinstance(duration, (int, float)):
                    total_ops_time += duration
            
            # Track max end time
            end = op.end
            if isinstance(end, (int, float)):
                max_end_time = max(max_end_time, end)
        
        # Update microbatch count
        # Priority: config > metadata > detected from op_ids
        if self.config.num_microbatches > 1:
            num_microbatches = self.config.num_microbatches
        elif "num_microbatches" in ir.metadata:
            num_microbatches = ir.metadata["num_microbatches"]
        elif seen_mbs:
            num_microbatches = len(seen_mbs)
        # else: keep default of 1
        
        # Estimate forward time from total compute time
        # Note: ScheduleIR represents a SINGLE microbatch execution
        # In training: total_time ≈ FW + 2×FW (BW) = 3×FW
        # So FW ≈ total_time / 3
        # total_ops_time is already per-microbatch
        total_compute_per_mb = total_ops_time
        
        # Forward time ≈ 1/3 of total (since BW ≈ 2×FW)
        forward_time_per_mb = total_compute_per_mb / 3.0
        
        # Total unique weight bytes (one per layer)
        total_weight_bytes = sum(unique_layer_weights.values())
        
        # Compute optimizer time (runs once)
        optimizer_time = self._compute_optimizer_time(total_weight_bytes)
        
        # Compute recomputation time (if gradient checkpointing enabled)
        # Recompute happens during every backward pass, for each microbatch
        recompute_time = 0
        recompute_time_per_mb = 0
        if self.config.gradient_checkpointing:
            # Recompute time ≈ forward time × checkpoint_ratio
            recompute_time_per_mb = forward_time_per_mb * self.config.checkpoint_ratio
            # Total recompute time = per_mb × num_microbatches
            recompute_time = recompute_time_per_mb * num_microbatches
        
        # Store in metadata
        ir.metadata["optimizer_time"] = optimizer_time
        ir.metadata["recompute_time"] = recompute_time
        ir.metadata["recompute_time_per_mb"] = recompute_time_per_mb
        ir.metadata["forward_time_per_mb"] = forward_time_per_mb
        ir.metadata["total_ops_time"] = total_ops_time
        ir.metadata["total_weight_bytes"] = total_weight_bytes
        ir.metadata["unique_layer_weights"] = unique_layer_weights
        ir.metadata["num_microbatches"] = num_microbatches
        ir.metadata["optimizer_config"] = {
            "type": self.config.optimizer_type,
            "master_weights": self.config.master_weights,
            "gradient_checkpointing": self.config.gradient_checkpointing,
            "checkpoint_ratio": self.config.checkpoint_ratio,
        }
        
        # Add optimizer op to schedule (runs once after all microbatches)
        if optimizer_time > 0:
            device = 0
            optim_op = ScheduledOp(
                op_id="optimizer_step",
                op_type="OptimizerStep",
                device=device,
                stream="compute",
                start=max_end_time,
                duration=optimizer_time,
                attrs={
                    "weight_bytes": total_weight_bytes,
                    "optimizer_type": self.config.optimizer_type,
                    "num_unique_layers": len(unique_layer_weights),
                },
            )
            ir.add_op(optim_op)
        
        # Add recompute op to schedule (if checkpointing enabled)
        if recompute_time > 0:
            # Recompute is interleaved with backward pass, but we model it as separate
            recompute_op = ScheduledOp(
                op_id="recompute_step",
                op_type="RecomputeStep",
                device=0,
                stream="compute",
                start=max_end_time + (optimizer_time if optimizer_time > 0 else 0),
                duration=recompute_time,
                attrs={
                    "forward_time_per_mb": forward_time_per_mb,
                    "checkpoint_ratio": self.config.checkpoint_ratio,
                    "num_microbatches": num_microbatches,
                },
            )
            ir.add_op(recompute_op)
            ir.metadata["effective_forward_time"] = forward_time_per_mb + recompute_time_per_mb
        
        return ir
    
    def _compute_optimizer_time(self, weight_bytes: Union[int, float, Expr]) -> Union[float, Expr]:
        """Compute optimizer step time.
        
        Optimizer update is an element-wise operation where compute and memory
        access CANNOT overlap (unlike matrix operations). Therefore we use:
        time = compute_time + memory_time (not max)
        
        This matches Calculon's behavior and real-world measurements.
        
        Memory access model (following Calculon):
        - Access optimizer state: m, v (FP32) + optionally master weights
        - Total: ~12 bytes per param for Adam with master weights
        """
        # Number of parameters (assuming FP16 weights)
        num_params = weight_bytes / self.config.dtype_bytes
        
        # Memory access per param (optimizer state size)
        mem_access_per_param = self.config.get_optimizer_memory_access_per_param()
        total_mem_access = num_params * mem_access_per_param
        
        # FLOPs per param (Adam: ~10-11 ops)
        flops_per_param = self.config.get_optimizer_flops_per_param()
        total_flops = num_params * flops_per_param
        
        # For optimizer (element-wise): time = compute_time + memory_time
        # Cannot use Roofline (max) because compute and memory access don't overlap
        # This is consistent with Calculon's model
        memory_time = total_mem_access / self.memory_bandwidth
        compute_time = total_flops / self.peak_tflops
        
        return memory_time + compute_time
    
    def _compute_recompute_time(self, forward_time: Union[int, float, Expr]) -> Union[float, Expr]:
        """Compute gradient checkpointing recomputation time.
        
        When using gradient checkpointing, we recompute activations during
        backward pass. The recompute time is approximately equal to the
        forward time for checkpointed layers.
        """
        return forward_time * self.config.checkpoint_ratio

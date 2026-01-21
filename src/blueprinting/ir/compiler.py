"""Compiler - Orchestrates IR passes to produce simulation results.

The Compiler combines multiple passes into a compilation pipeline,
transforming GraphIR through various stages to produce a SimulationResult.
"""

from typing import Any, Dict, List, Optional, Union

from sympy import Symbol

from .graph import GraphIR
from .schedule import ScheduleIR
from .result import SimulationResult
from .passes.base import Pass
from .passes.workload import WorkloadPass
from .passes.parallel import ParallelPass
from .passes.schedule import SchedulePass
from .passes.evaluate import EvaluatePass


class Compiler:
    """Compiler for IR-based model simulation.
    
    The Compiler orchestrates a pipeline of passes that transform
    a GraphIR into a SimulationResult. Each pass adds or transforms
    information in the IR.
    
    Example:
        # Build a graph
        graph = build_transformer_model(num_layers=32, hidden=4096, ...)
        
        # Create compiler with passes
        compiler = (Compiler()
            .add_pass(WorkloadPass())
            .add_pass(ParallelPass(tp=8, pp=4, dp=2))
            .add_pass(SchedulePass(system_config))
            .add_pass(EvaluatePass(subs={...})))
        
        # Compile
        result = compiler.compile(graph)
        print(f"Peak memory: {result.peak_memory / 1e9:.2f} GB")
        print(f"E2E time: {result.e2e_time * 1e3:.2f} ms")
    """
    
    def __init__(self, system_config: Optional[Dict[str, Any]] = None):
        """Initialize Compiler.
        
        Args:
            system_config: Optional system configuration to use for scheduling
        """
        self.system_config = system_config or {}
        self.passes: List[Pass] = []
    
    def add_pass(self, p: Pass) -> "Compiler":
        """Add a pass to the compilation pipeline.
        
        Args:
            p: Pass to add
            
        Returns:
            Self for chaining
        """
        self.passes.append(p)
        return self
    
    def compile(self, ir: GraphIR) -> SimulationResult:
        """Compile the graph through all passes.
        
        Args:
            ir: Input GraphIR
            
        Returns:
            SimulationResult after all passes
        """
        # Clear expression cache at the start of each compilation
        # This ensures id()-based cache keys are valid for this compilation
        from .symmax import clear_expr_cache
        clear_expr_cache()
        
        current: Any = ir
        
        for p in self.passes:
            current = p.run(current)
        
        # Ensure we return a SimulationResult
        if isinstance(current, SimulationResult):
            return current
        elif isinstance(current, ScheduleIR):
            # If no EvaluatePass was added, create a default one
            return EvaluatePass().run(current)
        else:
            raise ValueError(f"Unexpected final IR type: {type(current)}")
    
    def reset(self) -> "Compiler":
        """Clear all passes."""
        self.passes.clear()
        return self
    
    @staticmethod
    def default_pipeline(
        tp: int = 1,
        pp: int = 1,
        dp: int = 1,
        subs: Optional[Dict] = None,
        system_config: Optional[Dict] = None,
        training: bool = True,
    ) -> "Compiler":
        """Create a compiler with a default pass pipeline.
        
        Args:
            tp: Tensor parallelism degree
            pp: Pipeline parallelism degree
            dp: Data parallelism degree
            subs: Symbol substitutions
            system_config: System configuration
            training: Whether this is a training workload
            
        Returns:
            Configured Compiler
        """
        system_config = system_config or {
            "peak_tflops": 312,  # A100
            "memory_bandwidth_gbps": 2000,
            "network_bandwidth_gbps": 400,
            "memory_capacity_gb": 80,
        }
        
        return (Compiler(system_config)
            .add_pass(WorkloadPass())
            .add_pass(ParallelPass(tp=tp, pp=pp, dp=dp))
            .add_pass(SchedulePass(system_config=system_config))
            .add_pass(EvaluatePass(subs=subs, training=training)))
    
    def __repr__(self) -> str:
        passes_str = ", ".join(p.name for p in self.passes)
        return f"Compiler([{passes_str}])"


def compile_model(
    graph: GraphIR,
    tp: int = 1,
    pp: int = 1,
    dp: int = 1,
    batch_size: int = 1,
    seq_len: int = 2048,
    hidden: int = 4096,
    feedforward: Optional[int] = None,
    num_layers: int = 32,
    system_config: Optional[Dict] = None,
    training: bool = True,
) -> SimulationResult:
    """Convenience function to compile a model graph.
    
    Args:
        graph: GraphIR to compile
        tp: Tensor parallelism degree
        pp: Pipeline parallelism degree
        dp: Data parallelism degree
        batch_size: Batch size
        seq_len: Sequence length
        hidden: Hidden dimension
        feedforward: Feedforward dimension (default: 4 * hidden)
        num_layers: Number of transformer layers
        system_config: System configuration
        training: Whether this is a training workload
        
    Returns:
        SimulationResult
    """
    feedforward = feedforward or 4 * hidden
    
    subs = {
        "B": batch_size,
        "S": seq_len,
        "H": hidden,
        "FF": feedforward,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "hidden": hidden,
        "feedforward": feedforward,
        "batch_seq": batch_size * seq_len,
        "num_layers": num_layers,
    }
    
    compiler = Compiler.default_pipeline(
        tp=tp,
        pp=pp,
        dp=dp,
        subs=subs,
        system_config=system_config,
        training=training,
    )
    
    return compiler.compile(graph)

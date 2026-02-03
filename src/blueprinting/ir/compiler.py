"""Compiler - Orchestrates IR passes to produce simulation results.

The Compiler combines multiple passes into a compilation pipeline,
transforming hierarchical GraphIR through various stages to produce a SimulationResult.

IR Pipeline:
    GraphIR (Block)
         │
         ↓ ParallelPass (标记并行策略)
         │
         ↓ ExpandPass (Block → Op)
         │
    ScheduleIR (Op, 无 workload)
         │
         ↓ SchedulePass (计算 workload + timing)
         │
    ScheduleIR (Op, 有 workload)
         │
         ↓ OptimizerPass (追加反向 Op，训练时)
         │
         ↓ PipelineSchedulePass (PP 调度，PP>1 时)
         │
         ↓ TimelinePass (Op → Event)
         │
    TimelineIR (Event)
         │
         ↓ SimulatePass (评估)
         │
    SimulationResult
"""

from typing import Any, Dict, List, Optional

from .graph import GraphIR
from .passes.base import Pass
from .passes.expand import ExpandPass
from .passes.optimizer import OptimizerConfig, OptimizerPass
from .passes.parallel import ParallelPass
from .passes.pipeline import PipelineSchedulePass
from .passes.schedule import SchedulePass
from .passes.timeline_v2 import SimulatePass, TimelinePassV2
from .result import SimulationResult
from .schedule import ScheduleIR
from .timeline import TimelineIR


class Compiler:
    """Compiler for IR-based model simulation.

    The Compiler orchestrates a pipeline of passes that transform
    a hierarchical GraphIR into a SimulationResult. Each pass adds
    or transforms information in the IR.

    Example:
        # Build a graph
        graph = build_transformer_model(num_layers=32, hidden=4096, ...)

        # Create compiler with passes
        compiler = (Compiler()
            .add_pass(ParallelPass(tp=8, pp=4, dp=2))
            .add_pass(ExpandPass())
            .add_pass(SchedulePass(system_config))
            .add_pass(OptimizerPass(OptimizerConfig()))  # 训练时
            .add_pass(PipelineSchedulePass())  # PP>1 时
            .add_pass(TimelinePassV2())
            .add_pass(SimulatePass(subs={...})))

        # Compile
        result = compiler.compile(graph)
        print(f"Peak memory: {result.peak_memory / 1e9:.2f} GB")
        print(f"E2E time: {result.e2e_time * 1e3:.2f} ms")
    """

    def __init__(
        self, system_config: Optional[Dict[str, Any]] = None, debug: bool = False
    ):
        """Initialize Compiler.

        Args:
            system_config: Optional system configuration to use for scheduling
            debug: If True, print IR snapshot after each pass
        """
        self.system_config = system_config or {}
        self.passes: List[Pass] = []
        self.debug = debug

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
        from .symmax import clear_expr_cache

        clear_expr_cache()

        current: Any = ir

        for p in self.passes:
            current = p.run(current)
            if self.debug:
                self._print_ir_snapshot(p, current)

        # Ensure we return a SimulationResult
        if isinstance(current, SimulationResult):
            return current
        elif isinstance(current, TimelineIR):
            # If no SimulatePass was added, create a default one
            return SimulatePass().run(current)
        elif isinstance(current, ScheduleIR):
            # Need to convert to TimelineIR first
            timeline = TimelinePassV2().run(current)
            return SimulatePass().run(timeline)
        else:
            raise ValueError(f"Unexpected final IR type: {type(current)}")

    def _print_ir_snapshot(self, p: Pass, ir: Any) -> None:
        """Print a snapshot of IR after each pass."""
        name = getattr(p, "name", p.__class__.__name__)

        if isinstance(ir, GraphIR):
            # Use the hierarchical repr
            print(f"[IR] {name}: {ir}")
        elif isinstance(ir, ScheduleIR):
            print(f"[IR] {name}: {ir}")
        elif isinstance(ir, TimelineIR):
            print(
                f"[IR] {name}: TimelineIR(events={len(ir.events)}, devices={ir.num_devices})"
            )
        elif isinstance(ir, SimulationResult):
            print(
                f"[IR] {name}: SimulationResult(peak_mem={ir.peak_memory/1e9:.2f}GB, time={ir.e2e_time*1e3:.2f}ms)"
            )
        else:
            print(f"[IR] {name}: {type(ir).__name__}")

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
        debug: bool = False,
    ) -> "Compiler":
        """Create a compiler with a default pass pipeline.

        Args:
            tp: Tensor parallelism degree
            pp: Pipeline parallelism degree
            dp: Data parallelism degree
            subs: Symbol substitutions
            system_config: System configuration
            training: Whether this is a training workload
            debug: Print IR snapshot after each pass

        Returns:
            Configured Compiler
        """
        system_config = system_config or {
            "peak_tflops": 312,  # A100
            "memory_bandwidth_gbps": 2000,
            "network_bandwidth_gbps": 400,
            "memory_capacity_gb": 80,
        }

        peak_tflops = system_config.get("peak_tflops", 312)
        memory_bandwidth = (
            system_config.get("memory_bandwidth_gbps", 2000) * 1e9
        )  # Convert to bytes/s
        network_bandwidth = (
            system_config.get("network_bandwidth_gbps", 400) * 1e9
        )  # Convert to bits/s

        compiler = (
            Compiler(system_config, debug=debug)
            .add_pass(ParallelPass(tp=tp, pp=pp, dp=dp))
            .add_pass(ExpandPass())
            .add_pass(
                SchedulePass(
                    peak_tflops=peak_tflops,
                    memory_bandwidth=memory_bandwidth,
                    network_bandwidth=network_bandwidth,
                )
            )
        )

        if training:
            compiler.add_pass(OptimizerPass(OptimizerConfig(dp=dp)))

        if pp > 1:
            compiler.add_pass(PipelineSchedulePass())

        compiler.add_pass(TimelinePassV2())
        compiler.add_pass(
            SimulatePass(subs=subs, peak_tflops=peak_tflops, training=training)
        )

        return compiler

    def __repr__(self) -> str:
        pass_names = [getattr(p, "name", p.__class__.__name__) for p in self.passes]
        return f"Compiler([{', '.join(pass_names)}])"


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
    debug: bool = False,
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
        debug: Print IR snapshot after each pass

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
        debug=debug,
    )

    return compiler.compile(graph)

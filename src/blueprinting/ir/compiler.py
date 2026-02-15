"""Compiler - Orchestrates IR passes to produce simulation results.

The Compiler combines multiple passes into a compilation pipeline,
transforming hierarchical GraphIR through various stages to produce a SimulationResult.

参数管理:
- 各 Pass 通过 @hp.param 从 hp.scope() 自动注入参数
- system namespace: 硬件参数 (peak_tflops, memory_bandwidth, ...)
- parallel namespace: 并行策略 (tp, pp, dp, training, ...)
- 便捷函数 compile_model() / compile_inference() 自动设置 hp.scope

IR Pipeline (训练):
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
         ↓ SymbolicEstimatePass (符号化估算)
         │
         ↓ TimelinePass (Op → Event)
         │
         ↓ OverlapAnalysisPass (重叠分析)
         │
    TimelineIR (Event)
         │
         ↓ SimulatePass (评估)
         │
    SimulationResult

IR Pipeline (推理):
    GraphIR (Block)
         │
         ↓ InferenceParallelPass (标记并行策略 + 推理模式)
         │
         ↓ InferenceExpandPass (Block → Op, 区分 context/generation)
         │
    ScheduleIR (Op, 无 workload)
         │
         ↓ InferenceSchedulePass (workload + timing + memory_pools)
         │
    ScheduleIR (Op + workload + memory_pools[weight, activation, kv_cache])
         │
         ↓ TimelinePass (Op → Event, memory_pools → ALLOC/FREE)
         │
    TimelineIR (Event)
         │
         ↓ SimulatePass (纯观测, 不区分训练/推理)
         │
    SimulationResult
"""

from typing import Any, Dict, List, Optional

import hyperparameter as hp

from .passes.base import Pass
from .passes.expand import ExpandPass
from .passes.inference import (
    InferenceExpandPass,
    InferenceParallelPass,
    InferenceSchedulePass,
    QuantConfig,
)
from .passes.optimizer import OptimizerConfig, OptimizerPass
from .passes.overlap import OverlapAnalysisPass
from .passes.parallel import ParallelPass
from .passes.pipeline import PipelineConfig, PipelineSchedulePass, PPScheduleMode
from .passes.schedule import SchedulePass
from .passes.simulate import SimulatePass
from .passes.symbolic_estimate import SymbolicEstimatePass
from .passes.timeline import TimelinePass
from .perf_database import PerfDatabase
from .program import Program, SimulationBackend, SymbolicBackend
from .result import SimulationResult
from .types import GraphIR, ScheduleIR, TimelineIR


class Compiler:
    """Compiler for IR-based model simulation.

    The Compiler orchestrates a pipeline of passes that transform
    a hierarchical GraphIR into a SimulationResult. Each pass adds
    or transforms information in the IR.

    使用方式:

    1. 手动组装 (完全控制):
        compiler = Compiler()
        compiler.add_pass(ParallelPass())
        compiler.add_pass(ExpandPass())
        ...
        result = compiler.compile(graph)

    2. 工厂方法 + hp.scope (推荐):
        with hp.scope(
            system={"peak_tflops": 1000, "memory_bandwidth": 3.35e12},
            parallel={"tp": 4, "pp": 48, "dp": 12},
        ):
            compiler = Compiler.default_pipeline(num_microbatches=64)
            result = compiler.compile(graph)

    3. 便捷函数 (最简):
        result = compile_model(graph, tp=4, pp=48, dp=12,
                               system_config={...})
    """

    def __init__(self, debug: bool = False):
        """Initialize Compiler.

        Args:
            debug: If True, print IR snapshot after each pass
        """
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
        return self.compile_to_program(ir).simulate()

    def _split_for_program(self) -> tuple[list[Pass], list[Pass]]:
        """将 pass 链拆分为前端（到 ScheduleIR）和后端（用于仿真）."""
        boundary = len(self.passes)
        for idx, p in enumerate(self.passes):
            if isinstance(p, (TimelinePass, OverlapAnalysisPass, SimulatePass)):
                boundary = idx
                break
        return self.passes[:boundary], self.passes[boundary:]

    def compile_to_program(self, ir: GraphIR) -> Program:
        """编译到 Program（残留程序），支持后续多次/多目的塌缩."""
        frontend_passes, backend_passes = self._split_for_program()
        current: Any = ir

        for p in frontend_passes:
            current = p.run(current)
            if self.debug:
                self._print_ir_snapshot(p, current)

        if not isinstance(current, ScheduleIR):
            raise ValueError(
                "compile_to_program expects frontend passes to produce ScheduleIR, "
                f"got {type(current)}"
            )

        # 快照当前 scope 参数，供 Program 后续多次塌缩复用
        try:
            scope = hp.scope.current()
            system_ns = getattr(scope, "system", None)
            parallel_ns = getattr(scope, "parallel", None)
            if system_ns is not None:
                current.metadata.setdefault(
                    "system_config",
                    {
                        k: v
                        for k, v in vars(system_ns).items()
                        if not k.startswith("_")
                    },
                )
            if parallel_ns is not None:
                current.metadata.setdefault(
                    "parallel_config",
                    {
                        k: v
                        for k, v in vars(parallel_ns).items()
                        if not k.startswith("_")
                    },
                )
        except Exception:
            pass

        timeline_pass = None
        overlap_pass = None
        simulate_pass = None
        for p in backend_passes:
            if timeline_pass is None and isinstance(p, TimelinePass):
                timeline_pass = p
            elif overlap_pass is None and isinstance(p, OverlapAnalysisPass):
                overlap_pass = p
            elif simulate_pass is None and isinstance(p, SimulatePass):
                simulate_pass = p

        simulation_backend = SimulationBackend(
            timeline_pass=timeline_pass,
            overlap_pass=overlap_pass,
            simulate_pass=simulate_pass,
        )
        symbolic_backend = SymbolicBackend()
        return Program(
            ir=current,
            simulation_backend=simulation_backend,
            symbolic_backend=symbolic_backend,
        )

    def _print_ir_snapshot(self, p: Pass, ir: Any) -> None:
        """Print a snapshot of IR after each pass."""
        name = getattr(p, "name", p.__class__.__name__)

        if isinstance(ir, GraphIR):
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
        num_microbatches: int = 1,
        training: bool = True,
        gradient_checkpointing: bool = False,
        dp: int = 1,
        subs: Optional[Dict] = None,
        debug: bool = False,
    ) -> "Compiler":
        """Create a compiler with a default training pass pipeline.

        系统参数和并行参数通过 hp.scope() 自动注入到各 Pass，
        不需要显式传递硬件配置。

        使用示例:
            with hp.scope(
                system={"peak_tflops": 1000, "memory_bandwidth": 3.35e12},
                parallel={"tp": 4, "pp": 48, "dp": 12},
            ):
                compiler = Compiler.default_pipeline(
                    num_microbatches=64,
                    training=True,
                    gradient_checkpointing=True,
                    dp=12,
                )
                result = compiler.compile(graph)

        Args:
            num_microbatches: micro-batch 数量
            training: 是否训练模式
            gradient_checkpointing: 是否启用梯度检查点
            dp: Data Parallelism 度数 (用于 OptimizerConfig ZeRO 分片)
            subs: 符号替换字典 (e.g., {"B": 4, "S": 2048})
            debug: 是否打印调试信息

        Returns:
            Configured Compiler
        """
        compiler = Compiler(debug=debug)

        # ParallelPass: tp, pp, dp from hp.scope(parallel=...)
        compiler.add_pass(ParallelPass())

        # ExpandPass: Block → Op
        compiler.add_pass(ExpandPass())

        # SchedulePass: peak_tflops, memory_bandwidth etc from hp.scope(system=...)
        compiler.add_pass(SchedulePass())

        # OptimizerPass: training, memory_bandwidth, peak_flops from hp.scope(parallel=...)
        if training:
            compiler.add_pass(OptimizerPass(
                optimizer_config=OptimizerConfig(
                    optimizer_type="adam",
                    master_weights=True,
                    gradient_checkpointing=gradient_checkpointing,
                    recompute_mode="full" if gradient_checkpointing else "none",
                    zero_stage=1,
                    dp=dp,
                ),
            ))

        # PipelineSchedulePass: p2p params from hp.scope(parallel=...)
        if num_microbatches > 1:
            compiler.add_pass(PipelineSchedulePass(
                config=PipelineConfig(
                    mode=PPScheduleMode.ONE_F_ONE_B,
                    num_microbatches=num_microbatches,
                ),
            ))

        # SymbolicEstimatePass: 符号化聚合估算
        compiler.add_pass(SymbolicEstimatePass())

        # TimelinePass: Op → Event (含内存追踪)
        compiler.add_pass(TimelinePass(track_memory=True))

        # OverlapAnalysisPass: 重叠分析
        compiler.add_pass(OverlapAnalysisPass())

        # SimulatePass: peak_tflops from hp.scope(system=...)
        compiler.add_pass(SimulatePass(subs=subs))

        return compiler

    @staticmethod
    def inference_pipeline(
        phase: str = "context",
        subs: Optional[Dict] = None,
        quant_config: Optional["QuantConfig"] = None,
        perf_db: Optional["PerfDatabase"] = None,
        debug: bool = False,
    ) -> "Compiler":
        """Create a compiler with an inference pass pipeline.

        推理管道:
            InferenceParallelPass → InferenceExpandPass → InferenceSchedulePass
            → TimelinePass → SimulatePass

        系统参数和并行参数通过 hp.scope() 自动注入。

        使用示例:
            with hp.scope(
                system={"peak_tflops": 1000, "memory_bandwidth": 3.35e12},
                parallel={"tp": 8, "pp": 1, "phase": "generation"},
            ):
                compiler = Compiler.inference_pipeline(
                    phase="generation",
                    quant_config=QuantConfig.fp8(),
                )
                result = compiler.compile(graph)

        Args:
            phase: 推理阶段 ("context" for prefill, "generation" for decode)
            subs: 符号替换字典
            quant_config: 量化配置 (默认 fp16)
            perf_db: PerfDatabase 实例 (可选, 用于查表估算)
            debug: 是否打印调试信息

        Returns:
            Configured Compiler
        """
        compiler = Compiler(debug=debug)

        # InferenceParallelPass: tp, pp, phase from hp.scope(parallel=...)
        compiler.add_pass(InferenceParallelPass(phase=phase))

        # InferenceExpandPass: phase from hp.scope(parallel=...)
        compiler.add_pass(InferenceExpandPass(phase=phase))

        # InferenceSchedulePass: hardware params from hp.scope(parallel=...)
        compiler.add_pass(InferenceSchedulePass(
            quant_config=quant_config,
            perf_db=perf_db,
        ))

        # TimelinePass: Op → Event (含内存追踪)
        compiler.add_pass(TimelinePass(track_memory=True))

        # SimulatePass: peak_tflops from hp.scope(system=...)
        compiler.add_pass(SimulatePass(subs=subs))

        return compiler

    def __repr__(self) -> str:
        pass_names = [getattr(p, "name", p.__class__.__name__) for p in self.passes]
        return f"Compiler([{', '.join(pass_names)}])"


def _build_scope_params(
    system_config: Optional[Dict[str, Any]] = None,
    tp: int = 1,
    pp: int = 1,
    dp: int = 1,
    training: bool = True,
    gradient_checkpointing: bool = False,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """从 system_config 和并行参数构建 hp.scope 所需的 system/parallel 字典.

    Args:
        system_config: 系统配置 (包含 peak_tflops, memory_bandwidth_gbps 等)
        tp: Tensor Parallelism 度数
        pp: Pipeline Parallelism 度数
        dp: Data Parallelism 度数
        training: 是否训练模式
        gradient_checkpointing: 是否启用梯度检查点

    Returns:
        (system_params, parallel_params) tuple
    """
    system_config = system_config or {
        "peak_tflops": 312,  # A100
        "memory_bandwidth_gbps": 2000,
        "network_bandwidth_gbps": 400,
    }

    peak_tflops = system_config.get("peak_tflops", 312)
    memory_bandwidth = system_config.get("memory_bandwidth_gbps", 2000) * 1e9
    network_bandwidth = system_config.get("network_bandwidth_gbps", 400) * 1e9

    system_params = {
        "peak_tflops": peak_tflops,
        "memory_bandwidth": memory_bandwidth,
        "network_bandwidth": network_bandwidth,
        "network_efficiency": system_config.get("network_efficiency", 0.65),
        "network_latency": system_config.get("network_latency", 10e-6),
        "compute_efficiency": system_config.get("compute_efficiency", 0.95),
        "processing_mode": system_config.get("processing_mode", "roofline"),
        "all_reduce_offset": system_config.get("all_reduce_offset", 1.0),
    }

    parallel_params = {
        "tp": tp,
        "pp": pp,
        "dp": dp,
        "training": training,
        "gradient_checkpointing": gradient_checkpointing,
    }

    return system_params, parallel_params


def compile_model(
    graph: GraphIR,
    tp: int = 1,
    pp: int = 1,
    dp: int = 1,
    num_microbatches: int = 1,
    batch_size: int = 1,
    seq_len: int = 2048,
    hidden: int = 4096,
    feedforward: Optional[int] = None,
    num_layers: int = 32,
    system_config: Optional[Dict] = None,
    training: bool = True,
    gradient_checkpointing: bool = False,
    debug: bool = False,
) -> SimulationResult:
    """Convenience function to compile a model graph.

    自动设置 hp.scope 并创建 Compiler pipeline。
    适合一行代码完成编译的场景。

    Args:
        graph: GraphIR to compile
        tp: Tensor parallelism degree
        pp: Pipeline parallelism degree
        dp: Data parallelism degree
        num_microbatches: micro-batch 数量
        batch_size: Batch size
        seq_len: Sequence length
        hidden: Hidden dimension
        feedforward: Feedforward dimension (default: 4 * hidden)
        num_layers: Number of transformer layers
        system_config: System configuration
        training: Whether this is a training workload
        gradient_checkpointing: 是否启用梯度检查点
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

    system_params, parallel_params = _build_scope_params(
        system_config=system_config, tp=tp, pp=pp, dp=dp,
        training=training, gradient_checkpointing=gradient_checkpointing,
    )

    with hp.scope(system=system_params, parallel=parallel_params):
        compiler = Compiler.default_pipeline(
            num_microbatches=num_microbatches,
            training=training,
            gradient_checkpointing=gradient_checkpointing,
            dp=dp,
            subs=subs,
            debug=debug,
        )
        return compiler.compile(graph)


def compile_to_program(
    graph: GraphIR,
    tp: int = 1,
    pp: int = 1,
    dp: int = 1,
    num_microbatches: int = 1,
    system_config: Optional[Dict] = None,
    training: bool = True,
    gradient_checkpointing: bool = False,
    debug: bool = False,
) -> Program:
    """Convenience function to compile a graph into Program."""
    system_params, parallel_params = _build_scope_params(
        system_config=system_config, tp=tp, pp=pp, dp=dp,
        training=training, gradient_checkpointing=gradient_checkpointing,
    )

    with hp.scope(system=system_params, parallel=parallel_params):
        compiler = Compiler.default_pipeline(
            num_microbatches=num_microbatches,
            training=training,
            gradient_checkpointing=gradient_checkpointing,
            dp=dp,
            subs=None,
            debug=debug,
        )
        return compiler.compile_to_program(graph)


def compile_inference(
    graph: GraphIR,
    tp: int = 1,
    pp: int = 1,
    phase: str = "context",
    batch_size: int = 1,
    seq_len: int = 2048,
    hidden: int = 4096,
    feedforward: Optional[int] = None,
    num_layers: int = 32,
    num_heads: Optional[int] = None,
    num_kv_heads: Optional[int] = None,
    kv_len: Optional[int] = None,
    system_config: Optional[Dict] = None,
    quant_config: Optional["QuantConfig"] = None,
    perf_db: Optional["PerfDatabase"] = None,
    debug: bool = False,
) -> SimulationResult:
    """Convenience function to compile a model graph for inference.

    推理编译的便捷函数。自动设置 hp.scope 并创建推理 Compiler pipeline。

    Example:
        # FP8 量化推理 (roofline 模型)
        result = compile_inference(
            graph, tp=8, phase="generation",
            batch_size=32, seq_len=2048,
            quant_config=QuantConfig.fp8(),
        )

        # 使用实测性能数据库
        from blueprinting.ir.perf_database import PerfDatabase
        db = PerfDatabase("h100_sxm", "trtllm", "1.0.0rc3")
        result = compile_inference(
            graph, tp=8, phase="generation",
            batch_size=32, seq_len=2048,
            quant_config=QuantConfig.fp8(),
            perf_db=db,
        )

    Args:
        graph: GraphIR to compile
        tp: Tensor parallelism degree
        pp: Pipeline parallelism degree
        phase: Inference phase ("context" for prefill, "generation" for decode)
        batch_size: Batch size (推理请求数)
        seq_len: Sequence length (输入序列长度)
        hidden: Hidden dimension
        feedforward: Feedforward dimension (default: 4 * hidden)
        num_layers: Number of transformer layers
        num_heads: Number of attention heads (default: hidden // 128)
        num_kv_heads: Number of KV heads for GQA/MQA (default: same as num_heads)
        kv_len: KV cache length for generation (default: seq_len)
        system_config: System configuration
        quant_config: Quantization configuration (default: fp16)
        perf_db: PerfDatabase for table-driven estimation (optional)
        debug: Print IR snapshot after each pass

    Returns:
        SimulationResult
    """
    feedforward = feedforward or 4 * hidden
    num_heads = num_heads or hidden // 128
    num_kv_heads = num_kv_heads or num_heads
    kv_len = kv_len or seq_len

    subs = {
        "B": batch_size,
        "S": seq_len,
        "H": hidden,
        "FF": feedforward,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "hidden": hidden,
        "feedforward": feedforward,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "kv_len": kv_len,
    }

    system_params, parallel_params = _build_scope_params(
        system_config=system_config, tp=tp, pp=pp, dp=1, training=False,
    )
    # 推理阶段写入 parallel 空间
    parallel_params["phase"] = phase

    with hp.scope(system=system_params, parallel=parallel_params):
        compiler = Compiler.inference_pipeline(
            phase=phase,
            subs=subs,
            quant_config=quant_config,
            perf_db=perf_db,
            debug=debug,
        )
        return compiler.compile(graph)

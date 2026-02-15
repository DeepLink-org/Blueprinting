"""Tests for Program and progressive collapse APIs."""

from __future__ import annotations

import hyperparameter as hp
import pytest
from sympy import Symbol

from blueprinting.ir.compiler import Compiler
from blueprinting.ir.dsl import build_transformer_model
from blueprinting.ir.program import CollapseBackend, Program
from blueprinting.ir.types import MemoryPool, OpNode, ScheduleIR, ScheduledOp


def _build_test_graph():
    return build_transformer_model(
        model_name="program-test",
        hidden=1024,
        feedforward=4096,
        num_layers=1,
        num_heads=8,
        head_dim=128,
        seq_len=128,
        batch_size=2,
        tp=1,
    )


def _build_scope():
    system = {
        "peak_tflops": 312,
        "memory_bandwidth": 2e12,
        "network_bandwidth": 4e11,
        "network_efficiency": 0.65,
        "network_latency": 10e-6,
        "compute_efficiency": 0.95,
        "processing_mode": "roofline",
        "all_reduce_offset": 1.0,
    }
    parallel = {
        "tp": 1,
        "pp": 1,
        "dp": 1,
        "training": True,
        "gradient_checkpointing": False,
        "tp_comm_type": "ar",
        "sequence_parallel": False,
        "memory_bandwidth": 2e12,
        "peak_flops": 312e12,
    }
    return hp.scope(system=system, parallel=parallel)


def _build_symbolic_schedule_ir() -> ScheduleIR:
    x = Symbol("x", positive=True)
    ir = ScheduleIR(num_devices=1)
    op = ScheduledOp(
        op=OpNode(
            name="op0",
            op_type="Matmul",
            flops=x,
            memory_bytes=2 * x,
            comm_bytes=0,
        ),
        start=x,
        duration=2 * x,
    )
    ir.add_op(op, stage=0, device=0)
    ir.memory_pools.append(
        MemoryPool(
            name="act",
            mem_type="activation",
            size_bytes=4 * x,
            alloc_time=0,
            free_time=3 * x,
            device=0,
            metadata={"scale": x},
        )
    )
    ir.metadata["x_meta"] = x
    return ir


def test_schedule_ir_subs_and_free_symbols():
    ir = _build_symbolic_schedule_ir()
    assert {s.name for s in ir.free_symbols} == {"x"}

    substituted = ir.subs({"x": 4})
    assert substituted.free_symbols == set()

    op = list(substituted.iter_ops())[0]
    assert op.start == 4
    assert op.duration == 8
    assert op.op.flops == 4
    assert substituted.memory_pools[0].size_bytes == 16
    assert substituted.metadata["x_meta"] == 4

    # 原对象不应被 bind/subs 改写
    original_op = list(ir.iter_ops())[0]
    assert str(original_op.start) == "x"


def test_compile_to_program_returns_program():
    graph = _build_test_graph()
    with _build_scope():
        compiler = Compiler.default_pipeline(training=True, num_microbatches=1)
        program = compiler.compile_to_program(graph)
    assert isinstance(program, Program)
    assert program.schedule_ir.total_ops > 0


def test_program_simulate_matches_compile():
    graph = _build_test_graph()
    with _build_scope():
        compiler = Compiler.default_pipeline(training=True, num_microbatches=1)
        result_compile = compiler.compile(graph)
        result_program = compiler.compile_to_program(graph).simulate()

    assert result_compile.e2e_time == pytest.approx(result_program.e2e_time, rel=1e-9)
    assert result_compile.peak_memory == pytest.approx(result_program.peak_memory, rel=1e-9)


def test_program_bind_order_invariant():
    program = Program(_build_symbolic_schedule_ir())
    r1 = program.bind(x=3).simulate()
    r2 = program.bind(x=3).simulate()
    assert r1.e2e_time == pytest.approx(r2.e2e_time, rel=1e-9)
    assert r1.peak_memory == pytest.approx(r2.peak_memory, rel=1e-9)


def test_program_collapse_custom_backend():
    class DummyBackend(CollapseBackend):
        def collapse(self, ir: ScheduleIR, **params):
            return len(ir.free_symbols), params.get("tag", "none")

    program = Program(_build_symbolic_schedule_ir())
    symbols_left, tag = program.collapse(DummyBackend(), x=2, tag="scan")
    assert symbols_left == 0
    assert tag == "scan"


def test_program_reuse_multiple_directions():
    base = Program(_build_symbolic_schedule_ir())
    low = base.bind(x=2).simulate()
    high = base.bind(x=5).simulate()
    assert low.e2e_time < high.e2e_time


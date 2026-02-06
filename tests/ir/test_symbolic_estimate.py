"""Tests for SymbolicEstimate and SymbolicEstimatePass."""


from blueprinting.ir import GraphIR
from blueprinting.ir.dsl import build_transformer_model
from blueprinting.ir.estimate import SymbolicEstimate
from blueprinting.ir.passes import (
    ExpandPass,
    OptimizerConfig,
    OptimizerPass,
    ParallelPass,
    Pipeline,
    SchedulePass,
    SimulatePass,
    SymbolicEstimatePass,
    TimelinePass,
)


def _build_test_graph() -> GraphIR:
    """构建测试用的 GraphIR."""
    return build_transformer_model(
        model_name="test-model",
        hidden=4096,
        feedforward=11008,
        num_layers=2,
        num_heads=32,
        head_dim=128,
        seq_len=2048,
        batch_size=4,
        tp=1,
    )


def _build_schedule_ir(tp=1, pp=1):
    """构建 ScheduleIR（经过 Expand + Schedule）."""
    graph = _build_test_graph()
    pipeline = Pipeline([
        ParallelPass(tp=tp, pp=pp),
        ExpandPass(),
        SchedulePass(peak_tflops=312.0, memory_bandwidth=2.0e12),
        OptimizerPass(
            optimizer_config=OptimizerConfig(
                optimizer_type="adam",
                gradient_checkpointing=False,
            ),
            training=True,
        ),
    ])
    return pipeline.run(graph)


class TestSymbolicEstimatePass:
    """测试 SymbolicEstimatePass 的聚合逻辑."""

    def test_pass_is_transparent(self):
        """SymbolicEstimatePass 应透明传递 ScheduleIR."""
        schedule_ir = _build_schedule_ir()
        ops_before = list(schedule_ir.iter_ops())

        pass_ = SymbolicEstimatePass()
        result_ir = pass_.run(schedule_ir)

        # 输出应该还是同一个 ScheduleIR
        assert result_ir is schedule_ir
        # Op 数量不变
        assert len(list(result_ir.iter_ops())) == len(ops_before)

    def test_estimate_in_metadata(self):
        """SymbolicEstimatePass 应在 metadata 中存入 SymbolicEstimate."""
        schedule_ir = _build_schedule_ir()
        pass_ = SymbolicEstimatePass()
        pass_.run(schedule_ir)

        assert "symbolic_estimate" in schedule_ir.metadata
        est = schedule_ir.metadata["symbolic_estimate"]
        assert isinstance(est, SymbolicEstimate)

    def test_time_aggregation(self):
        """时间聚合应该产生非零值."""
        schedule_ir = _build_schedule_ir()
        pass_ = SymbolicEstimatePass()
        pass_.run(schedule_ir)
        est = schedule_ir.metadata["symbolic_estimate"]

        # 前向和反向时间应该 > 0
        assert est.forward_time > 0
        assert est.backward_time > 0
        # e2e 时间应该 > 0
        assert est.e2e_time > 0

    def test_memory_aggregation(self):
        """内存聚合应该产生非零值."""
        schedule_ir = _build_schedule_ir()
        pass_ = SymbolicEstimatePass()
        pass_.run(schedule_ir)
        est = schedule_ir.metadata["symbolic_estimate"]

        # 权重内存应该 > 0
        assert est.weight_memory > 0
        # 峰值内存应该 > 0
        assert est.peak_memory > 0

    def test_breakdown_by_op(self):
        """应按 op_type 分解时间."""
        schedule_ir = _build_schedule_ir()
        pass_ = SymbolicEstimatePass()
        pass_.run(schedule_ir)
        est = schedule_ir.metadata["symbolic_estimate"]

        # Matmul 应该在分解中
        assert "Matmul" in est.time_breakdown_by_op
        assert est.time_breakdown_by_op["Matmul"] > 0

    def test_breakdown_by_phase(self):
        """应按 phase 分解时间."""
        schedule_ir = _build_schedule_ir()
        pass_ = SymbolicEstimatePass()
        pass_.run(schedule_ir)
        est = schedule_ir.metadata["symbolic_estimate"]

        assert "forward" in est.time_breakdown_by_phase
        assert "backward" in est.time_breakdown_by_phase
        assert est.time_breakdown_by_phase["forward"] > 0

    def test_e2e_time_consistency(self):
        """e2e_time 应等于各项之和（overlap=0 时）."""
        schedule_ir = _build_schedule_ir()
        pass_ = SymbolicEstimatePass(overlap_ratio=0.0)
        pass_.run(schedule_ir)
        est = schedule_ir.metadata["symbolic_estimate"]

        expected = (
            est.forward_time
            + est.backward_time
            + est.recompute_time
            + est.comm_time
            + est.bubble_time
        )
        # float 精度比较
        assert abs(est.e2e_time - expected) < 1e-12

    def test_peak_memory_consistency(self):
        """peak_memory 应等于各项之和."""
        schedule_ir = _build_schedule_ir()
        pass_ = SymbolicEstimatePass()
        pass_.run(schedule_ir)
        est = schedule_ir.metadata["symbolic_estimate"]

        expected = (
            est.weight_memory
            + est.activation_memory
            + est.gradient_memory
            + est.optimizer_memory
        )
        assert abs(est.peak_memory - expected) < 1e-6


class TestSymbolicEstimateAnalysis:
    """测试 SymbolicEstimate 的分析方法."""

    def _get_estimate(self) -> SymbolicEstimate:
        schedule_ir = _build_schedule_ir()
        pass_ = SymbolicEstimatePass()
        pass_.run(schedule_ir)
        return schedule_ir.metadata["symbolic_estimate"]

    def test_sweep_basic(self):
        """sweep 应返回正确数量的结果."""
        est = self._get_estimate()
        # 由于该 graph 已经完全具体化（无符号），sweep 应能 eval
        results = est.sweep(batch=[4, 8])
        # 但因为表达式已是数值，所有 batch 值应返回同样的结果
        assert len(results) == 2
        for params, metrics in results:
            assert "e2e_time" in metrics
            assert "peak_memory" in metrics

    def test_bottleneck_time(self):
        """bottleneck 应返回排序的时间分解."""
        est = self._get_estimate()
        bn = est.bottleneck()

        assert "time" in bn
        assert "memory" in bn
        # 时间分解应有 5 项
        assert len(bn["time"]) == 5
        # 应按值从大到小排序
        values = [v for _, v, _ in bn["time"]]
        assert values == sorted(values, reverse=True)
        # 比例之和应接近 1
        fractions = [f for _, _, f in bn["time"]]
        assert abs(sum(fractions) - 1.0) < 1e-6

    def test_bottleneck_memory(self):
        """bottleneck 应返回排序的内存分解."""
        est = self._get_estimate()
        bn = est.bottleneck()

        assert len(bn["memory"]) == 4
        values = [v for _, v, _ in bn["memory"]]
        assert values == sorted(values, reverse=True)

    def test_sensitivity(self):
        """sensitivity 应返回偏导字典."""
        est = self._get_estimate()
        sens = est.sensitivity("batch")

        assert "e2e_time" in sens
        assert "forward_time" in sens
        # 对于纯数值表达式，导数可能为 0
        # 只要不报错就 OK

    def test_to_latex(self):
        """to_latex 应返回字符串字典."""
        est = self._get_estimate()
        latex = est.to_latex()

        assert isinstance(latex, dict)
        assert "e2e_time" in latex
        assert "peak_memory" in latex
        # 每个值应为字符串
        for v in latex.values():
            assert isinstance(v, str)

    def test_repr(self):
        """repr 应返回可读字符串."""
        est = self._get_estimate()
        s = repr(est)
        assert "SymbolicEstimate" in s
        assert "fw=" in s
        assert "bw=" in s


class TestPipelineIntegration:
    """测试 SymbolicEstimatePass 在完整 Pipeline 中的集成."""

    def test_full_pipeline_with_estimate(self):
        """完整 Pipeline 中应能获取 estimate."""
        graph = _build_test_graph()
        pipeline = Pipeline([
            ParallelPass(tp=1, pp=1),
            ExpandPass(),
            SchedulePass(peak_tflops=312.0, memory_bandwidth=2.0e12),
            OptimizerPass(
                optimizer_config=OptimizerConfig(
                    optimizer_type="adam",
                    gradient_checkpointing=False,
                ),
                training=True,
            ),
            SymbolicEstimatePass(),
            TimelinePass(),
            SimulatePass(peak_tflops=312.0),
        ])
        result = pipeline.run(graph)

        # result 应有 estimate
        assert result.estimate is not None
        assert isinstance(result.estimate, SymbolicEstimate)

        # estimate 的值应该合理
        assert result.estimate.forward_time > 0
        assert result.estimate.e2e_time > 0
        assert result.estimate.weight_memory > 0

    def test_pipeline_without_estimate(self):
        """没有 SymbolicEstimatePass 时，estimate 应为 None."""
        graph = _build_test_graph()
        pipeline = Pipeline([
            ParallelPass(tp=1, pp=1),
            ExpandPass(),
            SchedulePass(peak_tflops=312.0, memory_bandwidth=2.0e12),
            TimelinePass(),
            SimulatePass(peak_tflops=312.0),
        ])
        result = pipeline.run(graph)

        assert result.estimate is None

    def test_estimate_vs_simulation_comparison(self):
        """estimate 的值应与 simulation result 的值在同一量级."""
        graph = _build_test_graph()
        pipeline = Pipeline([
            ParallelPass(tp=1, pp=1),
            ExpandPass(),
            SchedulePass(peak_tflops=312.0, memory_bandwidth=2.0e12),
            OptimizerPass(
                optimizer_config=OptimizerConfig(
                    optimizer_type="adam",
                    gradient_checkpointing=False,
                ),
                training=True,
            ),
            SymbolicEstimatePass(),
            TimelinePass(),
            SimulatePass(peak_tflops=312.0),
        ])
        result = pipeline.run(graph)
        est = result.estimate

        # e2e_time 应在同一量级（不要求精确一致，因为 overlap 等差异）
        # estimate 没有 overlap 扣除，所以应 >= simulation e2e_time
        assert est.e2e_time > 0
        assert result.e2e_time > 0
        # 不超过 10x 的差异就算合理
        ratio = est.e2e_time / result.e2e_time if result.e2e_time > 0 else 0
        assert 0.1 < ratio < 10, f"e2e ratio={ratio:.2f}"

    def test_calibrate_with_simulation(self):
        """calibrate 应能从 SimulationResult 拟合 overlap_ratio."""
        graph = _build_test_graph()
        pipeline = Pipeline([
            ParallelPass(tp=1, pp=1),
            ExpandPass(),
            SchedulePass(peak_tflops=312.0, memory_bandwidth=2.0e12),
            OptimizerPass(
                optimizer_config=OptimizerConfig(
                    optimizer_type="adam",
                    gradient_checkpointing=False,
                ),
                training=True,
            ),
            SymbolicEstimatePass(),
            TimelinePass(),
            SimulatePass(peak_tflops=312.0),
        ])
        result = pipeline.run(graph)
        est = result.estimate

        # 校准前 overlap_ratio = 0
        assert est.overlap_ratio == 0.0

        # 校准
        calibrated = est.calibrate(result)
        # overlap_ratio 应在 [0, 1] 之间
        assert 0.0 <= calibrated.overlap_ratio <= 1.0

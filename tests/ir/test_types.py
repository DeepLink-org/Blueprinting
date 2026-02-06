"""Unit tests for types.py - Three-layer IR type definitions.

测试内容:
- Layer 1: BlockNode, GraphIR
- Layer 2: OpNode, ScheduledOp, DeviceSchedule, StageSchedule, ScheduleIR
- Layer 3: TimelineEvent, TimelineIR
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.blueprinting.ir.types import (
    # Layer 1
    BlockNode,
    DeviceSchedule,
    EventType,
    GraphIR,
    # Layer 2
    OpNode,
    ScheduledOp,
    ScheduleIR,
    StreamType,
    # Layer 3
    TimelineEvent,
    TimelineIR,
)

# ==============================================================================
# Layer 1: Graph IR (Block-level)
# ==============================================================================

class TestBlockNode:
    """测试 BlockNode 类."""

    def test_basic_creation(self):
        """测试基本创建."""
        block = BlockNode(name="fc1", block_type="Linear")
        assert block.name == "fc1"
        assert block.block_type == "Linear"
        assert block.children == []
        assert block.attrs == {}
        assert block.device is None

    def test_with_attrs(self):
        """测试带属性的创建."""
        block = BlockNode(
            name="fc1",
            block_type="Linear",
            attrs={"in_features": 1024, "out_features": 4096, "shard": "tp_col"}
        )
        assert block.attrs["in_features"] == 1024
        assert block.attrs["out_features"] == 4096
        assert block.attrs["shard"] == "tp_col"

    def test_add_child(self):
        """测试添加子节点."""
        parent = BlockNode(name="attn", block_type="Attention")
        child1 = BlockNode(name="q_proj", block_type="Linear")
        child2 = BlockNode(name="k_proj", block_type="Linear")

        parent.add_child(child1)
        parent.add_child(child2)

        assert len(parent.children) == 2
        assert parent.children[0].name == "q_proj"
        assert parent.children[1].name == "k_proj"

    def test_iter_blocks(self):
        """测试遍历所有 Block."""
        root = BlockNode(name="model", block_type="Transformer")
        layer = BlockNode(name="layer0", block_type="TransformerLayer")
        attn = BlockNode(name="attn", block_type="Attention")
        ffn = BlockNode(name="ffn", block_type="FFN")

        layer.add_child(attn)
        layer.add_child(ffn)
        root.add_child(layer)

        blocks = list(root.iter_blocks())
        assert len(blocks) == 4
        assert blocks[0].name == "model"
        assert blocks[1].name == "layer0"
        assert blocks[2].name == "attn"
        assert blocks[3].name == "ffn"

    def test_block_def_integration(self):
        """测试与 BlockDef 的集成."""
        # Linear 有参数
        linear = BlockNode(
            name="fc1",
            block_type="Linear",
            attrs={"in_features": 1024, "out_features": 4096}
        )
        assert linear.block_def is not None
        assert linear.has_params is True
        assert "weight" in linear.params

        # Attention 没有参数
        attn = BlockNode(name="attn", block_type="Attention")
        assert attn.block_def is not None
        assert attn.has_params is False
        assert attn.params == []

    def test_compute_methods(self):
        """测试计算方法委托."""
        linear = BlockNode(
            name="fc1",
            block_type="Linear",
            attrs={"in_features": 1024, "out_features": 4096, "batch_seq": 512}
        )

        flops = linear.compute_flops()
        param_bytes = linear.compute_param_bytes()

        # FLOPs = 2 * batch_seq * in * out + bias
        expected_flops = 2 * 512 * 1024 * 4096 + 512 * 4096
        assert flops == expected_flops

        # param_bytes = in * out * 2 + out * 2 (float16)
        expected_bytes = 1024 * 4096 * 2 + 4096 * 2
        assert param_bytes == expected_bytes

    def test_repr(self):
        """测试字符串表示."""
        block = BlockNode(name="fc1", block_type="Linear")
        repr_str = repr(block)
        assert "Linear" in repr_str
        assert "fc1" in repr_str


class TestGraphIR:
    """测试 GraphIR 类."""

    def test_empty_graph(self):
        """测试空图."""
        graph = GraphIR(name="empty")
        assert graph.name == "empty"
        assert graph.root is None
        assert graph.count_blocks() == 0
        assert "empty" in repr(graph)

    def test_with_root(self):
        """测试带根节点的图."""
        root = BlockNode(name="model", block_type="Transformer")
        layer = BlockNode(name="layer0", block_type="TransformerLayer")
        root.add_child(layer)

        graph = GraphIR(name="test_model", root=root)

        assert graph.root is root
        assert graph.count_blocks() == 2

    def test_iter_blocks(self):
        """测试遍历 Block."""
        root = BlockNode(name="model", block_type="Transformer")
        linear1 = BlockNode(name="fc1", block_type="Linear")
        linear2 = BlockNode(name="fc2", block_type="Linear")
        root.add_child(linear1)
        root.add_child(linear2)

        graph = GraphIR(name="test", root=root)
        blocks = list(graph.iter_blocks())

        assert len(blocks) == 3

    def test_count_params_blocks(self):
        """测试统计有参数的 Block."""
        root = BlockNode(name="model", block_type="Transformer")
        linear = BlockNode(name="fc1", block_type="Linear")
        attn = BlockNode(name="attn", block_type="Attention")
        norm = BlockNode(name="norm", block_type="RMSNorm")

        root.add_child(linear)
        root.add_child(attn)
        root.add_child(norm)

        graph = GraphIR(name="test", root=root)

        # Linear 和 RMSNorm 有参数，Attention 和 Transformer 没有
        assert graph.count_params_blocks() == 2

    def test_metadata(self):
        """测试元数据."""
        graph = GraphIR(
            name="test",
            metadata={"batch_size": 4, "seq_len": 2048, "hidden": 4096}
        )

        assert graph.metadata["batch_size"] == 4
        assert graph.metadata["seq_len"] == 2048
        assert graph.metadata["hidden"] == 4096


# ==============================================================================
# Layer 2: Schedule IR (Op-level)
# ==============================================================================

class TestOpNode:
    """测试 OpNode 类."""

    def test_basic_creation(self):
        """测试基本创建."""
        op = OpNode(name="mm_0", op_type="Matmul")
        assert op.name == "mm_0"
        assert op.op_type == "Matmul"
        assert op.inputs == []
        assert op.outputs == []

    def test_with_attrs(self):
        """测试带属性的创建."""
        op = OpNode(
            name="mm_0",
            op_type="Matmul",
            attrs={"M": 512, "K": 1024, "N": 4096}
        )
        assert op.attrs["M"] == 512
        assert op.attrs["K"] == 1024
        assert op.attrs["N"] == 4096

    def test_source_block(self):
        """测试来源 Block 追踪."""
        op = OpNode(
            name="mm_0",
            op_type="Matmul",
            source_block="/Transformer/Layer0/Attention/Linear(q_proj)"
        )
        assert "q_proj" in op.source_block

    def test_op_def_integration(self):
        """测试与 OpDef 的集成."""
        op = OpNode(name="mm_0", op_type="Matmul")
        assert op.op_def is not None
        assert op.category == "compute"
        assert op.is_comm is False

        comm_op = OpNode(name="ar_0", op_type="AllReduce")
        assert comm_op.category == "comm"
        assert comm_op.is_comm is True

    def test_compute_flops(self):
        """测试 FLOPs 计算."""
        op = OpNode(
            name="mm_0",
            op_type="Matmul",
            attrs={"M": 512, "K": 1024, "N": 4096}
        )

        flops = op.compute_flops()
        expected = 2 * 512 * 1024 * 4096
        assert flops == expected

    def test_explicit_workload(self):
        """测试显式设置 workload."""
        op = OpNode(
            name="mm_0",
            op_type="Matmul",
            flops=1000000,
            memory_bytes=2000000
        )

        assert op.compute_flops() == 1000000
        assert op.compute_memory() == 2000000


class TestScheduledOp:
    """测试 ScheduledOp 类."""

    def test_basic_creation(self):
        """测试基本创建."""
        op = OpNode(name="mm_0", op_type="Matmul")
        scheduled = ScheduledOp(op=op, device=0, start=0.0, duration=0.001)

        assert scheduled.op is op
        assert scheduled.device == 0
        assert scheduled.start == 0.0
        assert scheduled.duration == 0.001
        assert scheduled.end == 0.001

    def test_properties(self):
        """测试属性代理."""
        op = OpNode(name="mm_0", op_type="Matmul")
        scheduled = ScheduledOp(op=op, device=1, stage=0, stream="compute")

        assert scheduled.name == "mm_0"
        assert scheduled.op_type == "Matmul"

    def test_timing(self):
        """测试时序信息."""
        op = OpNode(name="mm_0", op_type="Matmul")
        scheduled = ScheduledOp(op=op, start=1.5, duration=0.5)

        assert scheduled.start == 1.5
        assert scheduled.duration == 0.5
        assert scheduled.end == 2.0


class TestDeviceSchedule:
    """测试 DeviceSchedule 类."""

    def test_add_ops(self):
        """测试添加 Op."""
        device = DeviceSchedule(device_id=0)

        op1 = ScheduledOp(OpNode("mm_0", "Matmul"), start=0, duration=1)
        op2 = ScheduledOp(OpNode("mm_1", "Matmul"), start=1, duration=1)

        device.add_op(op1)
        device.add_op(op2)

        assert len(device.ops) == 2
        assert device.end_time == 2

    def test_empty_device(self):
        """测试空设备."""
        device = DeviceSchedule(device_id=0)
        assert device.end_time == 0


class TestScheduleIR:
    """测试 ScheduleIR 类."""

    def test_basic_creation(self):
        """测试基本创建."""
        schedule = ScheduleIR()
        assert schedule.total_ops == 0

    def test_add_op(self):
        """测试添加 Op."""
        schedule = ScheduleIR()

        op1 = OpNode(name="mm_0", op_type="Matmul")
        op2 = OpNode(name="mm_1", op_type="Matmul")

        scheduled_op1 = ScheduledOp(op=op1, start=0.0, duration=0.001)
        scheduled_op2 = ScheduledOp(op=op2, start=0.001, duration=0.001)

        schedule.add_op(scheduled_op1, device=0, stage=0)
        schedule.add_op(scheduled_op2, device=0, stage=0)

        assert schedule.total_ops == 2

    def test_iter_ops(self):
        """测试遍历 Op."""
        schedule = ScheduleIR()

        for i in range(5):
            op = OpNode(name=f"mm_{i}", op_type="Matmul")
            scheduled = ScheduledOp(op=op, start=i * 0.001, duration=0.001)
            schedule.add_op(scheduled, device=0, stage=0)

        ops = list(schedule.iter_ops())
        assert len(ops) == 5


# ==============================================================================
# Layer 3: Timeline IR (Event-level)
# ==============================================================================

class TestTimelineEvent:
    """测试 TimelineEvent 类."""

    def test_basic_creation(self):
        """测试基本创建."""
        event = TimelineEvent(
            time=0.001,
            event_type=EventType.COMPUTE_START,
            resource_id="mm_0",
            device=0,
            stream=StreamType.COMPUTE
        )

        assert event.time == 0.001
        assert event.event_type == EventType.COMPUTE_START
        assert event.resource_id == "mm_0"
        assert event.device == 0
        assert event.stream == StreamType.COMPUTE

    def test_event_types(self):
        """测试事件类型."""
        assert EventType.COMPUTE_START.value == "compute_start"
        assert EventType.COMPUTE_END.value == "compute_end"
        assert EventType.COMM_START.value == "comm_start"
        assert EventType.COMM_END.value == "comm_end"

    def test_stream_types(self):
        """测试流类型."""
        assert StreamType.COMPUTE.value == "compute"
        assert StreamType.COMM.value == "comm"


class TestTimelineIR:
    """测试 TimelineIR 类."""

    def test_basic_creation(self):
        """测试基本创建."""
        timeline = TimelineIR()
        assert len(timeline.events) == 0
        assert timeline.end_time == 0

    def test_add_event(self):
        """测试添加事件."""
        timeline = TimelineIR()

        event = TimelineEvent(
            time=0.001,
            event_type=EventType.COMPUTE_START,
            resource_id="mm_0",
            device=0,
            stream=StreamType.COMPUTE
        )
        timeline.add_event(event)

        assert len(timeline.events) == 1
        assert timeline.end_time == 0.001

    def test_sort_events(self):
        """测试事件排序."""
        timeline = TimelineIR()

        # 乱序添加
        timeline.add_event(TimelineEvent(0.003, EventType.COMPUTE_END, "mm_1", 0, StreamType.COMPUTE))
        timeline.add_event(TimelineEvent(0.001, EventType.COMPUTE_START, "mm_0", 0, StreamType.COMPUTE))
        timeline.add_event(TimelineEvent(0.002, EventType.COMPUTE_END, "mm_0", 0, StreamType.COMPUTE))

        timeline.sort_events()

        assert timeline.events[0].time == 0.001
        assert timeline.events[1].time == 0.002
        assert timeline.events[2].time == 0.003


# ==============================================================================
# Run tests
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""测试三层 IR 架构.

运行方式:
    python -m pytest tests/test_ir_v2.py -v
    
或者直接运行:
    python tests/test_ir_v2.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.blueprinting.ir.types import (
    BlockNode, GraphIR,
    OpNode, ScheduledOp, ScheduleIR,
    TimelineEvent, EventType, TimelineIR,
)
from src.blueprinting.ir.dsl import (
    Model, Transformer,
    print_graph, graph_to_tree,
)
from src.blueprinting.ir.passes_v2 import (
    PrintGraphPass, PrintSchedulePass, PrintTimelinePass,
    ExpandPass, TimelinePassV2,
)
from src.blueprinting.ir.passes_v2.base import Pipeline


def test_graph_ir_construction():
    """测试 Graph IR 构建."""
    print("\n" + "="*70)
    print("测试 1: Graph IR 构建")
    print("="*70)
    
    # 使用 DSL 构建模型
    with Model("test_model") as m:
        m.metadata(batch_size=4, seq_len=2048, hidden=4096)
        
        with m.TransformerLayer("layer0") as layer:
            with layer.Attention("attn") as attn:
                attn.RMSNorm("input_norm", normalized_shape=4096)
                attn.Linear("q_proj", in_features=4096, out_features=4096, shard="tp_col")
                attn.Linear("k_proj", in_features=4096, out_features=4096, shard="tp_col")
                attn.Linear("v_proj", in_features=4096, out_features=4096, shard="tp_col")
                attn.Linear("o_proj", in_features=4096, out_features=4096, shard="tp_row")
            
            with layer.FFN("ffn") as ffn:
                ffn.RMSNorm("input_norm", normalized_shape=4096)
                ffn.Linear("gate", in_features=4096, out_features=11008, shard="tp_col")
                ffn.Linear("up", in_features=4096, out_features=11008, shard="tp_col")
                ffn.Linear("down", in_features=11008, out_features=4096, shard="tp_row")
    
    graph = m.build()
    
    # 验证基本结构
    assert graph.name == "test_model"
    assert graph.root is not None
    assert graph.root.block_type == "Transformer"
    assert len(graph.root.children) == 1  # 1 layer
    
    # 打印各种格式
    print("\n--- print_graph ---")
    print(print_graph(graph))
    
    print("\n--- graph_to_tree ---")
    print(graph_to_tree(graph))
    
    # 统计 Block 数量
    block_count = sum(1 for _ in graph.iter_blocks())
    print(f"\nBlock count: {block_count}")
    
    print("\n✅ Graph IR 构建测试通过!")
    return graph


def test_expand_pass():
    """测试 Expand Pass (Block → Op)."""
    print("\n" + "="*70)
    print("测试 2: Expand Pass (Block → Op)")
    print("="*70)
    
    # 构建 Graph IR
    with Model("expand_test") as m:
        m.metadata(batch_size=1, seq_len=2048, hidden=4096)
        
        with m.TransformerLayer("layer0") as layer:
            with layer.Attention("attn") as attn:
                attn.Linear("q_proj", in_features=4096, out_features=4096, shard="tp_col")
                attn.Linear("k_proj", in_features=4096, out_features=4096, shard="tp_col")
                attn.Linear("o_proj", in_features=4096, out_features=4096, shard="tp_row")
    
    graph = m.build()
    
    # 展开成 Schedule IR
    expand_pass = ExpandPass()
    schedule = expand_pass.run(graph)
    
    # 验证
    assert schedule.total_ops > 0
    print(f"\nSchedule IR: {schedule}")
    print(f"Total ops: {schedule.total_ops}")
    
    # 打印 Op 列表
    print("\nOps:")
    for op in schedule.iter_ops():
        print(f"  [{op.event_seq}] {op.op_type}({op.name}) t={op.start:.4f} d={op.duration:.6f}")
        if op.op.source_block:
            print(f"       from: {op.op.source_block}")
    
    print("\n✅ Expand Pass 测试通过!")
    return schedule


def test_timeline_pass():
    """测试 Timeline Pass (Op → Event)."""
    print("\n" + "="*70)
    print("测试 3: Timeline Pass (Op → Event)")
    print("="*70)
    
    # 构建 Graph IR
    with Model("timeline_test") as m:
        m.metadata(batch_size=1, seq_len=1024, hidden=4096)
        
        with m.TransformerLayer("layer0") as layer:
            layer.Linear("proj1", in_features=4096, out_features=4096)
            layer.Linear("proj2", in_features=4096, out_features=4096, shard="tp_row")
    
    graph = m.build()
    
    # 展开
    schedule = ExpandPass().run(graph)
    
    # 生成 Timeline
    timeline = TimelinePassV2().run(schedule)
    
    # 验证
    assert len(timeline.events) > 0
    print(f"\nTimeline IR: {timeline}")
    print(f"Total events: {len(timeline.events)}")
    print(f"End time: {timeline.end_time:.6f}")
    
    # 打印事件
    print("\nEvents:")
    for event in timeline.events[:20]:
        print(f"  t={event.time:.6f} {event.event_type.value:15s} {event.resource_id}")
    
    print("\n✅ Timeline Pass 测试通过!")
    return timeline


def test_full_pipeline():
    """测试完整的编译流水线."""
    print("\n" + "="*70)
    print("测试 4: 完整编译流水线")
    print("="*70)
    
    # 构建 Graph IR
    with Transformer("llama_mini") as m:
        m.metadata(
            batch_size=1,
            seq_len=2048,
            hidden=4096,
            num_heads=32,
            feedforward=11008,
        )
        
        # 2 layers
        for i in range(2):
            with m.TransformerLayer(f"layer{i}") as layer:
                with layer.Attention("attn") as attn:
                    attn.RMSNorm("norm")
                    attn.Linear("q_proj", in_features=4096, out_features=4096, shard="tp_col")
                    attn.Linear("k_proj", in_features=4096, out_features=4096, shard="tp_col")
                    attn.Linear("v_proj", in_features=4096, out_features=4096, shard="tp_col")
                    attn.Linear("o_proj", in_features=4096, out_features=4096, shard="tp_row")
                
                with layer.FFN("ffn") as ffn:
                    ffn.RMSNorm("norm")
                    ffn.Linear("gate", in_features=4096, out_features=11008, shard="tp_col")
                    ffn.Linear("up", in_features=4096, out_features=11008, shard="tp_col")
                    ffn.Linear("down", in_features=11008, out_features=4096, shard="tp_row")
    
    graph = m.build()
    
    # 构建流水线
    pipeline = Pipeline([
        PrintGraphPass("Input: Graph IR (Block-level)"),
        ExpandPass(),
        PrintSchedulePass("After Expand: Schedule IR (Op-level)"),
        TimelinePassV2(),
        PrintTimelinePass("After Timeline: Timeline IR (Event-level)"),
    ])
    
    print(f"\nPipeline: {pipeline}")
    print()
    
    # 运行流水线
    result = pipeline.run(graph)
    
    print("\n" + "="*70)
    print("编译流水线完成!")
    print("="*70)
    print(f"输入: GraphIR with {sum(1 for _ in graph.iter_blocks())} blocks")
    print(f"输出: TimelineIR with {len(result.events)} events")
    
    print("\n✅ 完整流水线测试通过!")


def test_types_relationship():
    """测试三层 IR 类型的关系."""
    print("\n" + "="*70)
    print("测试 5: 三层 IR 类型关系")
    print("="*70)
    
    print("""
三层 IR 架构:
=============

Layer 1: Graph IR (Block-level)
    - BlockNode: 模型结构的基本单元
    - GraphIR: Block 组成的树
    - 特点: 高层抽象，无具体计算

Layer 2: Schedule IR (Op-level)  
    - OpNode: 原子计算操作
    - ScheduledOp: 带时序的 Op
    - ScheduleIR: Stage → Device → Op
    - 特点: Block 展开成 Op 序列

Layer 3: Timeline IR (Event-level)
    - TimelineEvent: 细粒度事件
    - TimelineIR: 事件流
    - 特点: Op 展开成事件对
    """)
    
    # 创建各层 IR 实例
    block = BlockNode("test", "Linear", attrs={"in_features": 1024})
    graph = GraphIR(name="test", root=block)
    
    op = OpNode("mm", "Matmul")
    sched_op = ScheduledOp(op, device=0, start=0, duration=0.001)
    schedule = ScheduleIR()
    schedule.add_op(sched_op)
    
    event = TimelineEvent(0, EventType.COMPUTE_START, "mm")
    timeline = TimelineIR()
    timeline.add_event(event)
    
    print(f"GraphIR:    {graph}")
    print(f"ScheduleIR: {schedule}")
    print(f"TimelineIR: {timeline}")
    
    print("\n✅ 类型关系测试通过!")


if __name__ == "__main__":
    # 运行所有测试
    test_graph_ir_construction()
    test_expand_pass()
    test_timeline_pass()
    test_full_pipeline()
    test_types_relationship()
    
    print("\n" + "="*70)
    print("所有测试通过! ✅")
    print("="*70)

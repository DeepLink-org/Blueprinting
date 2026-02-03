#!/usr/bin/env python3
"""IR Workflow Demo - 演示三层 IR 架构的工作流程.

这个脚本展示了 blueprinting 的三层 IR 架构:

    Layer 1: Graph IR (Block-level)  - 模型结构描述
    Layer 2: Schedule IR (Op-level)  - 执行计划
    Layer 3: Timeline IR (Event-level) - 时间线模拟

运行方式:
    python examples/ir_workflow_demo.py

"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.blueprinting.ir.dsl import Model, Transformer
from src.blueprinting.ir.passes import (
    PrintGraphPass, PrintSchedulePass, PrintTimelinePass,
    ExpandPass, SchedulePass, TimelinePass,
    Pipeline,
)


def main():
    """主函数 - 演示完整的编译流水线."""
    print("=" * 80)
    print(" Blueprinting 三层 IR 架构演示")
    print("=" * 80)
    print("""
本演示展示 blueprinting 编译器的三层 IR 架构:

┌─────────────────────────────────────────────────────────────────────────────┐
│ Layer 1: Graph IR (Block-level) - 模型结构的高层抽象                          │
└─────────────────────────────────────────────────────────────────────────────┘
                            │
                            ↓ ExpandPass (Block → Op, 纯展开)
┌─────────────────────────────────────────────────────────────────────────────┐
│ Layer 2: Schedule IR (Op-level, 无 workload)                                │
└─────────────────────────────────────────────────────────────────────────────┘
                            │
                            ↓ SchedulePass (计算 workload + timing)
┌─────────────────────────────────────────────────────────────────────────────┐
│ Layer 2: Schedule IR (Op-level, 有 workload)                                │
└─────────────────────────────────────────────────────────────────────────────┘
                            │
                            ↓ TimelinePass (Op → Event)
┌─────────────────────────────────────────────────────────────────────────────┐
│ Layer 3: Timeline IR (Event-level) - 细粒度事件流，用于时间和内存模拟          │
└─────────────────────────────────────────────────────────────────────────────┘
    """)
    
    # ================================================================
    # Step 1: 使用 DSL 构建一个简单的两层模型
    # ================================================================
    print("\n【Step 1】使用 DSL 构建模型\n")
    
    with Transformer("simple_model") as m:
        m.metadata(batch_size=1, seq_len=512, hidden=1024)
        
        # Layer 0: Attention + FFN
        with m.TransformerLayer("layer0") as layer:
            with layer.Attention("attn") as attn:
                attn.RMSNorm("norm", normalized_shape=1024)
                attn.Linear("qkv", in_features=1024, out_features=3072)
                attn.Linear("out", in_features=1024, out_features=1024)
            
            with layer.FFN("ffn") as ffn:
                ffn.Linear("up", in_features=1024, out_features=2048)
                ffn.Linear("down", in_features=2048, out_features=1024)
        
        # Layer 1: Attention + FFN
        with m.TransformerLayer("layer1") as layer:
            with layer.Attention("attn") as attn:
                attn.RMSNorm("norm", normalized_shape=1024)
                attn.Linear("qkv", in_features=1024, out_features=3072)
                attn.Linear("out", in_features=1024, out_features=1024)
            
            with layer.FFN("ffn") as ffn:
                ffn.Linear("up", in_features=1024, out_features=2048)
                ffn.Linear("down", in_features=2048, out_features=1024)
    
    graph = m.build()
    
    print(f"  模型名称: {graph.name}")
    print(f"  Block 总数: {graph.count_blocks()}")
    print(f"  元数据: {graph.metadata}")
    
    # ================================================================
    # Step 2: 执行编译流水线
    # ================================================================
    print("\n\n【Step 2】执行编译流水线: Graph → Expand → Schedule → Timeline\n")
    
    pipeline = Pipeline([
        PrintGraphPass("Graph IR (Block 级别)"),
        ExpandPass(),
        SchedulePass(),  # 新增: 计算 workload 和 timing
        PrintSchedulePass("Schedule IR (Op 级别)", max_ops=20),
        TimelinePass(),
        PrintTimelinePass("Timeline IR (Event 级别)", max_events=25),
    ])
    
    print(f"Pipeline: {pipeline}\n")
    
    # 执行
    result = pipeline.run(graph)
    
    # ================================================================
    # Summary
    # ================================================================
    print("=" * 80)
    print(" 执行完成")
    print("=" * 80)
    print(f"  输入: GraphIR ({graph.count_blocks()} blocks)")
    print(f"  输出: TimelineIR ({len(result.events)} events)")
    print(f"  模拟总时间: {result.end_time*1e6:.1f} μs")
    print()


if __name__ == "__main__":
    main()

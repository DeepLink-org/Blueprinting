"""LLM 训练计算器 - 块粒度视图页面"""

import streamlit as st
import hyperparameter as hp
from sympy import Symbol

from blueprinting.ui import (
    setup_page,
    setup_sidebar,
    page_header_with_config,
    page_title,
    section_header,
    info_card,
    metrics_row,
)
from blueprinting.st import attention_block, block, ffn_block
from blueprinting.ir import Compiler, WorkloadPass, ParallelPass, SchedulePass, EvaluatePass
from blueprinting.ir.builder import build_transformer_model


# ============================================================================
# 页面初始化
# ============================================================================
setup_page(title="LLM训练计算器 - 块粒度")
app_json, sys_json, exe_json = setup_sidebar()

page_title(
    "块粒度分析",
    subtitle="Transformer 块级结构分析和 IR 编译模拟",
    icon="🧱"
)


# ============================================================================
# 主要超参配置
# ============================================================================
with hp.scope(app=app_json, model=app_json, sys=sys_json, exe=exe_json) as ps:
    config = page_header_with_config("主要超参", ps, mbs_param="exe.micro_batch_size")
    use_humanreadable = config["use_humanreadable"]
    use_raw_output = config["use_raw_output"]

    # ========================================================================
    # 块粒度视图
    # ========================================================================
    section_header("Transformer 块结构", "可视化 Transformer 块的内部组件")
    
    with st.expander("⚙️ 当前配置", expanded=False):
        st.json(ps.storage().storage())
    
    with block("transformer_block", 6):
        st.markdown("### 🔷 Transformer Block")
        ffn_block()
        attention_block()

    st.markdown("---")

    # ========================================================================
    # IR 编译器模拟
    # ========================================================================
    section_header("IR 编译器模拟", "基于计算图中间表示的性能估算")
    
    info_card(
        "编译流程",
        "GraphIR → WorkloadPass → ParallelPass → SchedulePass → EvaluatePass → SimulationResult",
        icon="🔄"
    )
    
    # 获取模型参数
    hidden = ps.model.hidden | 4096
    feedforward = ps.model.feedforward | (hidden * 4)
    num_heads = ps.model.attn_heads | 32
    head_dim = ps.model.attn_size | (hidden // num_heads)
    num_layers = ps.model.num_blocks | 32
    seq_len = ps.model.seq_size | 2048
    batch_size = ps.exe.microbatch_size | 1
    
    # 获取并行配置
    tp = ps.exe.tensor_par | 1
    pp = ps.exe.pipeline_par | 1
    dp = ps.exe.data_par | 1
    
    # 获取系统配置
    peak_tflops = ps.sys.matrix.float16.tflops | 312
    mem_bandwidth = ps.sys.mem1.GBps | 2000
    mem_capacity = ps.sys.mem1.GiB | 80
    
    # ========================================================================
    # 配置展示
    # ========================================================================
    col1, col2 = st.columns(2)
    
    with col1:
        with st.expander("📦 模型配置", expanded=True):
            st.markdown(f"""
            | 参数 | 值 |
            |------|-----|
            | Hidden | `{hidden}` |
            | FFN | `{feedforward}` |
            | Layers | `{num_layers}` |
            | Heads | `{num_heads}` |
            | Seq Len | `{seq_len}` |
            | Batch Size | `{batch_size}` |
            """)
    
    with col2:
        with st.expander("⚡ 系统配置", expanded=True):
            st.markdown(f"""
            | 参数 | 值 |
            |------|-----|
            | TP | `{tp}` |
            | PP | `{pp}` |
            | DP | `{dp}` |
            | Peak TFLOPS | `{peak_tflops}` |
            | Memory BW | `{mem_bandwidth} GB/s` |
            | Memory Cap | `{mem_capacity} GiB` |
            """)
    
    # ========================================================================
    # IR 编译器执行
    # ========================================================================
    st.markdown("")
    if st.button("🚀 运行 IR 编译器", key="run_ir_compiler", type="primary"):
        with st.spinner("构建计算图 IR..."):
            graph = build_transformer_model(
                num_layers=num_layers,
                hidden=Symbol("H"),
                feedforward=Symbol("FF"),
                num_heads=Symbol("heads"),
                head_dim=Symbol("head_dim"),
                seq_len=Symbol("S"),
                batch_size=Symbol("B"),
                tp=tp,
                pp=pp,
            )
            st.success(f"✅ 构建完成: {len(graph.nodes)} 个节点, {len(graph.edges)} 条边")
        
        with st.spinner("运行编译 Pass..."):
            subs = {
                "B": batch_size,
                "S": seq_len,
                "H": hidden,
                "FF": feedforward,
                "heads": num_heads,
                "head_dim": head_dim,
                "batch_seq": batch_size * seq_len,
                "num_layers": num_layers,
            }
            
            system_config = {
                "peak_tflops": peak_tflops,
                "memory_bandwidth_gbps": mem_bandwidth,
                "network_bandwidth_gbps": 400,
                "memory_capacity_gb": mem_capacity,
            }
            
            compiler = (
                Compiler()
                .add_pass(WorkloadPass())
                .add_pass(ParallelPass(tp=tp, pp=pp, dp=dp))
                .add_pass(SchedulePass(system_config=system_config))
                .add_pass(EvaluatePass(subs=subs, training=True))
            )
            
            try:
                result = compiler.compile(graph)
                st.success("✅ 编译完成!")
                
                # 显示结果
                st.markdown("#### 📊 模拟结果")
                
                metrics_row(
                    ("💾 峰值显存", f"{result.peak_memory / 1e9:.2f} GB"),
                    ("⏱️ 端到端时间", f"{result.e2e_time * 1e3:.2f} ms"),
                    ("📈 MFU", f"{result.mfu:.1%}"),
                    ("✓ 可行性", "✅ 可行" if result.is_feasible() else "❌ 不可行"),
                )
                
                # 详细分解
                col1, col2 = st.columns(2)
                with col1:
                    if result.memory_breakdown:
                        with st.expander("💾 显存分解", expanded=True):
                            st.metric("权重", f"{result.memory_breakdown.weights/1e9:.2f} GB")
                            st.metric("激活", f"{result.memory_breakdown.activations/1e9:.2f} GB")
                
                with col2:
                    if result.time_breakdown:
                        with st.expander("⏱️ 时间分解", expanded=True):
                            st.metric("前向", f"{result.time_breakdown.forward*1e3:.2f} ms")
                            st.metric("通信", f"{result.time_breakdown.communication*1e3:.2f} ms")
                
                # 警告信息
                if result.warnings:
                    for warning in result.warnings:
                        st.warning(warning)
                
                # 原始结果
                with st.expander("📋 原始结果 (JSON)", expanded=False):
                    st.json(result.to_dict())
                    
            except Exception as e:
                st.error(f"❌ 编译失败: {e}")
                import traceback
                with st.expander("错误详情"):
                    st.code(traceback.format_exc())

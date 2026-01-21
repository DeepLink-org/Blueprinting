"""LLM 训练计算器 - 总览页面"""

import json
import logging
from contextlib import nullcontext

import pandas as pd
import plotly.express as px
import streamlit as st
import hyperparameter as hp
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.row import row

from calculon.llm import Llm
from calculon.system import System
from blueprinting import Execution, Model
from blueprinting.ui import (
    human_readable_flops,
    human_readable_num,
    make_summary,
    setup_page,
    setup_sidebar,
    page_header_with_config,
    transformer_config_expander,
    raw_output_section,
    page_title,
    metrics_row,
    section_header,
    info_card,
)


# ============================================================================
# 页面初始化
# ============================================================================
setup_page(title="LLM训练计算器")
app_json, sys_json, exe_json = setup_sidebar()
logger = logging.getLogger()

# 页面标题
page_title(
    "LLM 训练计算器",
    subtitle="分析大语言模型训练的计算、通信和内存开销",
    icon="🧮"
)


# ============================================================================
# 主要超参配置
# ============================================================================
with hp.scope(app=app_json, model=app_json, sys=sys_json, exe=exe_json) as ps:
    config = page_header_with_config("主要超参", ps)
    use_humanreadable = config["use_humanreadable"]
    use_raw_output = config["use_raw_output"]

    # 编译模型
    app = Model.from_cfg(ps.app)
    exe = Execution(ps.exe)
    syst = System(sys_json)

    model = Llm(app, logger)
    model.compile(syst, exe)
    model.run(syst)


# ============================================================================
# 页面内容
# ============================================================================
tab_model, tab_detail = st.tabs(["📊 模型 Overview", "⚙️ 执行细节"])


# ============================================================================
# 模型 Overview Tab
# ============================================================================
with tab_model, ps:
    # Transformer 参数配置
    model_config = transformer_config_expander(ps)
    
    with hp.scope(**model_config) as ps:
        stats = model.get_stats_json(False)
        
        # 原始输出（可选）
        if use_raw_output:
            raw_output_section(stats, make_summary)

        # 整体性能指标 - 使用醒目的布局
        section_header("整体性能", "训练迭代时间和吞吐量")
        
        tgs = (
            app.seq_size
            * exe.global_batch_size
            / stats["total_time"]
            / (exe.tensor_par * exe.pipeline_par * exe.data_par)
        )
        
        metrics_row(
            ("⏱️ 迭代时间", "%.2f s" % stats["total_time"], None, "单次训练迭代所需时间"),
            ("🚀 TGS", "%.2f" % tgs, None, "每秒处理的 Token 数 (per GPU)"),
            ("🔢 总参数量", human_readable_num(app.nparam_total), None, "模型总参数量"),
            ("💻 计算量", human_readable_flops(app.flops_total * 3), None, "单次迭代总 FLOPs"),
        )
        
        st.markdown("")  # 间距

        # 参数量和计算量分析
        col1, col2 = st.columns(2)
        
        with col1:
            with st.expander("📦 参数量分析", expanded=True):
                st.dataframe(
                    pd.DataFrame.from_records([
                        {
                            "层级": "🔹 单块",
                            "Embedding": "-",
                            "Attention": app.nparam_attn,
                            "归一化": app.nparam_norm,
                            "FFN": app.nparam_mlp,
                            "合计": app.nparam_attn + app.nparam_norm + app.nparam_mlp,
                        },
                        {
                            "层级": "🔸 整体",
                            "Embedding": app.nparam_embedding,
                            "Attention": app.nparam_attn * app.num_blocks,
                            "归一化": app.nparam_norm * app.num_blocks,
                            "FFN": app.nparam_mlp * app.num_blocks,
                            "合计": app.nparam_total,
                        },
                    ]).style.format(
                        {col: human_readable_num for col in ["Embedding", "Attention", "归一化", "FFN", "合计"]}
                        if use_humanreadable else None
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

        with col2:
            with st.expander("⚡ 计算量分析 (FLOPs)", expanded=True):
                st.dataframe(
                    pd.DataFrame.from_records([
                        {
                            "层级": "🔹 单块",
                            "Embedding": "-",
                            "Attention": app.flops_attn,
                            "归一化": app.flops_norm,
                            "FFN": app.flops_mlp,
                            "合计": app.flops_attn + app.flops_norm + app.flops_mlp,
                        },
                        {
                            "层级": "🔸 整体×3",
                            "Embedding": 3 * app.flops_embedding,
                            "Attention": 3 * app.flops_attn * app.num_blocks,
                            "归一化": 3 * app.flops_norm * app.num_blocks,
                            "FFN": 3 * app.flops_mlp * app.num_blocks,
                            "合计": 3 * app.flops_total,
                        },
                    ]).style.format(
                        {col: human_readable_flops for col in ["Embedding", "Attention", "归一化", "FFN", "合计"]}
                        if use_humanreadable else None
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

        # 详细计算量表格（折叠）
        with st.expander("📋 详细计算量分解", expanded=False):
            st.dataframe(
                pd.DataFrame.from_records([
                    {"阶段": "块前向", "Embedding": 0.0, "Attention": app.flops_attn, "归一化": app.flops_norm, "FFN": app.flops_mlp},
                    {"阶段": "块反向", "Embedding": 0.0, "Attention": 2 * app.flops_attn, "归一化": 2 * app.flops_norm, "FFN": 2 * app.flops_mlp},
                    {"阶段": "整体前向", "Embedding": app.flops_embedding, "Attention": app.flops_attn * app.num_blocks, "归一化": app.flops_norm * app.num_blocks, "FFN": app.flops_norm * app.num_blocks, "总计": app.flops_total},
                    {"阶段": "整体反向", "Embedding": 2 * app.flops_embedding, "Attention": 2 * app.flops_attn * app.num_blocks, "归一化": 2 * app.flops_norm * app.num_blocks, "FFN": 2 * app.flops_norm * app.num_blocks, "总计": 2 * app.flops_total},
                ]).style.format(human_readable_flops if use_humanreadable else None),
                use_container_width=True,
                hide_index=True,
            )

        # 通信统计
        with st.expander("📡 通信量分析", expanded=False):
            st.caption("通信操作: all reduce / all gather / reduce scatter / send recv")
            comm_data = pd.DataFrame.from_records([
                {"阶段": "块前向", "Embedding": app.comm_embedding_fw, "Attention": app.comm_attn_fw, "FFN": app.comm_mlp_fw},
                {"阶段": "块反向", "Embedding": app.comm_embedding_bw, "Attention": app.comm_attn_bw, "FFN": app.comm_mlp_bw},
                {"阶段": "整体前向", "Embedding": app.comm_embedding_fw, "Attention": app.comm_attn_fw * app.num_blocks, "FFN": app.comm_mlp_fw * app.num_blocks, "总计": app.comm_total_fw * app.num_blocks + app.comm_pipeline_parallel_fw},
                {"阶段": "整体反向", "Embedding": app.comm_embedding_bw, "Attention": app.comm_attn_bw * app.num_blocks, "FFN": app.comm_mlp_bw * app.num_blocks, "总计": app.comm_total_bw * app.num_blocks + app.comm_pipeline_parallel_bw},
            ])
            st.dataframe(
                comm_data.map(lambda x: str(x).replace("\n", " | ") if isinstance(x, str) else x),
                use_container_width=True,
                hide_index=True,
            )


# ============================================================================
# 执行细节 Tab
# ============================================================================
with tab_detail:
    if use_raw_output:
        raw_output_section(stats, make_summary)

    stats_scope = hp.scope(**stats)

    section_header("阶段分解", "各训练阶段的 FLOPs 和显存访问量")

    def make_detail(da, title, expanded=True):
        """渲染详细信息表格和图表"""
        with st.expander(title, expanded=expanded):
            c1, c2 = st.columns([0.35, 0.65])
            with c1:
                st.dataframe(da, use_container_width=True, hide_index=True)
            with c2:
                fig = px.bar(
                    da, 
                    x="stage", 
                    y="value",
                    color="stage",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                fig.update_layout(
                    showlegend=False,
                    margin=dict(l=20, r=20, t=30, b=20),
                    xaxis_title="",
                    yaxis_title="",
                )
                st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    # FLOPs 详情
    with col1:
        make_detail(
            pd.DataFrame({
                "stage": ["前向", "激活梯度", "权重梯度", "优化器"],
                "value": [
                    stats_scope.block_fw_flops | 0,
                    stats_scope.block_agrad_flops | 0,
                    stats_scope.block_wgrad_flops | 0,
                    stats_scope.block_optim_flops | 0,
                ],
            }),
            "⚡ FLOPs 分布",
        )
    
    # 显存详情
    with col2:
        make_detail(
            pd.DataFrame({
                "stage": ["前向", "激活梯度", "权重梯度", "优化器"],
                "value": [
                    stats_scope.block_fw_mem_accessed | 0,
                    stats_scope.block_agrad_mem_accessed | 0,
                    stats_scope.block_wgrad_mem_accessed | 0,
                    stats_scope.block_optim_mem_accessed | 0,
                ],
            }),
            "💾 显存访问量",
        )

    # 提示信息
    info_card(
        "分析说明",
        "FLOPs 分布展示了各训练阶段的计算量，显存访问量反映了数据移动开销。优化器阶段通常占用较少的 FLOPs 但可能有较多的显存访问。",
        icon="💡"
    )

"""LLM 训练计算器 - 准确性校验页面"""

import streamlit as st

from blueprinting.ui import setup_page, page_title, section_header, info_card
from blueprinting.validations.cases.seqsel_fig1 import seqsel_fig1
from blueprinting.validations.cases.seqsel_fig7 import seqsel_fig7
from blueprinting.validations.cases.seqsel_tab5 import seqsel_tab5


# ============================================================================
# 页面初始化
# ============================================================================
setup_page(title="LLM训练计算器 - 准确性校验")

page_title(
    "模拟器准确性校验",
    subtitle="验证模拟结果与实际测量数据的一致性",
    icon="✅"
)


# ============================================================================
# 说明信息
# ============================================================================
info_card(
    "验证数据来源",
    "本页面展示模拟器与实际测量数据的对比验证结果。数据来源于 SeqSel 论文的公开数据集，用于评估模拟器的准确性。",
    icon="📊"
)

st.markdown("")


# ============================================================================
# SeqSel Figure 1 验证
# ============================================================================
with st.expander("📈 SeqSel Figure 1 - 基准吞吐量对比", expanded=True):
    st.markdown("**验证目标**: 基准配置下的吞吐量预测准确性")
    df_fig1 = seqsel_fig1(show=True)
    st.dataframe(df_fig1, hide_index=True, use_container_width=True)


# ============================================================================
# SeqSel Figure 7 验证
# ============================================================================
with st.expander("📊 SeqSel Figure 7 - 不同配置性能分析", expanded=True):
    st.markdown("**验证目标**: 不同并行配置下的性能预测")
    df_fig7 = seqsel_fig7(show=True)
    st.dataframe(df_fig7, hide_index=True, use_container_width=True)


# ============================================================================
# SeqSel Table 5 验证
# ============================================================================
with st.expander("📋 SeqSel Table 5 - 详细配置参数对比", expanded=True):
    st.markdown("**验证目标**: 详细配置参数的预测准确性")
    df_tab5 = seqsel_tab5(show=True)
    st.dataframe(df_tab5, hide_index=True, use_container_width=True)


# ============================================================================
# 验证说明
# ============================================================================
st.markdown("---")

section_header("验证方法说明")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    #### 📖 数据来源
    - 验证数据来自公开论文和测试结果
    - 包含多种模型规模和并行配置
    - 覆盖不同硬件环境
    """)

with col2:
    st.markdown("""
    #### 📏 准确性指标
    - 模拟结果与实测数据误差通常在 **5-10%** 以内
    - 主要用于趋势分析和配置优化
    - 适合相对比较，而非绝对预测
    """)

info_card(
    "使用建议",
    "模拟器主要用于快速评估不同配置的相对性能，帮助缩小搜索空间。实际部署前仍建议进行小规模实测验证。",
    icon="💡"
)

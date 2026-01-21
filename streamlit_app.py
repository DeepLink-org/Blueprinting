"""
XInsight Streamlit Application
==============================
异构计算模拟分析工具 - LLM 训练性能分析与浮点精度可视化

页面结构:
- LLM计算器
  - Overview: 模型总览和性能指标
  - 块粒度: Transformer 块级分析和 IR 编译
  - 分布式实验: TP/PP/DP 并行策略分析
  - 准确性校验: 模拟器验证结果

- 精度分析
  - 浮点精度: 浮点数格式和精度可视化
"""

import streamlit as st


# ============================================================================
# 页面导航配置
# ============================================================================
pg = st.navigation({
    "LLM计算器": [
        st.Page(
            "pages/LLM_Calc/overview.py",
            title="Overview",
            icon="📊",
            default=True,
        ),
        st.Page(
            "pages/LLM_Calc/blockwise.py",
            title="块粒度",
            icon="🧱",
        ),
        st.Page(
            "pages/LLM_Calc/distexp.py",
            title="分布式实验",
            icon="🔄",
        ),
        st.Page(
            "pages/LLM_Calc/validation.py",
            title="准确性校验",
            icon="✅",
        ),
    ],
    "精度分析": [
        st.Page(
            "pages/FloatAnalysis/float_precision.py",
            title="浮点精度",
            icon="🔢",
        ),
    ],
})


# ============================================================================
# 运行应用
# ============================================================================
pg.run()

"""Legacy Streamlit surface for Calculon and transitional Blueprinting pages.

The primary workbench is now launched with ``blueprinting-workbench``. Existing
Calculon pages remain available here as an independent baseline and are not
part of the Blueprinting product analysis path.
"""

import streamlit as st

# ============================================================================
# 页面导航配置
# ============================================================================
pg = st.navigation(
    {
        # The primary Blueprinting workflow stays directly visible. Secondary
        # and legacy tools become collapsed groups in Streamlit's top nav.
        "": [
            st.Page(
                "pages/Blueprinting/overview.py",
                title="分析总览",
                icon="🧭",
                url_path="blueprinting-overview",
                default=True,
            ),
            st.Page(
                "pages/Blueprinting/ir_audit.py",
                title="IR 推导审计",
                icon="🔬",
                url_path="blueprinting-ir-audit",
            ),
            st.Page(
                "pages/Blueprinting/strategy_explorer.py",
                title="策略空间探索",
                icon="🧩",
                url_path="blueprinting-strategy-explorer",
            ),
        ],
        "Calculon 基线（旧版）": [
            st.Page(
                "pages/LLM_Calc/overview.py",
                title="Overview",
                icon="📊",
                url_path="calculon-overview",
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
        ],
        "精度分析": [
            st.Page(
                "pages/FloatAnalysis/float_precision.py",
                title="浮点精度",
                icon="🔢",
            ),
        ],
    },
    position="top",
)


# ============================================================================
# 运行应用
# ============================================================================
pg.run()

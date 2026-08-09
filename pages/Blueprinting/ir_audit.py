"""Blueprinting derivation and IR audit workbench."""

import streamlit as st

from blueprinting.workbench.streamlit_ui import (
    analysis_form,
    cached_analysis,
    recalled,
    remember,
    render_ir_explorer,
    setup_workbench_page,
)

setup_workbench_page("IR 推导审计", "🔬")
st.caption(
    "逐层检查形式化推导边界、Verifier、lineage、内容摘要和 Portable task workload。"
    "Lowering 耗时表示分析器自身耗时，不代表硬件执行时间。"
)

STATE_KEY = "blueprinting.analysis.last_result"
draft = analysis_form("blueprinting.ir_audit", submit_label="生成并审计 IR")
if draft is not None:
    with st.spinner("正在生成 canonical IR checkpoints…"):
        remember(STATE_KEY, cached_analysis(draft))

outcome = recalled(STATE_KEY)
if outcome is None:
    st.info("运行一次分析后，这里会展示三个已实现的 IR 边界和完整 canonical snapshot。")
else:
    render_ir_explorer(outcome)

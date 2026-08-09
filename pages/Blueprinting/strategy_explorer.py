"""Blueprinting bounded parallel-strategy exploration workbench."""

import streamlit as st

from blueprinting.workbench.streamlit_ui import (
    cached_sweep,
    recalled,
    remember,
    render_sweep,
    setup_workbench_page,
    sweep_form,
)

setup_workbench_page("策略空间探索", "🧩")
st.caption(
    "批量推导 TP/PP/DP 候选，保留无效候选的结构化原因，并在可行解中标记延迟—内存 Pareto 前沿。"
)

STATE_KEY = "blueprinting.strategy_explorer.last_result"
request = sweep_form("blueprinting.strategy_explorer")
if request is not None:
    with st.spinner(f"正在分析 {request.candidate_count} 个候选…"):
        remember(STATE_KEY, cached_sweep(request))

report = recalled(STATE_KEY)
if report is None:
    st.info("从左侧选择候选集合。策略探索最多接受 128 个候选，并且不会丢弃失败点。")
else:
    render_sweep(report)

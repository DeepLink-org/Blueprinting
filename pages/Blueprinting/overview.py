"""Blueprinting single-point analysis workbench."""

import streamlit as st

from blueprinting.workbench.streamlit_ui import (
    analysis_form,
    cached_analysis,
    recalled,
    remember,
    render_analysis_summary,
    setup_workbench_page,
)

setup_workbench_page("分析总览", "🧭")
st.caption(
    "从模型语义和并行策略推导 PortablePlanIR，再以版本化硬件证据估算延迟与内存。"
    "Calculon 不参与这条产品分析路径。"
)

STATE_KEY = "blueprinting.analysis.last_result"
draft = analysis_form("blueprinting.overview")
if draft is not None:
    with st.spinner("正在进行形式化推导与证据估算…"):
        remember(STATE_KEY, cached_analysis(draft))

outcome = recalled(STATE_KEY)
if outcome is None:
    st.info("从左侧选择模型、硬件证据和执行策略，然后运行 Blueprinting 分析。")
    st.markdown(
        """
        这条工作台路径提供：

        - 类型化配置校验与可复现 request digest；
        - ModelIR → DistributedTaskIR → PortablePlanIR 推导；
        - 计算、访存、通信、Pipeline bubble 和单设备内存估算；
        - 明确的证据版本、实现边界与失败诊断。
        """
    )
else:
    render_analysis_summary(outcome)

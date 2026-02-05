"""LLM 训练计算器 - 块粒度视图页面"""

import streamlit as st
import hyperparameter as hp

from blueprinting.ui import (
    setup_page,
    setup_sidebar,
    page_header_with_config,
    page_title,
    section_header,
)
from blueprinting.st import attention_block, block, ffn_block


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

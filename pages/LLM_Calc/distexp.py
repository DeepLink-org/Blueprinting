"""LLM 训练计算器 - 分布式实验页面"""

import logging

import pandas as pd
import plotly.express as px
import streamlit as st
import hyperparameter as hp

from calculon.llm import Llm
from calculon.system import System
from blueprinting import Execution, Model
from blueprinting.ui import (
    setup_page,
    setup_sidebar,
    page_header_with_config,
    page_title,
    section_header,
    info_card,
)


# ============================================================================
# 页面初始化
# ============================================================================
setup_page(title="LLM训练计算器 - 分布式实验")
app_json, sys_json, exe_json = setup_sidebar()
logger = logging.getLogger()

page_title(
    "分布式实验",
    subtitle="探索不同 TP/PP/DP 并行配置对训练性能的影响",
    icon="🔄"
)


# ============================================================================
# 辅助函数
# ============================================================================
def gen_nums(rng):
    """生成范围内的 2 的幂次数列"""
    left, right = rng
    for i in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
        if left <= i <= right:
            yield i


def run_experiment(ps, app_json, sys_json, tp, pp, dp, logger):
    """运行单次实验，返回 TGS 或 None"""
    with hp.scope() as ps_exp:
        ps_exp.exe.tensor_par = tp
        ps_exp.exe.pipeline_par = pp
        ps_exp.exe.data_par = dp
        ps_exp.exe.num_procs = tp * pp * dp
        
        app = Model.from_cfg(ps.app)
        exe = Execution(ps_exp.exe)
        syst = System(sys_json)

        try:
            model = Llm(app, logger)
            model.compile(syst, exe)
            model.run(syst)

            stats = model.get_stats_json(False)
            tgs = (
                app.seq_size
                * exe.global_batch_size
                / stats["total_time"]
                / (exe.tensor_par * exe.pipeline_par * exe.data_par)
            )
            return tgs
        except Exception:
            return None


def plot_analysis(results, x_col, y_col, color_col, filter_col, filter_values, title_prefix):
    """绘制分析图表"""
    c1, c2 = st.columns(2)
    for i, val in enumerate(filter_values):
        filtered = results[results[filter_col] == val]
        col = c1 if i % 2 == 0 else c2
        with col:
            st.plotly_chart(
                px.line(
                    filtered,
                    x=x_col,
                    y=y_col,
                    color=color_col,
                    title=f"{title_prefix}[{filter_col}={val}]",
                ),
                use_container_width=True,
            )


# ============================================================================
# 主要超参配置
# ============================================================================
with hp.scope(app=app_json, model=app_json, sys=sys_json, exe=exe_json) as ps:
    config = page_header_with_config("主要超参", ps, use_slider=True)
    use_humanreadable = config["use_humanreadable"]
    use_raw_output = config["use_raw_output"]

    # ========================================================================
    # 运行实验
    # ========================================================================
    def gen_setups():
        """生成所有实验配置组合"""
        for tp in gen_nums(ps.exp.tp | [1, 1]):
            for pp in gen_nums(ps.exp.pp | [1, 1]):
                for dp in gen_nums(ps.exp.dp | [1, 1]):
                    yield tp, pp, dp

    results = []
    for tp, pp, dp in gen_setups():
        tgs = run_experiment(ps, app_json, sys_json, tp, pp, dp, logger)
        results.append({
            "tp": tp,
            "pp": pp,
            "dp": dp,
            "tgs": tgs,
        })
    
    results = pd.DataFrame.from_records(results)

    # ========================================================================
    # 实验数据展示
    # ========================================================================
    with st.expander("实验数据", expanded=True):
        st.dataframe(results, use_container_width=True)

    # ========================================================================
    # TP 分析
    # ========================================================================
    tp_range = ps.exp.tp | [1, 1]
    if tp_range[1] > tp_range[0]:
        with st.expander("TP分析", expanded=True):
            c1, c2 = st.columns(2)
            
            # 按 PP 分组
            with c1:
                for pp in gen_nums(ps.exp.pp | [1, 1]):
                    st.plotly_chart(
                        px.line(
                            results[results.pp == pp],
                            x="tp",
                            y="tgs",
                            color="dp",
                            title=f"TP分析[pp={pp}]",
                        ),
                        use_container_width=True,
                    )
            
            # 按 DP 分组
            with c2:
                for dp in gen_nums(ps.exp.dp | [1, 1]):
                    st.plotly_chart(
                        px.line(
                            results[results.dp == dp],
                            x="tp",
                            y="tgs",
                            color="pp",
                            title=f"TP分析[dp={dp}]",
                        ),
                        use_container_width=True,
                    )

    # ========================================================================
    # PP 分析
    # ========================================================================
    pp_range = ps.exp.pp | [1, 1]
    if pp_range[1] > pp_range[0]:
        with st.expander("PP分析", expanded=True):
            c1, c2 = st.columns(2)
            
            # 按 TP 分组
            with c1:
                for tp in gen_nums(ps.exp.tp | [1, 1]):
                    st.plotly_chart(
                        px.line(
                            results[results.tp == tp],
                            x="pp",
                            y="tgs",
                            color="dp",
                            title=f"PP分析[tp={tp}]",
                        ),
                        use_container_width=True,
                    )
            
            # 按 DP 分组
            with c2:
                for dp in gen_nums(ps.exp.dp | [1, 1]):
                    st.plotly_chart(
                        px.line(
                            results[results.dp == dp],
                            x="pp",
                            y="tgs",
                            color="tp",
                            title=f"PP分析[dp={dp}]",
                        ),
                        use_container_width=True,
                    )

    # ========================================================================
    # DP 分析
    # ========================================================================
    dp_range = ps.exp.dp | [1, 1]
    if dp_range[1] > dp_range[0]:
        with st.expander("DP分析", expanded=True):
            c1, c2 = st.columns(2)
            
            # 按 TP 分组
            with c1:
                for tp in gen_nums(ps.exp.tp | [1, 1]):
                    st.plotly_chart(
                        px.line(
                            results[results.tp == tp],
                            x="dp",
                            y="tgs",
                            color="pp",
                            title=f"DP分析[tp={tp}]",
                        ),
                        use_container_width=True,
                    )
            
            # 按 PP 分组
            with c2:
                for pp in gen_nums(ps.exp.pp | [1, 1]):
                    st.plotly_chart(
                        px.line(
                            results[results.pp == pp],
                            x="dp",
                            y="tgs",
                            color="tp",
                            title=f"DP分析[pp={pp}]",
                        ),
                        use_container_width=True,
                    )

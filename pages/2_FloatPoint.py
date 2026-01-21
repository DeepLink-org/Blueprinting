"""浮点精度分析页面"""

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_extras.row import row

from blueprinting.ui import setup_page
from blueprinting.fp import float_point_values_table
from blueprinting.fp.visualization import (
    plot_float_formats,
    plot_subnormal_distribution,
    plot_quantization_error,
    plot_fp16_precision_error,
    plot_basic_op_map,
    show_map,
    show_scatter,
)


# ============================================================================
# 页面初始化
# ============================================================================
setup_page(title="浮点精度分析")


# ============================================================================
# 浮点数格式定义
# ============================================================================
DEFAULT_FLOAT_FORMATS = {
    "fp32": (1, 8, 23),
    "tf32": (1, 8, 10),
    "bf16": (1, 8, 7),
    "fp16": (1, 5, 10),
    "fp8(E5M2)": (1, 5, 2),
    "fp8(E4M3)": (1, 4, 3),
    "fp4(E2M1)": (1, 2, 1),
    "custom": (1, 5, 2),
}


# ============================================================================
# 浮点数格式可视化
# ============================================================================
st.markdown("# 浮点数格式")

c1, c2 = st.columns([0.3, 0.7])

with c1.container():
    fp = st.data_editor(
        pd.DataFrame({
            "format": list(DEFAULT_FLOAT_FORMATS.keys()),
            "sign_bit": [bool(v[0]) for v in DEFAULT_FLOAT_FORMATS.values()],
            "exponent_bits": [v[1] for v in DEFAULT_FLOAT_FORMATS.values()],
            "mantissa_bits": [v[2] for v in DEFAULT_FLOAT_FORMATS.values()],
        }),
        column_config={
            "format": st.column_config.TextColumn("格式"),
            "sign_bit": st.column_config.CheckboxColumn("符号位"),
            "exponent_bits": st.column_config.NumberColumn(
                "指数位", min_value=0, max_value=8, step=1
            ),
            "mantissa_bits": st.column_config.NumberColumn(
                "精度位", min_value=1, max_value=23, step=1
            ),
        },
    )

# 从编辑器获取更新后的格式
float_formats = {
    row["format"]: (
        row["sign_bit"],
        row["exponent_bits"],
        row["mantissa_bits"],
    )
    for _, row in fp.iterrows()
}

with c2.container():
    st.pyplot(plot_float_formats(float_formats))


# ============================================================================
# 自定义浮点数计算器
# ============================================================================
exp_len = int(float_formats["custom"][1])
man_len = int(float_formats["custom"][2])
fp_name = f"FP{1 + exp_len + man_len}(E{exp_len}M{man_len})"

st.markdown(f"# {fp_name} 计算器")

c1, c2, c3 = st.columns(3)

# 符号位输入
with c1.expander("符号位", expanded=True):
    sign_val = st.checkbox("符号位", False, label_visibility="collapsed")

# 指数位输入
with c2.expander("指数位", expanded=True):
    r = row(exp_len)
    exp_val = [
        r.checkbox("x", False, label_visibility="collapsed", key=f"e{i}")
        for i in range(exp_len)
    ]

# 精度位输入
with c3.expander("精度位", expanded=True):
    r = row(man_len)
    man_val = [
        r.checkbox("x", False, label_visibility="collapsed", key=f"m{i}")
        for i in range(man_len)
    ]

# 计算结果显示
c1, c2, c3 = st.columns(3)

with c1:
    st.metric("符号位", "%d" % sign_val)

with c2:
    exponent = 0
    for x in exp_val:
        exponent = (exponent << 1) + x
    st.metric("指数位", "%d" % exponent)

with c3:
    mantissa = 0 if exponent == 0 else 1
    for i, x in enumerate(man_val):
        mantissa += float(x) / (2 ** (i + 1))
    st.metric("精度位", "%f" % mantissa)

# 计算公式展示
bias = 2 ** (exp_len - 1) - 1
result_value = ((-1) ** sign_val) * 2 ** (exponent - bias) * mantissa

st.markdown(
    r"""
$$
(-1)^{s=%d} \times 2^{e=%d-bias=%d} \times (%d+\frac{%d}{2^1}+\frac{%d}{2^2}+\cdots+\frac{%d}{2^{%d}}) = %f
$$
"""
    % (
        sign_val,
        exponent,
        bias,
        0 if exponent == 0 else 1,
        int(man_val[0]) if man_val else 0,
        int(man_val[1]) if len(man_val) > 1 else 0,
        int(man_val[-1]) if man_val else 0,
        len(man_val),
        result_value,
    )
)

# 特殊值计算
subnormal = 2 ** (1 - bias) * 1
zeroed = 2 ** (1 - bias) * (1 / (2 ** man_len))

r = row(2)
r.metric("subnormal", subnormal)
r.metric("zeroed", zeroed)

st.write("参考：https://flop.evanau.dev/half-precision-converter")


# ============================================================================
# 精度分析
# ============================================================================
st.markdown(f"# {fp_name} 精度分析")

# 生成浮点数值表
fp_values = float_point_values_table(
    sign_bit=True, exponent_bits=exp_len, mantissa_bits=man_len
)

# Subnormal 分析
with st.expander(f"{fp_name} subnormal分析", expanded=True):
    rng = st.select_slider(
        "可视化范围", options=[0.001, 0.01, 0.1, 1.0, 10, 100, 1000], value=0.1
    )
    
    st.pyplot(plot_subnormal_distribution(fp_values, subnormal, rng), use_container_width=True)
    
    st.markdown(
        r"""
    红色表示subnormal数值，蓝色表示正常数值，误差图说明：
    1. 上图蓝色曲线为数轴上各点的相对量化误差($\frac{|数值-量化数值|}{数值}$);
    2. 红色区间标识了subnormal number所在的区间；
    3. 绿色曲线标识了FP8无法表示的区间，这个区间内的数值被量化成0；
    """
    )
    
    st.pyplot(plot_quantization_error(fp_values, subnormal, rng))

# 量化误差分析
with st.expander(f"{fp_name} 量化误差分析", expanded=True):
    r = row(2)
    rng2 = r.select_slider(
        "可视化范围",
        options=[0.001, 0.01, 0.1, 1.0, 10, 100, 1000],
        value=0.1,
        key="_rng2",
    )
    spl = r.select_slider("采样间隔", options=[1, 10, 100, 1000], value=10, key="_spl2")
    
    st.pyplot(plot_fp16_precision_error(rng2, spl))

# 运算误差分析
with st.expander(f"{fp_name} 运算误差分析", expanded=True):
    r = row(2)
    r.text("四则运算对精度的影响")
    limit = r.select_slider(
        "可视化范围",
        options=[0.001, 0.01, 0.1, 1.0, 10, 100, None],
        value=None,
        key="_rng3",
    )
    
    r = row(2)
    
    # 热图可视化
    r.pyplot(
        plot_basic_op_map(
            fp_values,
            show_map,
            maxval=max(fp_values),
            subnormal=subnormal,
            zeroed=zeroed,
            traced_values=[1, 4, 16, 256],
            limit=limit,
        )
    )
    
    # 散点图可视化
    r.pyplot(
        plot_basic_op_map(
            fp_values,
            show_scatter,
            fp_values=fp_values,
            maxval=max(fp_values),
            subnormal=subnormal,
            zeroed=zeroed,
            traced_values=[1, 4, 16, 256],
            limit=limit,
        )
    )

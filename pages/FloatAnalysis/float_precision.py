"""浮点精度分析页面"""

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_extras.row import row

from blueprinting.ui import (
    setup_page,
    page_title,
    section_header,
    info_card,
    metrics_row,
)
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

page_title(
    "浮点精度分析",
    subtitle="探索不同浮点数格式的表示范围、精度和误差特性",
    icon="🔢"
)


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
section_header("浮点数格式对比", "比较不同浮点格式的位宽分配")

c1, c2 = st.columns([0.35, 0.65])

with c1:
    st.markdown("##### 📝 格式编辑器")
    fp = st.data_editor(
        pd.DataFrame({
            "format": list(DEFAULT_FLOAT_FORMATS.keys()),
            "sign_bit": [bool(v[0]) for v in DEFAULT_FLOAT_FORMATS.values()],
            "exponent_bits": [v[1] for v in DEFAULT_FLOAT_FORMATS.values()],
            "mantissa_bits": [v[2] for v in DEFAULT_FLOAT_FORMATS.values()],
        }),
        column_config={
            "format": st.column_config.TextColumn("格式", width="small"),
            "sign_bit": st.column_config.CheckboxColumn("符号", width="small"),
            "exponent_bits": st.column_config.NumberColumn(
                "指数位", min_value=0, max_value=8, step=1, width="small"
            ),
            "mantissa_bits": st.column_config.NumberColumn(
                "尾数位", min_value=1, max_value=23, step=1, width="small"
            ),
        },
        use_container_width=True,
        hide_index=True,
    )
    
    st.caption("💡 修改 custom 行来自定义浮点格式")

# 从编辑器获取更新后的格式
float_formats = {
    r["format"]: (r["sign_bit"], r["exponent_bits"], r["mantissa_bits"])
    for _, r in fp.iterrows()
}

with c2:
    st.markdown("##### 📊 位图布局")
    st.pyplot(plot_float_formats(float_formats), use_container_width=True)


st.markdown("---")


# ============================================================================
# 自定义浮点数计算器
# ============================================================================
exp_len = int(float_formats["custom"][1])
man_len = int(float_formats["custom"][2])
fp_name = f"FP{1 + exp_len + man_len}(E{exp_len}M{man_len})"

section_header(f"{fp_name} 交互计算器", "通过勾选位来构造浮点数并查看其值")

# 位输入区域
col1, col2, col3 = st.columns([1, 2, 3])

with col1:
    st.markdown("##### 符号位")
    sign_val = st.checkbox("S", False, help="符号位：0=正数, 1=负数")

with col2:
    st.markdown("##### 指数位")
    exp_cols = st.columns(exp_len)
    exp_val = [
        exp_cols[i].checkbox(f"E{i}", False, key=f"e{i}")
        for i in range(exp_len)
    ]

with col3:
    st.markdown("##### 尾数位")
    man_cols = st.columns(man_len)
    man_val = [
        man_cols[i].checkbox(f"M{i}", False, key=f"m{i}")
        for i in range(man_len)
    ]

# 计算结果
exponent = sum(x << (exp_len - 1 - i) for i, x in enumerate(exp_val))
mantissa = (0 if exponent == 0 else 1) + sum(float(x) / (2 ** (i + 1)) for i, x in enumerate(man_val))
bias = 2 ** (exp_len - 1) - 1
result_value = ((-1) ** sign_val) * 2 ** (exponent - bias) * mantissa

# 显示计算结果
st.markdown("")
metrics_row(
    ("符号", "-" if sign_val else "+", None, f"sign bit = {int(sign_val)}"),
    ("指数", str(exponent), None, f"raw exponent = {exponent}, biased = {exponent - bias}"),
    ("尾数", f"{mantissa:.4f}", None, f"1 + fraction = {mantissa}"),
    ("数值", f"{result_value:.6g}", None, "最终计算结果"),
)

# 计算公式
with st.expander("📐 计算公式", expanded=False):
    st.latex(
        r"(-1)^{%d} \times 2^{%d-%d} \times %.4f = %.6g"
        % (sign_val, exponent, bias, mantissa, result_value)
    )

# 特殊值
subnormal = 2 ** (1 - bias) * 1
zeroed = 2 ** (1 - bias) * (1 / (2 ** man_len))

col1, col2, col3 = st.columns(3)
col1.metric("Subnormal 阈值", f"{subnormal:.2e}", help="小于此值的数为 subnormal")
col2.metric("最小精度", f"{zeroed:.2e}", help="最小可表示的非零值")
col3.markdown(f"[📚 参考资料](https://flop.evanau.dev/half-precision-converter)")


st.markdown("---")


# ============================================================================
# 精度分析
# ============================================================================
section_header(f"{fp_name} 精度分析", "可视化浮点数的分布和误差特性")

# 生成浮点数值表
fp_values = float_point_values_table(
    sign_bit=True, exponent_bits=exp_len, mantissa_bits=man_len
)

# Subnormal 分析
with st.expander("🔍 Subnormal 数值分布", expanded=True):
    info_card(
        "什么是 Subnormal？",
        "Subnormal（次正规数）是指数全为 0 时的特殊浮点数，用于表示接近 0 的极小值。它们的精度较低但可以避免下溢。",
        icon="📖"
    )
    
    rng = st.select_slider(
        "📏 可视化范围",
        options=[0.001, 0.01, 0.1, 1.0, 10, 100, 1000],
        value=0.1,
        help="调整数轴显示范围"
    )
    
    st.pyplot(plot_subnormal_distribution(fp_values, subnormal, rng), use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **图例说明**
        - 🔴 **红色** - Subnormal 数值
        - 🔵 **蓝色** - 正常数值
        """)
    with col2:
        st.markdown("""
        **观察要点**
        - Subnormal 区间内数值分布稀疏
        - 接近 0 时精度逐渐降低
        """)
    
    st.pyplot(plot_quantization_error(fp_values, subnormal, rng))

# 量化误差分析
with st.expander("📈 量化误差分析", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        rng2 = st.select_slider(
            "可视化范围",
            options=[0.001, 0.01, 0.1, 1.0, 10, 100, 1000],
            value=0.1,
            key="_rng2",
        )
    with col2:
        spl = st.select_slider(
            "采样间隔",
            options=[1, 10, 100, 1000],
            value=10,
            key="_spl2",
            help="较大的间隔可加速绘图"
        )
    
    st.pyplot(plot_fp16_precision_error(rng2, spl))
    
    st.caption("**atol** = 绝对误差，**rtol** = 相对误差")

# 运算误差分析
with st.expander("🧮 四则运算精度影响", expanded=False):
    info_card(
        "运算如何影响精度？",
        "浮点运算可能导致结果超出表示范围（溢出）、进入 subnormal 区域（下溢）或被截断为零。下图展示了两个浮点数进行四则运算后的结果分布。",
        icon="⚠️"
    )
    
    col1, col2 = st.columns([0.7, 0.3])
    with col2:
        limit = st.select_slider(
            "可视化范围",
            options=[0.001, 0.01, 0.1, 1.0, 10, 100, None],
            value=None,
            key="_rng3",
            format_func=lambda x: "全部" if x is None else str(x)
        )
    
    st.markdown("##### 热图视图")
    st.pyplot(
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
    
    st.markdown("##### 散点图视图")
    st.pyplot(
        plot_basic_op_map(
            fp_values,
            show_scatter,
            need_fp_values=True,
            maxval=max(fp_values),
            subnormal=subnormal,
            zeroed=zeroed,
            traced_values=[1, 4, 16, 256],
            limit=limit,
        )
    )
    
    st.markdown("""
    **颜色说明**
    - 🟢 **绿色** - 正常范围
    - 🔴 **红色** - 溢出（超出最大值）
    - 🔵 **蓝色** - Subnormal 区域
    - ⚫ **黑色** - 下溢为零
    - 🟣 **紫色** - 追踪的特殊值 (1, 4, 16, 256)
    """)

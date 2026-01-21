"""UI utilities for blueprinting - Streamlit 共享组件和工具函数"""

import glob
import hmac
import json
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Union, Callable, Optional, Any

import pandas as pd
import streamlit as st
import hyperparameter as hp
from streamlit_extras.row import row as st_row

from . import io


# ============================================================================
# 页面配置和初始化
# ============================================================================


def check_password():
    """Returns `True` if the user had a correct password."""

    def login_form():
        """Form with widgets to collect user information"""
        with st.form("Credentials"):
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            st.form_submit_button("Log in", on_click=password_entered)

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["username"] in st.secrets["passwords"] and hmac.compare_digest(
            st.session_state["password"],
            st.secrets.passwords[st.session_state["username"]],
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the username or password.
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    # Return True if the username + password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show inputs for username + password.
    login_form()
    if "password_correct" in st.session_state:
        st.error("😕 User not known or password incorrect")
    return False


def ele(name, *args, scope=None, param=None, **kwargs):
    return (name, scope, param, args, kwargs)


def row(*args):
    columns = st.columns([1 for _ in args], vertical_alignment="bottom")
    for element, container in zip(args, columns):
        with container:
            value = getattr(st, element[0])(*element[3], **element[4])
            if element[1] is not None and element[2] is not None:
                setattr(element[1], element[2], value)


class Predefined:
    @property
    def models(self):
        return glob.glob("data/models/*.json")

    @property
    def model_names(self):
        return [x.replace("data/models/", "") for x in self.models]

    @property
    def systems(self):
        return glob.glob("data/systems/*.json")

    @property
    def system_names(self):
        return [x.replace("data/systems/", "") for x in self.systems]

    @property
    def executions(self):
        return glob.glob("data/examples/*.json")

    @property
    def execution_names(self):
        return [x.replace("data/examples/", "") for x in self.executions]


predefined = Predefined()


def value2widget(key, value, prefix=None):
    if prefix:
        key = f"{prefix}.{key}"
    if isinstance(value, bool):
        return st.checkbox(key, value=value)
    if isinstance(value, (int, float)):
        return st.number_input(key, value=value)
    if isinstance(value, str):
        return st.text_input(key, value=value)
    if isinstance(value, dict):
        return {k: value2widget(k, v, prefix=key) for k, v in value.items()}
    if isinstance(value, list):
        return [value2widget(str(k), v, prefix=key) for k, v in enumerate(value)]
    return value


def make_json_conf(conf):
    if isinstance(conf, str):
        conf = io.read_json_file(conf)
    return {k: value2widget(k, v) for k, v in conf.items()}


LEVELS = {"block": "块"}
STAGES = {
    "fw": "前向",
    "agrad": "激活梯度",
    "wgrad": "权重梯度",
    "optim": "优化器",
    "re": "重计算",
}
CATEGORIES = {
    "flops": "flops",
    "flops_time": "flops时间",
    "mem_accessed": "显存占用",
    "mem_time": "访存时间",
    "time": "耗时",
}

PARSER = {
    f"{l}_{s}_{c}": (lv, sv, cv) for l, lv in LEVELS.items() for s, sv in STAGES.items() for c, cv in CATEGORIES.items()
}

PARSER.update(
    {
        "fw_time": ("整体", "前向", "耗时"),
        "bw_time": ("整体", "反向", "耗时"),
        "optim_step_time": ("整体", "优化器", "耗时"),
        "recompute_time": ("整体", "重计算", "耗时"),
        "recomm_exposed_time": ("整体", "重计算通信", "耗时"),
        "bubble_time": ("整体", "空泡", "耗时"),
        "total_time": ("整体", "整体", "耗时"),
        "compute_efficiency": ("整体", "计算效率", "-"),
        "system_efficiency": ("整体", "系统效率", "-"),
        "total_efficiency": ("整体", "总效率", "-"),
        "sample_rate": ("整体", "采样率", "-"),
    }
)


@dataclass
class result:
    level: str
    stage: str
    category: str
    name: str
    value: Union[str, int, float]

    @staticmethod
    def parse(key, val):
        l, s, c = PARSER.get(key, (None, None, None))
        return result(l, s, c, key, val)


def human_format(num, round_to=1, use_tera=False):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num = round(num / 1000.0, round_to)
    formatter = ["", "K", "M", "G", "T"] if use_tera else ["", "K", "M", "B", "G"]
    return "{:.{}f}{}".format(num, round_to, formatter[magnitude])


def human_readable(num, round_to=1, formatter=None):
    if isinstance(num, (str,)):
        return num
    for factor, suffix in formatter.items():
        if num >= factor:
            return "{:.{}f}{}".format(num / factor, round_to, suffix)
    return "{:.{}f}".format(num, round_to)


def human_readable_flops(num, round_to=1):
    if not isinstance(num, (int, float)):
        return num
    return human_readable(
        num,
        round_to,
        formatter={
            1e15: "P",
            1e12: "T",
            1e6: "M",
            1e3: "K",
        },
    )


def human_readable_mem(num, round_to=1):
    return human_readable(
        num,
        round_to,
        formatter={
            1e12: "T",
            1e9: "G",
            1e6: "M",
            1e3: "K",
        },
    )


def human_readable_num(num, round_to=1):
    if not isinstance(num, (int, float)):
        return num
    return human_readable(
        num,
        round_to,
        formatter={
            1e12: "T",
            1e9: "B",
            1e6: "M",
            1e3: "K",
        },
    )


def human_readable_time(num, round_to=2):
    if num >= 1:
        return "{:.{}f} s".format(num, round_to)
    if num >= 1e-3:
        return "{:.{}f} ms".format(num * 1000, round_to)
    if num >= 1e-6:
        return "{:.{}f} ms".format(num * 1e6, round_to)


def make_summary(stats):
    data = pd.DataFrame(result.parse(k, v) for k, v in stats.items())
    summary = pd.pivot_table(data, index=["level", "stage"], columns="category", values="value")
    summary = summary.reindex(
        [
            "前向",
            "反向",
            "激活梯度",
            "权重梯度",
            "重计算",
            "重计算通信",
            "优化器",
            "空泡",
            "整体",
            "计算效率",
            "系统效率",
            "总效率",
            "采样率",
        ],
        level=1,
    )
    return summary.style.format({"flops": human_readable_flops, "显存占用": human_readable_mem})


class HumanReadableFormatter:
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __mod__(self, other):
        return self.func(other)


ReadableFLOPs = HumanReadableFormatter(human_readable_flops)
ReadableMem = HumanReadableFormatter(human_readable_mem)
ReadableNum = HumanReadableFormatter(human_readable_num)
ReadableTime = HumanReadableFormatter(human_readable_time)
NullFormatter = HumanReadableFormatter(lambda x: x)


def setup_sidebar():
    """设置公共侧边栏配置，返回 (app_json, sys_json, exe_json)"""
    st.sidebar.title("参数配置")
    with st.sidebar:
        tabm, tabs, tabe = st.tabs(["模型", "系统", "执行"])

        with tabm:
            app_json = st.selectbox("模型配置", predefined.model_names)
            app_json = make_json_conf(f"data/models/{app_json}")
        with tabs:
            sys_json = st.selectbox("系统配置", predefined.system_names)
            sys_json = make_json_conf(f"data/systems/{sys_json}")
        with tabe:
            exe_json = st.selectbox("执行参数", predefined.execution_names)
            exe_json = make_json_conf(f"data/examples/{exe_json}")

    return app_json, sys_json, exe_json


def setup_page(title="LLM训练计算器", icon=":eyeglasses:", layout="wide"):
    """设置页面配置"""
    st.set_page_config(page_title=title, page_icon=icon, layout=layout)


# ============================================================================
# 共享 UI 组件 - 可复用的页面元素
# ============================================================================

def parallel_config_row(
    ps: hp.scope,
    show_gbs: bool = True,
    show_mbs: bool = True,
    show_seqlen: bool = True,
    use_slider: bool = False,
    mbs_param: str = "exe.microbatch_size",
) -> None:
    """渲染并行配置行 (TP/PP/DP/GBS/MBS/SeqLen)
    
    Args:
        ps: hyperparameter scope 对象
        show_gbs: 是否显示 global batch size
        show_mbs: 是否显示 micro batch size
        show_seqlen: 是否显示 sequence length
        use_slider: 是否使用滑块（用于范围选择实验）
        mbs_param: microbatch size 的参数路径
    """
    if use_slider:
        tp_options = [1, 2, 4, 8, 16]
        pp_options = [1, 2, 4, 8, 16]
        dp_options = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        row(
            ele("select_slider", "TP", value=(1, 1), options=tp_options, scope=ps, param="exp.tp"),
            ele("select_slider", "PP", value=(1, 1), options=pp_options, scope=ps, param="exp.pp"),
            ele("select_slider", "DP", value=(1, 1), options=dp_options, scope=ps, param="exp.dp"),
            ele("text", ""),
            ele("number_input", "gbs", value=ps.exe.batch_size | 1, scope=ps, param="exe.batch_size") if show_gbs else ele("text", ""),
            ele("number_input", "mbs", value=ps.exe.microbatch_size | 1, scope=ps, param=mbs_param) if show_mbs else ele("text", ""),
            ele("text", ""),
            ele("number_input", "seqlen", value=ps.model.seq_size | 1, scope=ps, param="model.seq_size") if show_seqlen else ele("text", ""),
        )
    else:
        dp_value = int(
            (ps.exe.num_procs | 1) / 
            (ps.exe.tensor_par | 1) / 
            (ps.exe.pipeline_par | 1)
        )
        row(
            ele("number_input", "TP", value=ps.exe.tensor_par | 1, scope=ps, param="exe.tensor_par"),
            ele("number_input", "PP", value=ps.exe.pipeline_par | 1, scope=ps, param="exe.pipeline_par"),
            ele("number_input", "DP", value=dp_value, scope=ps, param="exe.data_par"),
            ele("text", ""),
            ele("number_input", "gbs", value=ps.exe.batch_size | 1, scope=ps, param="exe.batch_size") if show_gbs else ele("text", ""),
            ele("number_input", "mbs", value=ps.exe.microbatch_size | 1, scope=ps, param=mbs_param) if show_mbs else ele("text", ""),
            ele("text", ""),
            ele("number_input", "seqlen", value=ps.model.seq_size | 1, scope=ps, param="model.seq_size") if show_seqlen else ele("text", ""),
        )


def config_popover(ps: hp.scope, label: str = "配置") -> dict:
    """渲染配置弹窗
    
    Args:
        ps: hyperparameter scope 对象
        label: 弹窗按钮标签
        
    Returns:
        包含配置选项的字典
    """
    config = {}
    with st.popover(label, use_container_width=True):
        config["use_humanreadable"] = st.checkbox("显示人类可读数字", True)
        config["use_raw_output"] = st.checkbox("显示原始输出", False)
        config["count_add"] = st.checkbox("FLOPS统计含加法")
        ps.blueprinting.flops.count_add = config["count_add"]
    return config


def page_header_with_config(
    title: str,
    ps: hp.scope,
    use_slider: bool = False,
    mbs_param: str = "exe.microbatch_size",
) -> dict:
    """渲染页面头部，包含并行配置和配置弹窗
    
    Args:
        title: 页面标题
        ps: hyperparameter scope 对象
        use_slider: 是否使用滑块模式
        mbs_param: microbatch size 参数路径
        
    Returns:
        配置字典
    """
    st.header(title, divider=True)
    c1, c2 = st.columns([0.9, 0.1], vertical_alignment="bottom")
    
    with c1:
        parallel_config_row(ps, use_slider=use_slider, mbs_param=mbs_param)
        if not use_slider:
            ps.exe.num_proc = (
                (ps.exe.tensor_par | 1) * 
                (ps.exe.pipeline_par | 1) * 
                (ps.exe.data_par | 1)
            )
    
    with c2:
        config = config_popover(ps)
    
    return config


def transformer_config_expander(ps: hp.scope) -> dict:
    """Transformer 模型参数配置展开区
    
    Args:
        ps: hyperparameter scope 对象
        
    Returns:
        配置字典
    """
    config = {}
    with st.expander("transformer参数", expanded=True):
        r = st_row(3)
        bias_flags = r.container()
        emb_flags = r.container()
        norm_flags = r.container()
        
        config.update({
            "model.use_attn_bias": bias_flags.checkbox("attention投影bias", True),
            "model.use_qkv_bias": bias_flags.checkbox("attention输入bias", True),
            "model.use_mlp_bias": bias_flags.checkbox("MLP使用bias", True),
            "model.type_posemb": emb_flags.selectbox("位置编码", ["learned", "rope"], 0),
            "model.vocab_size": emb_flags.number_input("词表大小", ps.model.vocab_size | 51200),
            "model.type_norm": norm_flags.selectbox("归一化层", ["LN", "RMS"], 0),
        })
    
    return config


def raw_output_section(stats: dict, summary_fn: Callable = None) -> None:
    """渲染原始输出区域
    
    Args:
        stats: 统计数据字典
        summary_fn: 摘要生成函数
    """
    with st.expander("模拟器日志", expanded=False):
        st.text(json.dumps(stats, indent=2))
    
    if summary_fn:
        with st.expander("模拟器输出", expanded=False):
            st.dataframe(summary_fn(stats), use_container_width=True)


# ============================================================================
# 布局组件 - 页面结构和视觉元素
# ============================================================================

def page_title(title: str, subtitle: str = None, icon: str = None):
    """渲染页面标题区域
    
    Args:
        title: 主标题
        subtitle: 副标题/描述
        icon: 图标 emoji
    """
    if icon:
        st.markdown(f"# {icon} {title}")
    else:
        st.markdown(f"# {title}")
    
    if subtitle:
        st.caption(subtitle)
    
    st.markdown("---")


def section_header(title: str, description: str = None):
    """渲染章节标题
    
    Args:
        title: 章节标题
        description: 章节描述
    """
    st.markdown(f"### {title}")
    if description:
        st.caption(description)


def info_card(title: str, content: str, icon: str = "ℹ️"):
    """渲染信息卡片
    
    Args:
        title: 卡片标题
        content: 卡片内容
        icon: 图标
    """
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #e0f2fe 0%, #f0f9ff 100%);
        padding: 1rem 1.25rem;
        border-radius: 0.5rem;
        border-left: 4px solid #0284c7;
        margin: 0.5rem 0;
    ">
        <div style="font-weight: 600; color: #0369a1; margin-bottom: 0.25rem;">
            {icon} {title}
        </div>
        <div style="color: #475569; font-size: 0.9rem;">
            {content}
        </div>
    </div>
    """, unsafe_allow_html=True)


def metric_card(label: str, value, delta=None, help_text: str = None):
    """渲染指标卡片
    
    Args:
        label: 指标标签
        value: 指标值
        delta: 变化值
        help_text: 帮助文本
    """
    st.metric(label=label, value=value, delta=delta, help=help_text)


def metrics_row(*metrics):
    """渲染一行指标
    
    Args:
        metrics: [(label, value, delta?, help?), ...] 列表
    """
    cols = st.columns(len(metrics))
    for col, metric in zip(cols, metrics):
        with col:
            if len(metric) == 2:
                metric_card(metric[0], metric[1])
            elif len(metric) == 3:
                metric_card(metric[0], metric[1], metric[2])
            else:
                metric_card(metric[0], metric[1], metric[2], metric[3])


def two_column_layout(left_ratio: float = 0.5):
    """创建两列布局
    
    Args:
        left_ratio: 左列占比
        
    Returns:
        (left_col, right_col) 元组
    """
    return st.columns([left_ratio, 1 - left_ratio])

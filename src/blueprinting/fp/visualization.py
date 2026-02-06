"""浮点数精度可视化工具函数"""

import matplotlib
import numpy as np
from matplotlib import pyplot as plt


def plot_float_formats(formats: dict) -> plt.Figure:
    """绘制浮点数格式的位图布局

    Args:
        formats: 格式字典 {name: (sign_bits, exponent_bits, mantissa_bits)}

    Returns:
        matplotlib Figure 对象
    """
    labels = []
    data = []
    for k, v in formats.items():
        labels.append(k)
        # 0.0 = 符号位(红色), 0.25 = 指数位(绿色), 0.5 = 尾数位(蓝色), 0.75 = 未使用(白色)
        v_data = [0.0] * v[0] + [0.25] * v[1] + [0.5] * v[2]
        v_data += [0.75] * (32 - len(v_data))
        data.append(v_data)

    data = np.array(data)
    fig = plt.figure(figsize=(10, 5))
    ax = plt.gca()

    ax.set_xticks(list(range(32)))
    ax.set_yticks(list(range(len(labels))), labels=labels)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    im = ax.imshow(
        data,
        cmap=matplotlib.colors.ListedColormap(
            ["#ff000050", "#00ff0050", "#0000ff50", "#ffffff80"]
        ),
    )

    # 添加标签文字
    data = im.get_array()
    kw = {"horizontalalignment": "center", "verticalalignment": "center"}
    label_map = {0.0: "S", 0.25: "E", 0.5: "M", 0.75: ""}
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            im.axes.text(j, i, label_map[data[i, j]], **kw)

    plt.tight_layout()
    plt.title("bitmap layouts of float pointing numbers")
    return fig


def plot_subnormal_distribution(
    fp_values: list,
    subnormal: float,
    rng: float = 0.1,
) -> plt.Figure:
    """绘制浮点数分布图，标识 subnormal 区域

    Args:
        fp_values: 浮点数值列表
        subnormal: subnormal 阈值
        rng: 可视化范围

    Returns:
        matplotlib Figure 对象
    """
    fig = plt.figure(figsize=(20, 2))

    # 上方：完整分布
    plt.subplot(2, 1, 1)
    plt.scatter(
        fp_values,
        [0 for _ in fp_values],
        c=["r" if abs(v) < subnormal else "b" for v in fp_values],
        linewidths=1,
        marker="|",
    )
    plt.yticks([])
    plt.tight_layout()

    # 下方：指定范围内的分布
    plt.subplot(2, 1, 2)
    plt.scatter(
        fp_values,
        [0 for _ in fp_values],
        c=["r" if abs(v) < subnormal else "b" for v in fp_values],
        linewidths=1,
        marker="|",
    )
    plt.xlim(-rng, rng)
    plt.yticks([])
    plt.tight_layout()

    return fig


def plot_quantization_error(
    fp_values: list,
    subnormal: float,
    rng: float = 0.1,
) -> plt.Figure:
    """绘制量化误差图

    Args:
        fp_values: 浮点数值列表
        subnormal: subnormal 阈值
        rng: 可视化范围

    Returns:
        matplotlib Figure 对象
    """
    x = np.arange(-rng, rng, 2 * rng / 1e4, dtype=np.float64)
    y = [fp_values[i] for i in np.digitize(x, fp_values)]
    err = np.abs(x - y)

    fig = plt.figure(figsize=(20, 3))
    plt.plot(
        x,
        err,
        "b",
        [-subnormal, -subnormal, subnormal, subnormal],
        [0.0, 0.0, 0.0, 0.0],
        "r",
    )
    plt.legend(labels=["rtol", "subnormal area", "zeroed area"])

    return fig


def plot_fp16_precision_error(
    rng: float = 0.1,
    spl: int = 10,
) -> plt.Figure:
    """绘制 FP16 精度误差图

    Args:
        rng: 可视化范围
        spl: 采样间隔

    Returns:
        matplotlib Figure 对象
    """
    import torch

    x = np.arange(-rng, rng, 2 * rng / 1e4, dtype=np.float64)
    y = torch.tensor(x).to(torch.float16).to(torch.float64).numpy()

    fp16_atol = np.abs(x - y)
    fp16_rtol = np.abs(x - y) / np.abs(x)
    fp16_rtol[np.argmax(fp16_rtol)] = 0.0

    fig = plt.figure(figsize=(20, 5))

    plt.subplot(2, 1, 1)
    plt.plot(x[0::spl], fp16_atol[0::spl], "b")
    plt.title("atol")

    plt.subplot(2, 1, 2)
    plt.plot(x[0::spl], fp16_rtol[0::spl], "b")
    plt.title("rtol")

    plt.tight_layout()
    return fig


# 默认颜色映射
DEFAULT_COLORMAP = matplotlib.colors.ListedColormap(
    [
        "#00ff0050",  # 0 => normal
        "#ff000080",  # 0.25 => maxval
        "#0000ff80",  # 0.5 => subnormal
        "#000000FF",  # 0.75 => zeroed
        "#f000f080",  # 1.0 => traced
    ]
)


def show_map(
    x: np.ndarray,
    name: str,
    maxval: float = 0,
    subnormal: float = 0,
    zeroed: float = 0,
    traced_values: list = None,
    colormap=None,
    limit: float = None,
) -> None:
    """绘制操作结果的热图

    Args:
        x: 操作结果矩阵
        name: 图表标题
        maxval: 最大值阈值
        subnormal: subnormal 阈值
        zeroed: 零值阈值
        traced_values: 要追踪的特殊值
        colormap: 颜色映射
        limit: 坐标轴范围限制
    """
    if colormap is None:
        colormap = DEFAULT_COLORMAP
    if traced_values is None:
        traced_values = []

    cm = (
        ((x > maxval) | (x < -maxval)) * 0.25
        + (np.abs(x) <= subnormal) * 0.50
        + (np.abs(x) < zeroed) * 0.25
    )
    for traced in traced_values:
        cm += (x == traced) * 1.0 + (x == -traced) * 1.0

    plt.imshow(cm, cmap=colormap)
    plt.title(name)


def show_scatter(
    x: np.ndarray,
    name: str,
    fp_values: list = None,
    maxval: float = 0,
    subnormal: float = 0,
    zeroed: float = 0,
    traced_values: list = None,
    colormap=None,
    limit: float = None,
) -> None:
    """绘制操作结果的散点图

    Args:
        x: 操作结果矩阵
        name: 图表标题
        fp_values: 浮点数值列表
        maxval: 最大值阈值
        subnormal: subnormal 阈值
        zeroed: 零值阈值
        traced_values: 要追踪的特殊值
        colormap: 颜色映射
        limit: 坐标轴范围限制
    """
    if colormap is None:
        colormap = DEFAULT_COLORMAP
    if traced_values is None:
        traced_values = []
    if fp_values is None:
        fp_values = []

    fp = np.array([fp_values])
    xl = fp.repeat(len(fp_values), axis=0)
    yl = xl.T

    cm = (
        ((x > maxval) | (x < -maxval)) * 0.25
        + (np.abs(x) <= subnormal) * 0.50
        + (np.abs(x) < zeroed) * 0.25
    )
    for traced in traced_values:
        cm += (x == traced) * 1.0 + (x == -traced) * 1.0

    plt.scatter(xl, yl, c=cm, linewidths=0.5, marker=".", cmap=colormap)
    if limit is not None:
        plt.xlim([-limit, +limit])
        plt.ylim([-limit, +limit])
    plt.title(name)


def plot_basic_op_map(
    fp_values: list,
    show_fn,
    **kwargs,
) -> plt.Figure:
    """绘制四则运算对精度的影响

    Args:
        fp_values: 浮点数值列表
        show_fn: 显示函数 (show_map 或 show_scatter)
        **kwargs: 传递给 show_fn 的参数
            - need_fp_values: 如果为 True，则将处理后的 fp_values 传递给 show_fn

    Returns:
        matplotlib Figure 对象
    """
    fig = plt.figure(figsize=(15, 15))

    fv = fp_values[:]
    limit = kwargs.get("limit")
    if limit is not None:
        fv = [v for v in fv if abs(v) < limit]

    # 限制数据点数量
    while len(fv) > 512:
        fv = fv[::2]

    # 如果 show_fn 需要 fp_values 参数（如 show_scatter），则传递处理后的值
    if kwargs.pop("need_fp_values", False):
        kwargs["fp_values"] = fv

    fv_arr = np.array([fv])
    xv = fv_arr.repeat(fv_arr.shape[1], axis=0)
    yv = xv.T

    # 加法
    plt.subplot(2, 2, 1)
    show_fn(xv + yv, "A+B", **kwargs)

    # 减法
    plt.subplot(2, 2, 2)
    show_fn(xv - yv, "A-B", **kwargs)

    # 乘法
    plt.subplot(2, 2, 3)
    show_fn(xv * yv, "AxB", **kwargs)

    # 除法
    plt.subplot(2, 2, 4)
    show_fn(xv / yv, "A / B", **kwargs)

    plt.tight_layout()
    return fig

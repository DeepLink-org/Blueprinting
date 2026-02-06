"""Performance Database - 基于实测数据的算子性能查表 + roofline fallback.

移植自 aiconfigurator 的 PerfDatabase，提供:
1. 加载硬件规格 (system spec YAML) 和性能数据 (CSV)
2. 给定算子类型和规模参数，自动查表获取延迟 (ms)
3. 使用插值 (1D/2D/3D) 处理表中不存在的维度
4. 当查表失败时，fallback 到 roofline / 经验模型估算

数据目录结构 (来自 aiconfigurator):
    systems/
      {system}.yaml                      # 硬件规格 (GPU, 网络, 内存)
      data/{system}/{backend}/{version}/  # 性能数据 CSV
        gemm_perf.txt
        context_attention_perf.txt
        generation_attention_perf.txt
        custom_allreduce_perf.txt
        nccl/{nccl_version}/nccl_perf.txt

支持的查询:
    - query_gemm(m, n, k, quant_mode) → latency (ms)
    - query_context_attention(b, s, prefix, n, n_kv, ...) → latency (ms)
    - query_generation_attention(b, s, n, n_kv, ...) → latency (ms)
    - query_custom_allreduce(quant_mode, tp_size, size) → latency (ms)
    - query_nccl(dtype, num_gpus, operation, message_size) → latency (ms)
    - query_mem_op(mem_bytes) → latency (ms)
    - query_p2p(message_bytes) → latency (ms)
"""

from __future__ import annotations

import csv
import functools
import importlib.resources as pkg_resources
import logging
import math
import os
from collections import defaultdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# 枚举类型 (移植自 aiconfigurator/sdk/common.py)
# ============================================================================


class DatabaseMode(Enum):
    """数据库查询模式."""

    SILICON = 0  # 使用实测数据 (默认)
    HYBRID = 1  # 实测数据优先，失败时用经验模型
    EMPIRICAL = 2  # SOL + 经验修正因子
    SOL = 3  # 纯理论峰值 (Speed Of Light)


class _QuantMapping:
    """量化模式的属性映射."""

    __slots__ = ("memory", "compute", "name")

    def __init__(self, memory: float, compute: float, name: str):
        self.memory = memory  # bytes per element
        self.compute = compute  # compute multiplier (相对于 fp16)
        self.name = name

    def __repr__(self) -> str:
        return f"_QuantMapping(memory={self.memory}, compute={self.compute}, name='{self.name}')"


class GEMMQuantMode(Enum):
    """GEMM 量化模式.

    每种模式定义:
    - memory: 每元素字节数 (影响权重/激活内存带宽)
    - compute: 计算吞吐倍率 (相对于 fp16 tensor core)
    """

    float16 = _QuantMapping(2, 1, "float16")  # w16a16
    int8_wo = _QuantMapping(1, 1, "int8_wo")  # w8a16
    int4_wo = _QuantMapping(0.5, 1, "int4_wo")  # w4a16
    fp8 = _QuantMapping(1, 2, "fp8")  # w8fp8
    fp8_static = _QuantMapping(1, 2, "fp8_static")  # fp8 静态量化
    sq = _QuantMapping(1, 2, "sq")  # w8int8 (SmoothQuant)
    fp8_block = _QuantMapping(1, 2, "fp8_block")  # fp8 block-wise
    nvfp4 = _QuantMapping(0.5, 4, "nvfp4")  # nvfp4 (Blackwell)


class CommQuantMode(Enum):
    """通信量化模式."""

    half = _QuantMapping(2, 0, "half")
    int8 = _QuantMapping(1, 0, "int8")
    fp8 = _QuantMapping(1, 0, "fp8")


class KVCacheQuantMode(Enum):
    """KV-Cache 量化模式."""

    float16 = _QuantMapping(2, 0, "float16")
    int8 = _QuantMapping(1, 0, "int8")
    fp8 = _QuantMapping(1, 0, "fp8")


class FMHAQuantMode(Enum):
    """FMHA (Flash Multi-Head Attention) 量化模式."""

    float16 = _QuantMapping(2, 1, "float16")
    fp8 = _QuantMapping(1, 2, "fp8")


# ============================================================================
# 数据加载函数 (移植自 aiconfigurator)
# ============================================================================


def _load_csv_data(filepath: str) -> Optional[List[Dict[str, str]]]:
    """加载 CSV 文件为字典列表."""
    if not os.path.exists(filepath):
        logger.warning(f"数据文件未找到: {filepath}")
        return None
    with open(filepath, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def load_gemm_data(gemm_file: str) -> Optional[Dict]:
    """加载 GEMM 性能数据.

    数据结构: {GEMMQuantMode: {m: {n: {k: latency}}}}
    """
    rows = _load_csv_data(gemm_file)
    if rows is None:
        return None

    gemm_data: Dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict())))

    for row in rows:
        quant_str = row["gemm_dtype"]
        # 跳过不支持的量化模式
        if quant_str in ("awq", "gptq"):
            continue
        try:
            quant_mode = GEMMQuantMode[quant_str]
        except KeyError:
            logger.debug(f"跳过未知 GEMM 量化模式: {quant_str}")
            continue

        m = int(row["m"])
        n = int(row["n"])
        k = int(row["k"])
        latency = float(row["latency"])

        try:
            gemm_data[quant_mode][m][n][k]  # 检查冲突
            logger.debug(f"GEMM 数据冲突: {quant_mode} {m} {n} {k}")
        except KeyError:
            gemm_data[quant_mode][m][n][k] = latency

    return gemm_data


def load_context_attention_data(attention_file: str) -> Optional[Dict]:
    """加载 context attention 性能数据.

    数据结构: {fmha_quant: {kv_cache_quant: {n_kv: {head_size: {window: {n_heads: {s: {batch: latency}}}}}}}}
    """
    rows = _load_csv_data(attention_file)
    if rows is None:
        return None

    data: Dict = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(
                        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict()))
                    )
                )
            )
        )
    )

    for row in rows:
        try:
            attn_dtype = row.get("attn_dtype", "float16")
            kv_dtype = row.get("kv_cache_dtype", "float16")
            fmha_quant = FMHAQuantMode[attn_dtype]
            kv_quant = KVCacheQuantMode[kv_dtype]
        except KeyError:
            continue

        batch = int(row["batch_size"])
        s = int(row["isl"])
        n_heads = int(row["num_heads"])
        n_kv = int(row["num_key_value_heads"])
        head_size = int(row.get("head_dim", 128))
        window_size = 0  # 默认无 sliding window
        latency = float(row["latency"])

        try:
            data[fmha_quant][kv_quant][n_kv][head_size][window_size][n_heads][s][batch]
            logger.debug(f"Context attention 数据冲突")
        except KeyError:
            data[fmha_quant][kv_quant][n_kv][head_size][window_size][n_heads][s][batch] = latency

    return data


def load_generation_attention_data(attention_file: str) -> Optional[Dict]:
    """加载 generation attention 性能数据.

    数据结构: {fmha_quant: {kv_cache_quant: {n_kv: {head_size: {window: {n_heads: {batch: {step: latency}}}}}}}}
    """
    rows = _load_csv_data(attention_file)
    if rows is None:
        return None

    data: Dict = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(
                        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict()))
                    )
                )
            )
        )
    )

    for row in rows:
        try:
            attn_dtype = row.get("attn_dtype", "float16")
            kv_dtype = row.get("kv_cache_dtype", "float16")
            fmha_quant = FMHAQuantMode[attn_dtype]
            kv_quant = KVCacheQuantMode[kv_dtype]
        except KeyError:
            continue

        batch = int(row["batch_size"])
        step = int(row.get("step", 0))
        n_heads = int(row["num_heads"])
        n_kv = int(row["num_key_value_heads"])
        head_size = int(row.get("head_dim", 128))
        window_size = 0
        latency = float(row["latency"])

        try:
            data[fmha_quant][kv_quant][n_kv][head_size][window_size][n_heads][batch][step]
        except KeyError:
            data[fmha_quant][kv_quant][n_kv][head_size][window_size][n_heads][batch][step] = latency

    return data


def load_custom_allreduce_data(allreduce_file: str) -> Optional[Dict]:
    """加载 custom allreduce 性能数据.

    数据结构: {CommQuantMode: {tp_size: {strategy: {message_size: latency}}}}
    """
    rows = _load_csv_data(allreduce_file)
    if rows is None:
        return None

    data: Dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict())))

    for row in rows:
        dtype = CommQuantMode.half  # 当前都是 half
        tp_size = int(row["num_gpus"])
        message_size = int(row["message_size"])
        latency = float(row["latency"])
        strategy = "AUTO"

        try:
            data[dtype][tp_size][strategy][message_size]
            logger.debug(f"AllReduce 数据冲突: {dtype} {tp_size} {strategy} {message_size}")
        except KeyError:
            data[dtype][tp_size][strategy][message_size] = latency

    return data


def load_nccl_data(nccl_file: str) -> Optional[Dict]:
    """加载 NCCL 性能数据.

    数据结构: {CommQuantMode: {operation: {num_gpus: {message_size: latency}}}}
    """
    rows = _load_csv_data(nccl_file)
    if rows is None:
        return None

    data: Dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict())))

    for row in rows:
        try:
            dtype = CommQuantMode[row["nccl_dtype"]]
        except KeyError:
            logger.debug(f"跳过未知 NCCL dtype: {row['nccl_dtype']}")
            continue

        num_gpus = int(row["num_gpus"])
        message_size = int(row["message_size"])
        op_name = row["op_name"]
        latency = float(row["latency"])

        try:
            data[dtype][op_name][num_gpus][message_size]
        except KeyError:
            data[dtype][op_name][num_gpus][message_size] = latency

    return data


# ============================================================================
# 辅助函数: 获取默认数据目录
# ============================================================================


def get_systems_dir() -> str:
    """获取 blueprinting 包内的 systems 数据目录路径."""
    return str(pkg_resources.files("blueprinting") / "systems")


def get_supported_databases(
    systems_dir: Optional[str] = None,
) -> Dict[str, Dict[str, List[str]]]:
    """获取所有支持的系统/后端/版本组合.

    Returns:
        {system: {backend: [versions]}}
    """
    import yaml

    systems_dir = systems_dir or get_systems_dir()
    supported: Dict = defaultdict(lambda: defaultdict(list))

    if not os.path.isdir(systems_dir):
        logger.warning(f"Systems 目录不存在: {systems_dir}")
        return supported

    for fname in os.listdir(systems_dir):
        if not fname.endswith(".yaml"):
            continue
        system = fname.rsplit(".", 1)[0]
        yaml_path = os.path.join(systems_dir, fname)
        try:
            with open(yaml_path) as f:
                spec = yaml.safe_load(f)
            data_dir = os.path.join(systems_dir, spec.get("data_dir", ""))
            if not os.path.isdir(data_dir):
                continue
            for backend in ("trtllm", "vllm", "sglang"):
                backend_path = os.path.join(data_dir, backend)
                if not os.path.isdir(backend_path):
                    continue
                versions = sorted(
                    v
                    for v in os.listdir(backend_path)
                    if not v.startswith(".") and os.path.isdir(os.path.join(backend_path, v))
                )
                if versions:
                    supported[system][backend] = versions
        except Exception as e:
            logger.warning(f"无法处理 {fname}: {e}")

    return supported


# ============================================================================
# PerfDatabase 核心类
# ============================================================================


class PerfDatabase:
    """性能数据库: 基于实测数据的算子延迟查表.

    使用方式:
        db = PerfDatabase("h100_sxm", "trtllm", "1.0.0rc3")
        # 查询 GEMM
        lat = db.query_gemm(4096, 4096, 4096, GEMMQuantMode.fp8)
        # 查询 NCCL allreduce
        lat = db.query_nccl(CommQuantMode.half, 8, "all_reduce", 16384)
        # 查询内存操作 (纯模型)
        lat = db.query_mem_op(1024 * 1024)

    参数:
        system: 系统名称 (e.g., "h100_sxm", "b200_sxm")
        backend: 后端名称 (e.g., "trtllm", "vllm", "sglang")
        version: 后端版本 (e.g., "1.0.0rc3")
        systems_dir: systems 数据根目录路径
    """

    def __init__(
        self,
        system: str,
        backend: str,
        version: str,
        systems_dir: Optional[str] = None,
    ) -> None:
        import yaml

        self.system = system
        self.backend = backend
        self.version = version

        systems_dir = systems_dir or get_systems_dir()
        yaml_path = os.path.join(systems_dir, f"{system}.yaml")
        with open(yaml_path) as f:
            self.system_spec: Dict[str, Any] = yaml.safe_load(f)

        self._default_database_mode = DatabaseMode.SILICON

        # 构建数据目录路径
        data_dir = os.path.join(systems_dir, self.system_spec["data_dir"], backend, version)
        nccl_version = self.system_spec["misc"]["nccl_version"]
        nccl_file = os.path.join(
            systems_dir, self.system_spec["data_dir"], "nccl", nccl_version, "nccl_perf.txt"
        )

        # 加载性能数据
        self._gemm_data = load_gemm_data(os.path.join(data_dir, "gemm_perf.txt"))
        self._context_attention_data = load_context_attention_data(
            os.path.join(data_dir, "context_attention_perf.txt")
        )
        self._generation_attention_data = load_generation_attention_data(
            os.path.join(data_dir, "generation_attention_perf.txt")
        )
        self._custom_allreduce_data = load_custom_allreduce_data(
            os.path.join(data_dir, "custom_allreduce_perf.txt")
        )
        self._nccl_data = load_nccl_data(nccl_file)

        logger.info(
            f"PerfDatabase 已加载: system={system}, backend={backend}, version={version}, "
            f"gemm={'OK' if self._gemm_data else 'MISSING'}, "
            f"ctx_attn={'OK' if self._context_attention_data else 'MISSING'}, "
            f"gen_attn={'OK' if self._generation_attention_data else 'MISSING'}, "
            f"allreduce={'OK' if self._custom_allreduce_data else 'MISSING'}, "
            f"nccl={'OK' if self._nccl_data else 'MISSING'}"
        )

    # ---- 属性 ----

    @property
    def default_database_mode(self) -> DatabaseMode:
        return self._default_database_mode

    @default_database_mode.setter
    def default_database_mode(self, mode: DatabaseMode) -> None:
        if mode != self._default_database_mode:
            # 清除缓存
            for attr_name in dir(self):
                attr = getattr(self, attr_name)
                if hasattr(attr, "cache_clear") and callable(attr):
                    attr.cache_clear()
            self._default_database_mode = mode

    # ---- 插值方法 (移植自 aiconfigurator) ----

    @staticmethod
    def _nearest_1d_point(x: int, values: List[int], inner_only: bool = True) -> Tuple[int, int]:
        """找到最近的两个点用于 1D 插值.

        Args:
            x: 目标值
            values: 已知点列表
            inner_only: 是否仅允许内部插值 (否则外推到最近两点)

        Returns:
            (left, right) 两个最近的点
        """
        assert values and len(values) >= 2, f"需要至少 2 个数据点, 实际: {values}"
        sorted_vals = sorted(values)

        if x < sorted_vals[0]:
            if inner_only:
                raise ValueError(f"x={x} 低于数据范围 {sorted_vals}")
            return sorted_vals[0], sorted_vals[1]
        elif x > sorted_vals[-1]:
            if inner_only:
                raise ValueError(f"x={x} 高于数据范围 {sorted_vals}")
            return sorted_vals[-2], sorted_vals[-1]

        for i, val in enumerate(sorted_vals):
            if x >= val and i != len(sorted_vals) - 1:
                continue
            return sorted_vals[i - 1], val

        return sorted_vals[-2], sorted_vals[-1]

    @staticmethod
    def _interp_1d(x: List[int], y: List[float], value: int) -> float:
        """1D 线性插值.

        包含外推保护: 如果趋势与坐标方向相反且在范围外，
        则使用最近点的值 (防止负值外推)。

        Args:
            x: [x0, x1] 两个已知坐标
            y: [y0, y1] 两个已知值
            value: 目标坐标

        Returns:
            插值/外推后的值
        """
        x0, x1 = x
        y0, y1 = y

        # 外推保护
        if (x0 - x1) * (y0 - y1) < 0 and (value - x0) * (value - x1) > 0:
            y1 = y0
        if y0 == y1:
            return y0
        return y0 + (y1 - y0) / (x1 - x0) * (value - x0)

    def _interp_3d_scipy(self, x: int, y: int, z: int, data: Dict) -> float:
        """3D 线性插值 (使用 scipy griddata)."""
        from scipy import interpolate as sp_interp

        points_list = []
        values_list = []
        x_left, x_right = self._nearest_1d_point(x, list(data.keys()))
        for i in [x_left, x_right]:
            y_left, y_right = self._nearest_1d_point(y, list(data[i].keys()))
            for j in [y_left, y_right]:
                z_left, z_right = self._nearest_1d_point(z, list(data[i][j].keys()))
                points_list.append([i, j, z_left])
                points_list.append([i, j, z_right])
                values_list.append(data[i][j][z_left])
                values_list.append(data[i][j][z_right])

        result = sp_interp.griddata(
            np.array(points_list), np.array(values_list), (x, y, z), method="linear"
        )
        return max(0.0, float(result))

    def _bilinear_interpolation(
        self, x_list: List[int], y_list: List[int], x: int, y: int, data: Dict
    ) -> float:
        """2D 双线性插值."""
        x1, x2 = x_list
        y1, y2 = y_list
        Q11 = data[x1][y1]
        Q12 = data[x1][y2]
        Q21 = data[x2][y1]
        Q22 = data[x2][y2]

        total_weight = (x2 - x1) * (y2 - y1)
        if total_weight == 0:
            return Q11

        value = (
            Q11 * (x2 - x) * (y2 - y)
            + Q12 * (x2 - x) * (y - y1)
            + Q21 * (x - x1) * (y2 - y)
            + Q22 * (x - x1) * (y - y1)
        ) / total_weight
        return value

    def _interp_2d_1d(
        self, x: int, y: int, z: int, data: Dict, method: str = "bilinear"
    ) -> float:
        """3D 插值: 先在 (y, z) 平面 2D 插值，再沿 x 方向 1D 插值.

        支持 "cubic" (使用 scipy) 和 "bilinear" (自定义)。
        """
        from scipy import interpolate as sp_interp

        x_values = []
        x_left, x_right = self._nearest_1d_point(x, list(data.keys()))

        for i in [x_left, x_right]:
            y_left, y_right = self._nearest_1d_point(y, list(data[i].keys()))
            points = []
            vals = []
            for j in [y_left, y_right]:
                z_left, z_right = self._nearest_1d_point(z, list(data[i][j].keys()))
                points.append([j, z_left])
                points.append([j, z_right])
                vals.append(data[i][j][z_left])
                vals.append(data[i][j][z_right])

            if method == "cubic":
                result = sp_interp.griddata(
                    np.array(points), np.array(vals), (y, z), method="cubic"
                )
                x_values.append(max(0.0, float(result)))
            else:  # bilinear
                result = self._bilinear_interpolation(
                    [y_left, y_right], [z_left, z_right], y, z, data[i]
                )
                x_values.append(max(0.0, float(result)))

        return max(0.0, self._interp_1d([x_left, x_right], x_values, x))

    def _interp_3d(self, x: int, y: int, z: int, data: Dict, method: str = "cubic") -> float:
        """3D 插值主入口.

        Args:
            x, y, z: 目标坐标
            data: 3 层嵌套字典 {x: {y: {z: latency}}}
            method: "linear" (scipy griddata), "cubic" (2D scipy + 1D), "bilinear" (2D 手动 + 1D)

        Returns:
            插值后的延迟值 (ms)
        """
        if method == "linear":
            return self._interp_3d_scipy(x, y, z, data)
        else:
            return self._interp_2d_1d(x, y, z, data, method)

    # ---- 带宽辅助 ----

    def _get_p2p_bandwidth(self, num_gpus: int) -> float:
        """获取适当的点对点带宽 (bytes/s).

        - num_gpus <= num_gpus_per_node: intra_node_bw (NVLink)
        - num_gpus > num_gpus_per_node: inter_node_bw (IB)
        """
        node_spec = self.system_spec["node"]
        if num_gpus <= node_spec["num_gpus_per_node"]:
            return node_spec["intra_node_bw"]
        else:
            return node_spec["inter_node_bw"]

    # ---- 查询方法 ----

    @functools.lru_cache(maxsize=32768)
    def query_gemm(
        self,
        m: int,
        n: int,
        k: int,
        quant_mode: GEMMQuantMode = GEMMQuantMode.float16,
        database_mode: Optional[DatabaseMode] = None,
    ) -> float:
        """查询 GEMM 延迟 (ms).

        Args:
            m: 输出矩阵行数 (batch 维度)
            n: 输出矩阵列数 (hidden 维度)
            k: 内部维度
            quant_mode: 量化模式
            database_mode: 查询模式 (默认使用数据库模式)

        Returns:
            latency (ms)

        Example:
            >>> db.query_gemm(4096, 4096, 4096, GEMMQuantMode.fp8)
            1.234  # ms
        """

        def get_sol() -> float:
            sol_math = (
                2 * m * n * k
                / (self.system_spec["gpu"]["float16_tc_flops"] * quant_mode.value.compute)
                * 1000
            )
            sol_mem = (
                quant_mode.value.memory
                * (m * n + m * k + n * k)
                / self.system_spec["gpu"]["mem_bw"]
                * 1000
            )
            return max(sol_math, sol_mem)

        def get_empirical() -> float:
            return get_sol() / 0.8

        mode = database_mode or self._default_database_mode

        # 规范化 fp8_static -> fp8
        table_quant = quant_mode
        if quant_mode == GEMMQuantMode.fp8_static:
            table_quant = GEMMQuantMode.fp8

        if mode == DatabaseMode.SOL:
            return get_sol()
        elif mode == DatabaseMode.EMPIRICAL:
            return get_empirical()
        else:
            # SILICON or HYBRID: 尝试查表
            try:
                if self._gemm_data is None:
                    raise ValueError("GEMM 数据未加载")
                if table_quant not in self._gemm_data:
                    supported = sorted(k.name for k in self._gemm_data)
                    raise ValueError(
                        f"GEMM 不支持量化模式 {quant_mode.name}, 可用: {supported}"
                    )
                return self._interp_3d(m, n, k, self._gemm_data[table_quant], "cubic")
            except Exception as e:
                if mode == DatabaseMode.HYBRID:
                    logger.debug(f"GEMM 查表失败 ({e}), fallback 到经验模型")
                    return get_empirical()
                raise

    @functools.lru_cache(maxsize=32768)
    def query_context_attention(
        self,
        b: int,
        s: int,
        prefix: int = 0,
        n: int = 32,
        n_kv: int = 8,
        kvcache_quant_mode: KVCacheQuantMode = KVCacheQuantMode.float16,
        fmha_quant_mode: FMHAQuantMode = FMHAQuantMode.float16,
        database_mode: Optional[DatabaseMode] = None,
        window_size: int = 0,
        head_size: int = 128,
    ) -> float:
        """查询 context (prefill) attention 延迟 (ms).

        Args:
            b: batch size
            s: 输入序列长度 (当前 step 的 token 数)
            prefix: 前缀长度 (prefix caching)
            n: attention head 数
            n_kv: KV head 数
            kvcache_quant_mode: KV-cache 量化
            fmha_quant_mode: FMHA 量化
            database_mode: 查询模式
            window_size: sliding window 大小 (0 = 全注意力)
            head_size: head 维度

        Returns:
            latency (ms)
        """

        def get_sol() -> float:
            """Context attention SOL: O(b * n * s^2 * head_size)"""
            full_s = s + prefix
            flops = 2 * b * n * full_s * full_s * head_size
            mem_bytes = (
                2 * b * n_kv * full_s * head_size * kvcache_quant_mode.value.memory  # KV read
                + 2 * b * n * full_s * head_size * fmha_quant_mode.value.memory  # Q+O
            )
            sol_math = flops / self.system_spec["gpu"]["float16_tc_flops"] * 1000
            sol_mem = mem_bytes / self.system_spec["gpu"]["mem_bw"] * 1000
            result = max(sol_math, sol_mem)
            # prefix correction
            if prefix > 0:
                result *= (full_s * full_s - prefix * prefix) / (full_s * full_s)
            return result

        def get_empirical() -> float:
            return get_sol() / 0.8

        mode = database_mode or self._default_database_mode

        if mode == DatabaseMode.SOL:
            return get_sol()
        elif mode == DatabaseMode.EMPIRICAL:
            return get_empirical()
        else:
            try:
                if self._context_attention_data is None:
                    raise ValueError("Context attention 数据未加载")
                attn_data = self._context_attention_data
                if fmha_quant_mode not in attn_data:
                    raise ValueError(f"不支持 FMHA 量化 {fmha_quant_mode}")
                sub = attn_data[fmha_quant_mode][kvcache_quant_mode][n_kv][head_size][window_size]
                full_s = s + prefix
                result = self._interp_3d(n, full_s, b, sub, "cubic")
                if prefix > 0:
                    result *= (full_s * full_s - prefix * prefix) / (full_s * full_s)
                return result
            except Exception as e:
                if mode == DatabaseMode.HYBRID:
                    logger.debug(f"Context attention 查表失败 ({e}), fallback")
                    return get_empirical()
                raise

    @functools.lru_cache(maxsize=32768)
    def query_generation_attention(
        self,
        b: int,
        s: int,
        n: int = 32,
        n_kv: int = 8,
        kvcache_quant_mode: KVCacheQuantMode = KVCacheQuantMode.float16,
        database_mode: Optional[DatabaseMode] = None,
        window_size: int = 0,
        head_size: int = 128,
    ) -> float:
        """查询 generation (decode) attention 延迟 (ms).

        Args:
            b: batch size
            s: KV-cache 长度 (已生成的 token 数)
            n: attention head 数
            n_kv: KV head 数
            kvcache_quant_mode: KV-cache 量化
            database_mode: 查询模式
            window_size: sliding window 大小
            head_size: head 维度

        Returns:
            latency (ms)
        """

        def get_sol() -> float:
            """Generation attention: memory-bandwidth bound, 读 KV-cache"""
            mem_bytes = 2 * b * n_kv * s * head_size * kvcache_quant_mode.value.memory
            return mem_bytes / self.system_spec["gpu"]["mem_bw"] * 1000

        def get_empirical() -> float:
            return get_sol() / 0.8

        mode = database_mode or self._default_database_mode

        if mode == DatabaseMode.SOL:
            return get_sol()
        elif mode == DatabaseMode.EMPIRICAL:
            return get_empirical()
        else:
            try:
                if self._generation_attention_data is None:
                    raise ValueError("Generation attention 数据未加载")
                fmha_quant = FMHAQuantMode.float16  # generation 通常用 float16
                attn_data = self._generation_attention_data
                if fmha_quant not in attn_data:
                    raise ValueError(f"不支持 FMHA 量化 {fmha_quant}")
                sub = attn_data[fmha_quant][kvcache_quant_mode][n_kv][head_size][window_size]
                return self._interp_3d(n, b, s, sub, "bilinear")
            except Exception as e:
                if mode == DatabaseMode.HYBRID:
                    logger.debug(f"Generation attention 查表失败 ({e}), fallback")
                    return get_empirical()
                raise

    @functools.lru_cache(maxsize=32768)
    def query_custom_allreduce(
        self,
        quant_mode: CommQuantMode = CommQuantMode.half,
        tp_size: int = 8,
        size: int = 0,
        database_mode: Optional[DatabaseMode] = None,
    ) -> float:
        """查询 custom AllReduce 延迟 (ms).

        Args:
            quant_mode: 通信量化模式
            tp_size: TP 并行度
            size: 元素数 (不是字节数)
            database_mode: 查询模式

        Returns:
            latency (ms)
        """

        def get_sol() -> float:
            if tp_size == 1:
                return 0.0
            p2p_bw = self._get_p2p_bandwidth(tp_size)
            return 2 * size * 2 / tp_size * (tp_size - 1) / p2p_bw * 1000

        def get_empirical() -> float:
            return get_sol() / 0.8

        mode = database_mode or self._default_database_mode

        if mode == DatabaseMode.SOL:
            return get_sol()
        elif mode == DatabaseMode.EMPIRICAL:
            return get_empirical()
        else:
            try:
                if tp_size == 1:
                    return 0.0
                if self._custom_allreduce_data is None:
                    raise ValueError("Custom allreduce 数据未加载")

                comm_dict = self._custom_allreduce_data[quant_mode][min(tp_size, 8)]["AUTO"]
                size_left, size_right = self._nearest_1d_point(
                    size, list(comm_dict.keys()), inner_only=False
                )
                lat = self._interp_1d(
                    [size_left, size_right],
                    [comm_dict[size_left], comm_dict[size_right]],
                    size,
                )

                # 跨节点修正
                if tp_size > self.system_spec["node"]["num_gpus_per_node"]:
                    base_bw = self._get_p2p_bandwidth(
                        self.system_spec["node"]["num_gpus_per_node"]
                    )
                    target_bw = self._get_p2p_bandwidth(tp_size)
                    scale = (
                        (tp_size - 1)
                        / tp_size
                        * self.system_spec["node"]["num_gpus_per_node"]
                        / (self.system_spec["node"]["num_gpus_per_node"] - 1)
                        * base_bw
                        / target_bw
                    )
                    lat *= scale

                return lat
            except Exception as e:
                if mode == DatabaseMode.HYBRID:
                    logger.debug(f"AllReduce 查表失败 ({e}), fallback")
                    return get_empirical()
                raise

    @functools.lru_cache(maxsize=32768)
    def query_nccl(
        self,
        dtype: CommQuantMode = CommQuantMode.half,
        num_gpus: int = 8,
        operation: str = "all_reduce",
        message_size: int = 0,
        database_mode: Optional[DatabaseMode] = None,
    ) -> float:
        """查询 NCCL 集合通信延迟 (ms).

        Args:
            dtype: 通信数据类型
            num_gpus: 参与通信的 GPU 数
            operation: 通信操作类型 ("all_reduce", "all_gather", "reduce_scatter", "alltoall")
            message_size: 消息大小 (元素数)
            database_mode: 查询模式

        Returns:
            latency (ms)
        """

        def get_sol() -> float:
            if num_gpus == 1:
                return 0.0
            p2p_bw = self._get_p2p_bandwidth(num_gpus)
            if operation in ("all_gather", "alltoall", "reduce_scatter"):
                return (
                    dtype.value.memory
                    * message_size
                    * (num_gpus - 1)
                    / num_gpus
                    / p2p_bw
                    * 1000
                )
            elif operation == "all_reduce":
                return (
                    2
                    * dtype.value.memory
                    * message_size
                    * (num_gpus - 1)
                    / num_gpus
                    / p2p_bw
                    * 1000
                )
            return 0.0

        def get_empirical() -> float:
            return get_sol() / 0.8

        mode = database_mode or self._default_database_mode

        if mode == DatabaseMode.SOL:
            return get_sol()
        elif mode == DatabaseMode.EMPIRICAL:
            return get_empirical()
        else:
            try:
                if num_gpus == 1:
                    return 0.0
                if self._nccl_data is None:
                    raise ValueError("NCCL 数据未加载")

                max_measured_gpus = max(self._nccl_data[dtype][operation].keys())
                nccl_dict = self._nccl_data[dtype][operation][min(num_gpus, max_measured_gpus)]
                size_left, size_right = self._nearest_1d_point(
                    message_size, list(nccl_dict.keys()), inner_only=False
                )
                lat = self._interp_1d(
                    [size_left, size_right],
                    [nccl_dict[size_left], nccl_dict[size_right]],
                    message_size,
                )

                # GPU 数超出实测范围时的修正
                if num_gpus > max_measured_gpus:
                    max_bw = self._get_p2p_bandwidth(max_measured_gpus)
                    target_bw = self._get_p2p_bandwidth(num_gpus)
                    scale = (
                        (num_gpus - 1)
                        / num_gpus
                        * max_measured_gpus
                        / (max_measured_gpus - 1)
                        * max_bw
                        / target_bw
                    )
                    lat *= scale

                return lat
            except Exception as e:
                if mode == DatabaseMode.HYBRID:
                    logger.debug(f"NCCL 查表失败 ({e}), fallback")
                    return get_empirical()
                raise

    def query_mem_op(
        self, mem_bytes: int, database_mode: Optional[DatabaseMode] = None
    ) -> float:
        """查询内存操作延迟 (ms).

        纯分析模型 (无实测数据), 基于内存带宽计算。

        Args:
            mem_bytes: 内存操作数据量 (字节)
            database_mode: 查询模式

        Returns:
            latency (ms)
        """

        def get_sol() -> float:
            return mem_bytes / self.system_spec["gpu"]["mem_bw"] * 1000

        def get_empirical() -> float:
            gpu = self.system_spec["gpu"]
            return (
                mem_bytes / (gpu["mem_bw"] * gpu["mem_bw_empirical_scaling_factor"])
                + gpu["mem_empirical_constant_latency"]
            ) * 1000

        mode = database_mode or self._default_database_mode

        if mode == DatabaseMode.SOL:
            return get_sol()
        else:
            return get_empirical()

    def query_p2p(
        self, message_bytes: int, database_mode: Optional[DatabaseMode] = None
    ) -> float:
        """查询点对点 (P2P) 通信延迟 (ms).

        纯分析模型, 基于节点间带宽和延迟。

        Args:
            message_bytes: 消息大小 (字节)
            database_mode: 查询模式

        Returns:
            latency (ms)
        """

        def get_sol() -> float:
            return message_bytes / self.system_spec["node"]["inter_node_bw"] * 1000

        def get_empirical() -> float:
            node = self.system_spec["node"]
            return (message_bytes / node["inter_node_bw"] + node["p2p_latency"]) * 1000

        mode = database_mode or self._default_database_mode

        if mode == DatabaseMode.SOL:
            return get_sol()
        else:
            return get_empirical()

    # ---- 便捷查询: 自动推断查询方法 ----

    def query(
        self,
        op_type: str,
        database_mode: Optional[DatabaseMode] = None,
        **kwargs,
    ) -> float:
        """统一查询接口: 给定算子类型和参数，自动分发到对应的查询方法.

        这是最常用的接口，根据 op_type 自动选择查表方法并传递参数。

        Args:
            op_type: 算子类型，支持:
                - "gemm" / "matmul": GEMM 查询 (需要 m, n, k, quant_mode)
                - "context_attention": Context attention (需要 b, s, n, n_kv, ...)
                - "generation_attention": Generation attention (需要 b, s, n, n_kv, ...)
                - "allreduce": Custom allreduce (需要 tp_size, size, ...)
                - "nccl": NCCL 通信 (需要 num_gpus, operation, message_size, ...)
                - "mem_op" / "memory": 内存操作 (需要 mem_bytes)
                - "p2p": 点对点通信 (需要 message_bytes)
            database_mode: 查询模式 (默认使用数据库设置)
            **kwargs: 传递给具体查询方法的参数

        Returns:
            latency (ms)

        Example:
            >>> db.query("gemm", m=4096, n=4096, k=4096, quant_mode=GEMMQuantMode.fp8)
            >>> db.query("allreduce", tp_size=8, size=16384)
            >>> db.query("mem_op", mem_bytes=1024*1024)
        """
        op = op_type.lower()

        if op in ("gemm", "matmul"):
            return self.query_gemm(
                m=kwargs["m"],
                n=kwargs["n"],
                k=kwargs["k"],
                quant_mode=kwargs.get("quant_mode", GEMMQuantMode.float16),
                database_mode=database_mode,
            )
        elif op == "context_attention":
            return self.query_context_attention(
                b=kwargs["b"],
                s=kwargs["s"],
                prefix=kwargs.get("prefix", 0),
                n=kwargs.get("n", 32),
                n_kv=kwargs.get("n_kv", 8),
                kvcache_quant_mode=kwargs.get("kvcache_quant_mode", KVCacheQuantMode.float16),
                fmha_quant_mode=kwargs.get("fmha_quant_mode", FMHAQuantMode.float16),
                database_mode=database_mode,
                window_size=kwargs.get("window_size", 0),
                head_size=kwargs.get("head_size", 128),
            )
        elif op == "generation_attention":
            return self.query_generation_attention(
                b=kwargs["b"],
                s=kwargs["s"],
                n=kwargs.get("n", 32),
                n_kv=kwargs.get("n_kv", 8),
                kvcache_quant_mode=kwargs.get("kvcache_quant_mode", KVCacheQuantMode.float16),
                database_mode=database_mode,
                window_size=kwargs.get("window_size", 0),
                head_size=kwargs.get("head_size", 128),
            )
        elif op in ("allreduce", "custom_allreduce"):
            return self.query_custom_allreduce(
                quant_mode=kwargs.get("quant_mode", CommQuantMode.half),
                tp_size=kwargs.get("tp_size", 8),
                size=kwargs.get("size", 0),
                database_mode=database_mode,
            )
        elif op == "nccl":
            return self.query_nccl(
                dtype=kwargs.get("dtype", CommQuantMode.half),
                num_gpus=kwargs.get("num_gpus", 8),
                operation=kwargs.get("operation", "all_reduce"),
                message_size=kwargs.get("message_size", 0),
                database_mode=database_mode,
            )
        elif op in ("mem_op", "memory"):
            return self.query_mem_op(
                mem_bytes=kwargs["mem_bytes"],
                database_mode=database_mode,
            )
        elif op == "p2p":
            return self.query_p2p(
                message_bytes=kwargs["message_bytes"],
                database_mode=database_mode,
            )
        else:
            raise ValueError(f"未知算子类型: {op_type}")

    def __repr__(self) -> str:
        return f"PerfDatabase(system='{self.system}', backend='{self.backend}', version='{self.version}')"

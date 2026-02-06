"""Blueprinting 核心计算模块

提供符号化计算基础设施：

符号类型:
- SymPick: 符号条件表达式 (if-then-else)
- SymMax/SymMin: 惰性符号最大/最小值
- LazyExprMixin: 惰性表达式分析方法基类

预定义符号:
- 模型参数: BATCH, SEQ, HIDDEN, FEEDFORWARD, NUM_LAYERS, ATTN_HEADS, HEAD_DIM, VOCAB_SIZE
- 并行策略: TP, PP, DP, CP, EP
- 执行参数: MICRO_BATCH, NUM_MICRO_BATCHES
- 硬件参数: PEAK_FLOPS, MEM_BANDWIDTH, NET_BANDWIDTH

工具函数:
- sym_max/sym_min: 创建惰性最大/最小值
- eval_lazy: 统一的表达式求值
- is_symbolic/is_concrete: 检查表达式是否包含符号
- free_symbols: 获取自由符号集合
- substitute: 符号替换

分析方法 (LazyExprMixin):
- sweep(): 参数空间扫描
- sensitivity(): 敏感度分析（求偏导）
- solve(): 约束求解
- to_latex(): 导出 LaTeX 公式
- to_sympy(): 转换为 SymPy 表达式

上下文管理:
- SubsContext: 符号替换上下文
- ExprCache: 表达式求值缓存
"""
from .symbolic import (
    # 基类
    LazyExprMixin,
    # 符号类型
    SymPick,
    SymMax,
    SymMin,
    # 预定义符号 - 模型参数
    BATCH,
    SEQ,
    HIDDEN,
    FEEDFORWARD,
    NUM_LAYERS,
    ATTN_HEADS,
    HEAD_DIM,
    VOCAB_SIZE,
    # 预定义符号 - 并行策略
    TP,
    PP,
    DP,
    CP,
    EP,
    # 预定义符号 - 执行参数
    MICRO_BATCH,
    NUM_MICRO_BATCHES,
    # 预定义符号 - 硬件参数
    PEAK_FLOPS,
    MEM_BANDWIDTH,
    NET_BANDWIDTH,
    # 符号集合
    MODEL_SYMBOLS,
    PARALLEL_SYMBOLS,
    EXEC_SYMBOLS,
    HARDWARE_SYMBOLS,
    ALL_SYMBOLS,
    # 工具函数
    get_symbol,
    sym_max,
    sym_min,
    eval_lazy,
    is_symbolic,
    is_concrete,
    free_symbols,
    substitute,
    # 缓存管理
    clear_expr_cache,
    get_cache_stats,
)
from .context import SubsContext
from .cache import ExprCache, expr_cache

__all__ = [
    # 基类
    "LazyExprMixin",
    # 符号类型
    "SymPick",
    "SymMax",
    "SymMin",
    # 预定义符号 - 模型参数
    "BATCH",
    "SEQ",
    "HIDDEN",
    "FEEDFORWARD",
    "NUM_LAYERS",
    "ATTN_HEADS",
    "HEAD_DIM",
    "VOCAB_SIZE",
    # 预定义符号 - 并行策略
    "TP",
    "PP",
    "DP",
    "CP",
    "EP",
    # 预定义符号 - 执行参数
    "MICRO_BATCH",
    "NUM_MICRO_BATCHES",
    # 预定义符号 - 硬件参数
    "PEAK_FLOPS",
    "MEM_BANDWIDTH",
    "NET_BANDWIDTH",
    # 符号集合
    "MODEL_SYMBOLS",
    "PARALLEL_SYMBOLS",
    "EXEC_SYMBOLS",
    "HARDWARE_SYMBOLS",
    "ALL_SYMBOLS",
    # 工具函数
    "get_symbol",
    "sym_max",
    "sym_min",
    "eval_lazy",
    "is_symbolic",
    "is_concrete",
    "free_symbols",
    "substitute",
    # 缓存管理
    "clear_expr_cache",
    "get_cache_stats",
    # 上下文
    "SubsContext",
    "ExprCache",
    "expr_cache",
]

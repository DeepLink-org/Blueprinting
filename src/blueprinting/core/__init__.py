"""Blueprinting 核心计算模块

符号类型 (均基于 sympy.Function，惰性求值):
- SymPick: 条件表达式 (if-then-else)
- SymMax / SymMin: 惰性最大 / 最小值

预定义符号 (按类组织: Model, Parallel, Exec, Hardware):
- 也可扁平导入: from blueprinting.core import BATCH, TP, ...
"""
from .symbols import (
    Model, Parallel, Exec, Hardware,
    MODEL_SYMBOLS, PARALLEL_SYMBOLS, EXEC_SYMBOLS, HARDWARE_SYMBOLS, ALL_SYMBOLS,
    BATCH, SEQ, HIDDEN, FEEDFORWARD, NUM_LAYERS, ATTN_HEADS, HEAD_DIM, VOCAB_SIZE,
    TP, PP, DP, CP, EP,
    MICRO_BATCH, NUM_MICRO_BATCHES,
    PEAK_FLOPS, MEM_BANDWIDTH, NET_BANDWIDTH,
)
from .symbolic import (
    SymPick, SymMax, SymMin,
    get_symbol, sym_max, sym_min, eval_lazy,
)

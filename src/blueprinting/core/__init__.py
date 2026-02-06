"""Blueprinting 核心计算模块

符号类型 (均基于 sympy.Function，惰性求值):
- SymPick: 条件表达式 (if-then-else)
- SymMax / SymMin: 惰性最大 / 最小值

预定义符号 (按类组织: Model, Parallel, Exec, Hardware):
- 也可扁平导入: from blueprinting.core import BATCH, TP, ...
"""

from .symbolic import SymMax, SymMin, SymPick, eval_lazy, get_symbol, sym_max, sym_min
from .symbols import (
                       ALL_SYMBOLS,
                       ATTN_HEADS,
                       BATCH,
                       CP,
                       DP,
                       EP,
                       EXEC_SYMBOLS,
                       FEEDFORWARD,
                       HARDWARE_SYMBOLS,
                       HEAD_DIM,
                       HIDDEN,
                       MEM_BANDWIDTH,
                       MICRO_BATCH,
                       MODEL_SYMBOLS,
                       NET_BANDWIDTH,
                       NUM_LAYERS,
                       NUM_MICRO_BATCHES,
                       PARALLEL_SYMBOLS,
                       PEAK_FLOPS,
                       PP,
                       SEQ,
                       TP,
                       VOCAB_SIZE,
                       Exec,
                       Hardware,
                       Model,
                       Parallel,
)

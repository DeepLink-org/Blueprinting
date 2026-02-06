"""预定义符号常量

按类别组织仿真建模中的符号，便于引用和批量操作。

    from blueprinting.core import Model, Parallel, BATCH, TP, ALL_SYMBOLS
"""

from sympy import Symbol


class Model:
    """模型结构相关符号."""
    BATCH = Symbol("batch", positive=True, integer=True)
    SEQ = Symbol("seq", positive=True, integer=True)
    HIDDEN = Symbol("hidden", positive=True, integer=True)
    FEEDFORWARD = Symbol("feedforward", positive=True, integer=True)
    NUM_LAYERS = Symbol("num_layers", positive=True, integer=True)
    ATTN_HEADS = Symbol("attn_heads", positive=True, integer=True)
    HEAD_DIM = Symbol("head_dim", positive=True, integer=True)
    VOCAB_SIZE = Symbol("vocab_size", positive=True, integer=True)


class Parallel:
    """并行策略相关符号."""
    TP = Symbol("tp", positive=True, integer=True)
    PP = Symbol("pp", positive=True, integer=True)
    DP = Symbol("dp", positive=True, integer=True)
    CP = Symbol("cp", positive=True, integer=True)
    EP = Symbol("ep", positive=True, integer=True)


class Exec:
    """执行 / 调度相关符号."""
    MICRO_BATCH = Symbol("micro_batch", positive=True, integer=True)
    NUM_MICRO_BATCHES = Symbol("num_micro_batches", positive=True, integer=True)


class Hardware:
    """硬件能力相关符号."""
    PEAK_FLOPS = Symbol("peak_flops", positive=True)
    MEM_BANDWIDTH = Symbol("mem_bandwidth", positive=True)
    NET_BANDWIDTH = Symbol("net_bandwidth", positive=True)


# 符号集合
MODEL_SYMBOLS = {
    Model.BATCH, Model.SEQ, Model.HIDDEN, Model.FEEDFORWARD,
    Model.NUM_LAYERS, Model.ATTN_HEADS, Model.HEAD_DIM, Model.VOCAB_SIZE,
}
PARALLEL_SYMBOLS = {Parallel.TP, Parallel.PP, Parallel.DP, Parallel.CP, Parallel.EP}
EXEC_SYMBOLS = {Exec.MICRO_BATCH, Exec.NUM_MICRO_BATCHES}
HARDWARE_SYMBOLS = {Hardware.PEAK_FLOPS, Hardware.MEM_BANDWIDTH, Hardware.NET_BANDWIDTH}
ALL_SYMBOLS = MODEL_SYMBOLS | PARALLEL_SYMBOLS | EXEC_SYMBOLS | HARDWARE_SYMBOLS

# 扁平别名
BATCH = Model.BATCH
SEQ = Model.SEQ
HIDDEN = Model.HIDDEN
FEEDFORWARD = Model.FEEDFORWARD
NUM_LAYERS = Model.NUM_LAYERS
ATTN_HEADS = Model.ATTN_HEADS
HEAD_DIM = Model.HEAD_DIM
VOCAB_SIZE = Model.VOCAB_SIZE
TP = Parallel.TP
PP = Parallel.PP
DP = Parallel.DP
CP = Parallel.CP
EP = Parallel.EP
MICRO_BATCH = Exec.MICRO_BATCH
NUM_MICRO_BATCHES = Exec.NUM_MICRO_BATCHES
PEAK_FLOPS = Hardware.PEAK_FLOPS
MEM_BANDWIDTH = Hardware.MEM_BANDWIDTH
NET_BANDWIDTH = Hardware.NET_BANDWIDTH

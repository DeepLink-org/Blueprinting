import streamlit as st
import hyperparameter as hp
from streamlit_extras.row import row
from sympy import Symbol

from blueprinting.nn import (
    AddDef,
    ColumnParallelLinear,
    MulDef,
    RMSNormDef,
    RowParallelLinear,
    SiLUDef,
    SequenceParallelAdd,
    SequenceParallelRMSNorm,
    TensorDef,
)
from blueprinting.st.blocks.container import block
from blueprinting.st.blocks.formatter import DefaultFormatter, auto_symbol
from blueprinting.types.base import DType
from blueprinting.util import pick


@hp.param("model")
def ffn_block(hidden=1024, feedforward=1024):
    """FFN 模块计算

    使用sympy的符号化计算，方便追踪计算过程：
    - 通过 `hp.param` 将全局超参中的`model.hidden`与`model.feedforward`映射给`hidden`和`feedforward`
    - 创建参数的（数值、符号）对，比如（hidden、HIDDEN）
    - 运算过程使用符号进行计算，但输出时同时显示数值与符号
    """

    # 读取hyperparameter配置，并创建 数值<->符号 对
    dtype = DType(hp.scope.exe.datatype | "float16")
    seqlen = hp.scope.model.seq_size | 64
    bsize, BSIZE = hp.scope.exe.microbatch_size | 0, Symbol("bsize")
    seqlen, SEQLEN = seqlen, Symbol("seqlen")
    hidden, HIDDEN = hidden, Symbol("hidden")
    feedforward, FEEDFORWARD = feedforward, Symbol("feedforward")
    tpsize, TPSIZE = hp.scope.exe.tensor_par | 1, Symbol("tpsize")

    tensor_par_comm_type = (
        hp.scope.exe.tensor_par_comm_type | "ar"
    )  # "ar" 表示 all reduce， "rs_ag" 表示 reduce scatter and all gather
    sequence_par = hp.scope.exe.sequence_par | False

    # 创建层级容器
    with block("ffn_block", 4):
        st.markdown(f"#### Feed Forward Block")

        with block("ffn_res_block", 2):
            st.markdown("#### Residual Connection")
            res_layer = pick(
                sequence_par,
                SequenceParallelAdd(dtype, tensor_model_parallel_size=TPSIZE),
                AddDef(dtype),
            )
            res_out = res_layer(
                TensorDef([BSIZE, SEQLEN, HIDDEN], dtype=dtype),
                TensorDef([BSIZE, SEQLEN, HIDDEN], dtype=dtype),
            )
            res_out | DefaultFormatter.with_subs(auto_symbol())

        with block("ffn_pro_block", 2):
            st.markdown("##### Down Projection [Linear]")

            # 创建输出层的定义
            output_layer = RowParallelLinear(dtype, FEEDFORWARD, HIDDEN, False, TPSIZE, tensor_par_comm_type)

            # 进行计算
            output_proj = output_layer(TensorDef(shape=[BSIZE, SEQLEN, FEEDFORWARD], dtype=dtype))

            # 渲染结果
            output_proj | DefaultFormatter.with_subs(auto_symbol())

        with block("ffn_SwiGLU_mul", 2):
            st.markdown("##### SwiGLU Element-wise Mul: silu(gate) * up")
            mul_layer = MulDef(dtype)
            mul_out = mul_layer(
                TensorDef([BSIZE, SEQLEN, FEEDFORWARD], dtype=dtype),  # silu(gate)
                TensorDef([BSIZE, SEQLEN, FEEDFORWARD], dtype=dtype),  # up
            )
            mul_out | DefaultFormatter.with_subs(auto_symbol())

        with block("ffn_SwiGLU", 2):
            st.markdown("##### SiLU Activation on Gate")
            silu_layer = SiLUDef(dtype)
            silu_out = silu_layer(TensorDef([BSIZE, SEQLEN, FEEDFORWARD], dtype=dtype))
            silu_out | DefaultFormatter.with_subs(auto_symbol())

        r = row(2)

        with r.container():
            with block("ffn_input_block", 2):
                st.markdown("##### Up Projection [Linear]")
                input_layer = ColumnParallelLinear(dtype, HIDDEN, FEEDFORWARD, False, TPSIZE, tensor_par_comm_type)
                input_proj = input_layer(TensorDef(shape=[BSIZE, SEQLEN, HIDDEN], dtype=dtype))
                input_proj | DefaultFormatter.with_subs(auto_symbol())

        with r.container():
            with block("ffn_gate_block", 2):
                st.markdown("##### Gate Projection [Linear]")
                gate_layer = ColumnParallelLinear(dtype, HIDDEN, FEEDFORWARD, False, TPSIZE, tensor_par_comm_type)
                gate_proj = gate_layer(TensorDef(shape=[BSIZE, SEQLEN, HIDDEN], dtype=dtype))
                gate_proj | DefaultFormatter.with_subs(auto_symbol())

        with block("ffn_rms_block", 2):
            st.markdown("##### Pre-Norm [RMSNorm]")
            rms_layer = pick(
                sequence_par,
                SequenceParallelRMSNorm(dtype, HIDDEN, tensor_model_parallel_size=TPSIZE),
                RMSNormDef(dtype, HIDDEN),
            )
            rms_out = rms_layer(TensorDef([BSIZE, SEQLEN, HIDDEN], dtype=dtype))
            rms_out | DefaultFormatter.with_subs(auto_symbol())

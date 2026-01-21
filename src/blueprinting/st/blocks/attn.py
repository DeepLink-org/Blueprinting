import streamlit as st
import hyperparameter as hp
from streamlit_extras.row import row
from sympy import Symbol

from blueprinting.nn import (
    AddDef,
    BatchMatmulDef,
    ColumnParallelLinear,
    LinearDef,
    RMSNormDef,
    RowParallelLinear,
    SoftmaxDef,
    SequenceParallelAdd,
    SequenceParallelRMSNorm,
    TensorDef,
)
from blueprinting.st.blocks.container import block
from blueprinting.st.blocks.formatter import DefaultFormatter, auto_symbol
from blueprinting.types.base import DType
from blueprinting.util import pick


@hp.param("model")
def attention_block():
    """Attention 模块计算

    参考llama2中的Attention模块
    """
    # 读取hyperparameter的参数配置， 创建 数值:符号 对
    dtype = DType(hp.scope.exe.datatype | "float16")
    seqlen, SEQLEN = hp.scope.model.seq_size | 4096, Symbol("seqlen")
    bsize, BSIZE = hp.scope.exe.microbatch_size | 0, Symbol("bsize")
    hidden, HIDDEN = hp.scope.model.hidden | 4096, Symbol("hidden")
    attnheads, ATTNHEADS = hp.scope.model.attn_heads | 32, Symbol("attnheads")
    attnsize, ATTNSIZE = (
        hp.scope.model.attn_size | 128,
        Symbol("attnsize"),
    )  # hidden // attn_heads
    tpsize, TPSIZE = hp.scope.exe.tensor_par | 1, Symbol("tpsize")

    tensor_par_comm_type = (
        hp.scope.exe.tensor_par_comm_type | "ar"
    )  # "ar" 表示 all reduce， "rs_ag" 表示 reduce scatter and all gather
    sequence_par = hp.scope.exe.sequence_par | "true"

    #  创建层级容器
    with block("attn_block", 4):
        st.markdown(f"#### Attention Block")

        with block("attn_res_block", 2):
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

        with block("attn_linear_block", 2):
            st.markdown("#### Output Projection [Linear]")
            linear_layer = RowParallelLinear(dtype, HIDDEN, HIDDEN, False, TPSIZE, tensor_par_comm_type)
            linear_out = linear_layer(TensorDef([BSIZE, SEQLEN, HIDDEN], dtype=dtype))
            linear_out | DefaultFormatter.with_subs(auto_symbol())

        with block("attn_mha_block", 2):
            st.markdown("#### Multi-Head Attention")
            # TP 切分后，每个设备上的 heads 数量
            ATTNHEADS_PER_TP = ATTNHEADS / TPSIZE

            with block("attn_batch_matmul_block_1", 2):
                st.markdown("#### Score @ V [BatchMatmul]")
                batchmatmul_layer = BatchMatmulDef(dtype)
                # score: [B, heads/TP, S, S], V: [B, heads/TP, S, size]
                # output: [B, heads/TP, S, size]
                batchmatmul_out = batchmatmul_layer(
                    TensorDef([BSIZE, ATTNHEADS_PER_TP, SEQLEN, SEQLEN], dtype=dtype),
                    TensorDef([BSIZE, ATTNHEADS_PER_TP, SEQLEN, ATTNSIZE], dtype=dtype),
                )
                batchmatmul_out | DefaultFormatter.with_subs(auto_symbol())

            with block("attn_softmax_block", 2):
                st.markdown("#### QK Score [Softmax]")
                softmax_layer = SoftmaxDef(dtype)
                softmax_out = softmax_layer(TensorDef([BSIZE, ATTNHEADS_PER_TP, SEQLEN, SEQLEN], dtype=dtype))
                softmax_out | DefaultFormatter.with_subs(auto_symbol())

            with block("attn_batch_matmul_block_2", 2):
                st.markdown("#### Q @ K^T [BatchMatmul]")
                batchmatmul_layer = BatchMatmulDef(dtype)
                # Q: [B, heads/TP, S, size], K^T: [B, heads/TP, size, S]
                # output: [B, heads/TP, S, S]
                batchmatmul_out = batchmatmul_layer(
                    TensorDef([BSIZE, ATTNHEADS_PER_TP, SEQLEN, ATTNSIZE], dtype=dtype),
                    TensorDef([BSIZE, ATTNHEADS_PER_TP, ATTNSIZE, SEQLEN], dtype=dtype),
                )
                batchmatmul_out | DefaultFormatter.with_subs(auto_symbol())

        r = row(3)

        with r.container():
            with block("attn_query_block", 2):
                st.markdown("#### Query Projection [Linear]")

                q_layer = ColumnParallelLinear(dtype, HIDDEN, HIDDEN, False, TPSIZE, tensor_par_comm_type)

                q_proj = q_layer(TensorDef(shape=[BSIZE, SEQLEN, HIDDEN], dtype=dtype))

                q_proj | DefaultFormatter.with_subs(auto_symbol())

        with r.container():
            with block("attn_key_block", 2):
                st.markdown("#### Key Projection [Linear]")
                k_layer = ColumnParallelLinear(dtype, HIDDEN, HIDDEN, False, TPSIZE, tensor_par_comm_type)

                k_proj = k_layer(TensorDef(shape=[BSIZE, SEQLEN, HIDDEN], dtype=dtype))

                k_proj | DefaultFormatter.with_subs(auto_symbol())

        with r.container():
            with block("attn_value_block", 2):
                st.markdown("#### Value Projection [Linear]")
                v_layer = ColumnParallelLinear(dtype, HIDDEN, HIDDEN, False, TPSIZE, tensor_par_comm_type)

                v_proj = v_layer(TensorDef(shape=[BSIZE, SEQLEN, HIDDEN], dtype=dtype))

                v_proj | DefaultFormatter.with_subs(auto_symbol())

        with block("attn_rms_block", 2):
            st.markdown("#### PreNorm [RMSNorm]")
            rms_layer = pick(
                sequence_par,
                SequenceParallelRMSNorm(dtype, HIDDEN, tensor_model_parallel_size=TPSIZE),
                RMSNormDef(dtype, HIDDEN),
            )
            rms_out = rms_layer(TensorDef([BSIZE, SEQLEN, HIDDEN], dtype=dtype))
            rms_out | DefaultFormatter.with_subs(auto_symbol())

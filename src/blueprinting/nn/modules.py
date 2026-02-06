from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import List, Tuple, Union

import hyperparameter as hp
from sympy import Expr

from blueprinting.core import SymPick
from blueprinting.nn.base import LayerDef, c2c_nbytes, c2c_throughput, c2c_times
from blueprinting.types.base import DType, TensorDef
from blueprinting.util import pick


@hp.param("blueprinting.flops")
def gemm_flops(
    A: TensorDef, B: TensorDef, with_bias=False, count_add=True
) -> Union[int, Expr]:
    """Calculate flops for GEMM

    Examples
    --------
    >>> from sympy import symbols
    >>> a, b, c, d, e, f, g = symbols("a b c d e f g")
    >>> gemm_flops(TensorDef([a, b]), TensorDef([b, c]))
    a*b*c
    >>> gemm_flops(TensorDef([a, b]), TensorDef([b, c]), count_add=True)
    a*b*(2*c - 1)
    >>> gemm_flops(TensorDef([a, b]), TensorDef([b, c]), with_bias=True, count_add=True)
    2*a*b*c
    >>> gemm_flops(TensorDef([a, b, c]), TensorDef([c, d]))
    a*b*c*d
    >>> gemm_flops(TensorDef([a, b, c]), TensorDef([c, d]), count_add=True)
    a*b*c*(2*d - 1)
    >>> gemm_flops(TensorDef([a, b, c]), TensorDef([c, d]), with_bias=True, count_add=True)
    2*a*b*c*d
    >>> gemm_flops(TensorDef([a, b, c]), TensorDef([c, b, d]))
    a*b*c*d
    >>> gemm_flops(TensorDef([a, b, c]), TensorDef([c, b, d]), count_add=True)
    a*b*c*(2*d - 1)
    >>> gemm_flops(TensorDef([a, b, c]), TensorDef([c, b, d]), with_bias=True, count_add=True)
    2*a*b*c*d
    """
    if A.shape[-1] != B.shape[0]:
        raise Exception(f"bad gemm: {A} x {B}: {A.shape[-1]} != {B.shape[0]}")
    offset = len(B.shape) - 1
    for idx, (a, b) in enumerate(zip(reversed(A.shape), B.shape[:-1])):
        if a != b:
            offset = idx
            break

    if count_add:
        if with_bias:
            return reduce(mul, A.shape) * 2 * reduce(mul, B.shape[offset:])
        return reduce(mul, A.shape) * (2 * reduce(mul, B.shape[offset:]) - 1)
    return reduce(mul, A.shape) * reduce(mul, B.shape[offset:])


@hp.param("blueprinting.flops")
def batch_gemm_flops(
    A: TensorDef, B: TensorDef, with_bias=False, count_add=True
) -> Union[int, Expr]:
    if A.shape[-1] != B.shape[-2]:
        raise Exception(f"bad batch_gemm: {A} x {B}: {A.shape[-1]} != {B.shape[-2]}")

    if count_add:
        if with_bias:
            return reduce(mul, A.shape) * 2 * B.shape[-1]
        return reduce(mul, A.shape) * (2 * (B.shape[-1]) - 1)
    return reduce(mul, A.shape) * B.shape[-1]


@dataclass
class LinearDef(LayerDef):
    dtype: DType
    in_features: Union[int, Expr]
    out_features: Union[int, Expr]
    bias: bool = True

    def __post_init__(self):
        self.weight = TensorDef((self.in_features, self.out_features), dtype=self.dtype)

    def forward(self, *inputs) -> TensorDef:
        return TensorDef(inputs[0].shape[:-1] + [self.out_features], inputs[0].dtype)

    @property
    def nelems(self):
        w = self.weight.nelems
        return w + self.out_features if self.bias else w

    @property
    @hp.param("blueprinting.layerdef")
    def flops_fw(self, inputs: List[TensorDef] = None):
        if inputs is None:
            inputs = []
        if not inputs:
            return 0
        return gemm_flops(inputs[0], self.weight, with_bias=self.bias)

    @property
    @hp.param("blueprinting.layerdef")
    def flops_bw(self, inputs: List[TensorDef] = None):
        if inputs is None:
            inputs = []
        if not inputs:
            return 0
        grad_output = self(*inputs).belike()
        act_flops = gemm_flops(grad_output, self.weight.T, with_bias=self.bias)
        w_flops = gemm_flops(inputs[0].T, grad_output, with_bias=self.bias)
        return act_flops + w_flops


@dataclass
class ColumnParallelLinear(LinearDef):
    tensor_model_parallel_size: int = 1
    tensor_par_comm_type: str = "ar"

    def __post_init__(self):
        self.weight = TensorDef(
            (self.in_features, self.out_features / self.tensor_model_parallel_size),
            dtype=self.dtype,
        )

    def forward(self, *inputs) -> TensorDef:
        return TensorDef(
            inputs[0].shape[:-1]
            + [self.out_features / self.tensor_model_parallel_size],
            inputs[0].dtype,
        )

    @property
    @hp.param("blueprinting.layerdef")
    def flops_bw(self, inputs: List[TensorDef] = None):
        """ColumnParallelLinear backward FLOPs (不包含通信开销)"""
        if inputs is None:
            inputs = []
        if not inputs:
            return 0
        grad_output = self(*inputs).belike()
        act_flops = gemm_flops(grad_output, self.weight.T, with_bias=self.bias)
        w_flops = gemm_flops(inputs[0].T, grad_output, with_bias=self.bias)
        # 注意: all_reduce 是通信开销，不应计入 FLOPs
        return act_flops + w_flops

    @property
    @hp.param("blueprinting.layerdef")
    def c2c_fw(self, inputs: List[TensorDef] = None) -> int:
        """ColumnParallelLinear forward c2c: rs_ag 模式下才有通信"""
        if inputs is None:
            inputs = []
        if not inputs:
            return 0
        nbytes = self.forward(*inputs).nbytes
        # rs_ag 模式且 TP>1 时有 all_gather 通信
        if self.tensor_par_comm_type == "rs_ag":
            return SymPick(self.tensor_model_parallel_size > 1, nbytes, 0)
        return 0

    @property
    @hp.param("blueprinting.layerdef")
    def c2c_bw(self, inputs: List[TensorDef] = None) -> int:
        """ColumnParallelLinear backward c2c: TP>1 时有通信"""
        if inputs is None:
            inputs = []
        if not inputs:
            return 0
        nbytes = self.forward(*inputs).nbytes
        return SymPick(self.tensor_model_parallel_size > 1, nbytes, 0)

    @property
    @hp.param("blueprinting.layerdef")
    def time_c2c_fw(self, inputs: List[TensorDef] = None) -> int:
        if inputs is None:
            inputs = []
        c2c = self.c2c_fw
        comm_type = pick(self.tensor_par_comm_type == "rs_ag", "all_gather", "identity")
        throughput = c2c_throughput()
        comm_size = c2c_nbytes(c2c, comm_type, self.tensor_model_parallel_size)
        return c2c_times(comm_type, comm_size, throughput)

    @property
    @hp.param("blueprinting.layerdef")
    def time_c2c_bw(self, inputs: List[TensorDef] = None) -> int:
        if inputs is None:
            inputs = []
        c2c = self.c2c_bw
        comm_type = pick(
            self.tensor_par_comm_type == "rs_ag", "reduce_scatter", "all_reduce"
        )
        throughput = c2c_throughput()
        comm_size = c2c_nbytes(c2c, comm_type, self.tensor_model_parallel_size)
        return c2c_times(comm_type, comm_size, throughput)

    @property
    @hp.param("blueprinting.layerdef")
    def placement_weight(self, inputs: List[TensorDef] = None) -> Tuple[TensorDef, ...]:
        if inputs is None:
            inputs = []
        row = self.in_features
        column = self.out_features / self.tensor_model_parallel_size
        return [TensorDef([row, column], dtype=self.dtype)]


@dataclass
class RowParallelLinear(LinearDef):
    tensor_model_parallel_size: int = 1
    tensor_par_comm_type: str = "ar"

    def __post_init__(self):
        self.weight = TensorDef(
            (self.in_features / self.tensor_model_parallel_size, self.out_features),
            dtype=self.dtype,
        )

    def forward(self, *inputs) -> TensorDef:
        return TensorDef(inputs[0].shape[:-1] + [self.out_features], inputs[0].dtype)

    @property
    @hp.param("blueprinting.layerdef")
    def nbytes_input(self, inputs: List[TensorDef] = None):
        if inputs is None:
            inputs = []
        input = TensorDef(
            inputs[0].shape[:-1] + [self.in_features / self.tensor_model_parallel_size],
            inputs[0].dtype,
        )
        return input.nbytes

    @property
    @hp.param("blueprinting.layerdef")
    def nbytes_activity(self, inputs: List[TensorDef] = None):
        if inputs is None:
            inputs = []
        input = TensorDef(
            inputs[0].shape[:-1] + [self.in_features / self.tensor_model_parallel_size],
            inputs[0].dtype,
        )
        return input.nbytes

    @property
    @hp.param("blueprinting.layerdef")
    def flops_fw(self, inputs: List[TensorDef] = None) -> int:
        """RowParallelLinear forward FLOPs (不包含通信开销)"""
        if inputs is None:
            inputs = []
        input = TensorDef(
            inputs[0].shape[:-1] + [self.in_features / self.tensor_model_parallel_size],
            inputs[0].dtype,
        )
        output_flops = gemm_flops(input, self.weight, with_bias=self.bias)
        # 注意: all_reduce 是通信开销，不应计入 FLOPs
        return output_flops

    @property
    @hp.param("blueprinting.layerdef")
    def flops_bw(self, inputs: List[TensorDef] = None) -> int:
        if inputs is None:
            inputs = []
        if not inputs:
            return 0
        input = TensorDef(
            inputs[0].shape[:-1] + [self.in_features / self.tensor_model_parallel_size],
            inputs[0].dtype,
        )
        grad_output = self(*inputs).belike()
        act_flops = gemm_flops(grad_output, self.weight.T, with_bias=self.bias)
        w_flops = gemm_flops(input.T, grad_output, with_bias=self.bias)
        return act_flops + w_flops

    @property
    @hp.param("blueprinting.layerdef")
    def c2c_fw(self, inputs: List[TensorDef] = None) -> int:
        """RowParallelLinear forward c2c: TP>1 时有 all_reduce/reduce_scatter 通信"""
        if inputs is None:
            inputs = []
        if not inputs:
            return 0
        nbytes = self.forward(*inputs).nbytes
        return SymPick(self.tensor_model_parallel_size > 1, nbytes, 0)

    @property
    @hp.param("blueprinting.layerdef")
    def c2c_bw(self, inputs: List[TensorDef] = None) -> int:
        """RowParallelLinear backward c2c: rs_ag 模式下才有通信"""
        if inputs is None:
            inputs = []
        if not inputs:
            return 0
        nbytes = self.forward(*inputs).nbytes
        # rs_ag 模式且 TP>1 时有 all_gather 通信
        if self.tensor_par_comm_type == "rs_ag":
            return SymPick(self.tensor_model_parallel_size > 1, nbytes, 0)
        return 0

    @property
    @hp.param("blueprinting.layerdef")
    def time_c2c_fw(self, inputs: List[TensorDef] = None) -> int:
        if inputs is None:
            inputs = []
        c2c = self.c2c_fw
        comm_type = pick(
            self.tensor_par_comm_type == "rs_ag", "reduce_scatter", "all_reduce"
        )
        throughput = c2c_throughput()
        comm_size = c2c_nbytes(c2c, comm_type, self.tensor_model_parallel_size)
        return c2c_times(comm_type, comm_size, throughput)

    @property
    @hp.param("blueprinting.layerdef")
    def time_c2c_bw(self, inputs: List[TensorDef] = None) -> int:
        if inputs is None:
            inputs = []
        c2c = self.c2c_bw
        comm_type = pick(self.tensor_par_comm_type == "rs_ag", "all_gather", "identity")
        throughput = c2c_throughput()
        comm_size = c2c_nbytes(c2c, comm_type, self.tensor_model_parallel_size)
        return c2c_times(comm_type, comm_size, throughput)

    @property
    @hp.param("blueprinting.layerdef")
    def placement_weight(self, inputs: List[TensorDef] = None) -> Tuple[TensorDef, ...]:
        if inputs is None:
            inputs = []
        row = self.in_features / self.tensor_model_parallel_size
        column = self.out_features
        return [TensorDef([row, column], dtype=self.dtype)]


@dataclass
class LayerNormDef(LayerDef):
    dtype: DType
    normalized_shape: List[Union[int, Expr]]
    bias: bool = True

    def __post_init__(self):
        self.weight = TensorDef(self.normalized_shape, dtype=self.dtype)

    def forward(self, *inputs) -> TensorDef:
        return TensorDef(inputs[0].shape, inputs[0].dtype)

    @property
    def nelems(self) -> int:
        return self.normalized_shape * 2 if self.bias else self.normalized_shape

    @property
    @hp.param("blueprinting.layerdef")
    def nbytes_input(self, inputs: List[TensorDef] = None):
        if inputs is None:
            inputs = []
        return inputs[0].nbytes

    @property
    @hp.param("blueprinting.layerdef")
    def nbytes_output(self, inputs: List[TensorDef] = None):
        if inputs is None:
            inputs = []
        return self(*inputs).nbytes

    @property
    @hp.param("blueprinting.layerdef")
    def nbytes_activity_grads(self, inputs: List[TensorDef] = None):
        if inputs is None:
            inputs = []
        return inputs[0].nbytes

    @property
    @hp.param("blueprinting.layerdef")
    def nbytes_weight(self, inputs: List[TensorDef] = None):
        if inputs is None:
            inputs = []
        return self.normalized_shape * 2 * self.dsize

    @property
    @hp.param("blueprinting.layerdef")
    def nbytes_weight_grads(self, inputs: List[TensorDef] = None):
        if inputs is None:
            inputs = []
        return self.normalized_shape * 2 * self.dsize

    @property
    @hp.param("blueprinting.layerdef")
    def flops_fw(self, inputs: List[TensorDef] = None):
        if inputs is None:
            inputs = []
        if not inputs:
            return 0
        num_features = reduce(mul, inputs[0].shape)
        mean_flops = num_features
        var_flops = 2 * num_features
        offset_flops = num_features
        scale_flops = num_features
        bias_flops = num_features if self.bias else 0
        return mean_flops + var_flops + offset_flops + scale_flops + bias_flops

    @property
    @hp.param("blueprinting.layerdef")
    def flops_bw(self, inputs: List[TensorDef] = None):
        """LayerNorm backward: 涉及 mean/var 梯度计算，约 10-12 ops per element

        参考: https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
        """
        if inputs is None:
            inputs = []
        if not inputs:
            return 0
        num_features = reduce(mul, inputs[0].shape)
        # 反向传播涉及: dx_hat, dvar, dmean, dx, dgamma, dbeta
        # 总计约 10-12 ops per element
        return 10 * num_features

    @property
    @hp.param("blueprinting.layerdef")
    def placement_weight(self, inputs: List[TensorDef] = None) -> Tuple[TensorDef, ...]:
        if inputs is None:
            inputs = []
        return [
            TensorDef([self.normalized_shape], dtype=self.dtype),
            TensorDef([self.normalized_shape], dtype=self.dtype),
        ]


@dataclass
class Conv2dDef(LayerDef):
    dtype: str
    in_channels: int
    out_channels: int


"""
    https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
    https://cthorey.github.io./blog/2016/backpropagation/
"""


@dataclass
class RMSNormDef(LayerDef):
    """RMSNorm: Root Mean Square Layer Normalization

    Forward: y = x / sqrt(mean(x^2) + eps) * gamma
    Backward: 需要计算 dx 和 dgamma
    """

    dtype: DType
    normalized_shape: List[Union[int, Expr]]
    eps: float = 1e-5
    elementwise_affine: bool = True

    def forward(self, *inputs):
        return TensorDef(inputs[0].shape, inputs[0].dtype)

    @property
    def nelems(self):
        return self.normalized_shape

    @property
    @hp.param("blueprinting.layerdef")
    def flops_fw(self, inputs: List[TensorDef] = None):
        if inputs is None:
            inputs = []
        if not inputs:
            return 0
        num_features = reduce(mul, inputs[0].shape)
        square_flops = num_features  # 平方运算
        mean_flops = num_features
        div_flops = num_features
        scale_flops = num_features
        return square_flops + div_flops + mean_flops + scale_flops

    @property
    @hp.param("blueprinting.layerdef")
    def flops_bw(self, inputs: List[TensorDef] = None):
        """RMSNorm backward: ~8 ops per element for gradient computation"""
        if inputs is None:
            inputs = []
        if not inputs:
            return 0
        num_features = reduce(mul, inputs[0].shape)
        # Backward pass involves: computing input gradients and weight gradients
        # Similar complexity to forward, approximately 2x forward
        return 8 * num_features

    @property
    @hp.param("blueprinting.layerdef")
    def placement_weight(self, inputs: List[TensorDef] = None) -> Tuple[TensorDef, ...]:
        if inputs is None:
            inputs = []
        return [TensorDef([self.normalized_shape], dtype=self.dtype)]


@dataclass
class SequenceParallelRMSNorm(RMSNormDef):
    tensor_model_parallel_size: int = 1

    def forward(self, *inputs):
        seq_len = inputs[0].shape[-2] / self.tensor_model_parallel_size
        return TensorDef(
            inputs[0].shape[:-2] + [seq_len, inputs[0].shape[-1]], inputs[0].dtype
        )

    @property
    @hp.param("blueprinting.layerdef")
    def nbytes_input(self, inputs: List[TensorDef] = None):
        if inputs is None:
            inputs = []
        seq_len = inputs[0].shape[-2] / self.tensor_model_parallel_size
        input = TensorDef(
            inputs[0].shape[:-2] + [seq_len, inputs[0].shape[-1]], inputs[0].dtype
        )
        return input.nbytes

    @property
    @hp.param("blueprinting.layerdef")
    def nbytes_activity(self, inputs: List[TensorDef] = None):
        if inputs is None:
            inputs = []
        seq_len = inputs[0].shape[-2] / self.tensor_model_parallel_size
        input = TensorDef(
            inputs[0].shape[:-2] + [seq_len, inputs[0].shape[-1]], inputs[0].dtype
        )
        return input.nbytes

    @property
    @hp.param("blueprinting.layerdef")
    def flops_fw(self, inputs: List[TensorDef] = None):
        if inputs is None:
            inputs = []
        if not inputs:
            return 0
        seq_len = inputs[0].shape[-2] / self.tensor_model_parallel_size
        input = TensorDef(
            inputs[0].shape[:-2] + [seq_len, inputs[0].shape[-1]], inputs[0].dtype
        )
        num_features = reduce(mul, input.shape)
        square_flops = num_features  # 平方运算
        mean_flops = num_features
        div_flops = num_features
        scale_flops = num_features
        return square_flops + div_flops + mean_flops + scale_flops

    @property
    @hp.param("blueprinting.layerdef")
    def flops_bw(self, inputs: List[TensorDef] = None):
        """SequenceParallelRMSNorm backward"""
        if inputs is None:
            inputs = []
        if not inputs:
            return 0
        seq_len = inputs[0].shape[-2] / self.tensor_model_parallel_size
        input = TensorDef(
            inputs[0].shape[:-2] + [seq_len, inputs[0].shape[-1]], inputs[0].dtype
        )
        num_features = reduce(mul, input.shape)
        return 8 * num_features

    @property
    @hp.param("blueprinting.layerdef")
    def placement_weight(self, inputs: List[TensorDef] = None) -> Tuple[TensorDef, ...]:
        if inputs is None:
            inputs = []
        return [TensorDef([self.normalized_shape], dtype=self.dtype)]


"""
    https://pytorch.org/docs/stable/generated/torch.bmm.html
"""


@dataclass
class BatchMatmulDef(LayerDef):
    """Batch Matrix Multiplication: torch.bmm

    无权重层，需要两个输入张量进行批量矩阵乘法。
    """

    dtype: DType

    def forward(self, *inputs):
        return TensorDef(inputs[0].shape[:-1] + [inputs[1].shape[-1]], inputs[0].dtype)

    @property
    @hp.param("blueprinting.layerdef")
    def nbytes_activity(self, inputs: List[TensorDef] = None) -> int:
        if inputs is None:
            inputs = []
        if len(inputs) < 2:
            return 0
        return (
            reduce(mul, inputs[0].shape) + reduce(mul, inputs[1].shape)
        ) * self.dsize

    @property
    @hp.param("blueprinting.layerdef")
    def nbytes_activity_grads(self, inputs: List[TensorDef] = None) -> int:
        if inputs is None:
            inputs = []
        if len(inputs) < 2:
            return 0
        activity_grads = TensorDef(
            inputs[0].shape[:-1] + [inputs[1].shape[-1]], inputs[0].dtype
        )
        return reduce(mul, activity_grads.shape) * self.dsize

    @property
    @hp.param("blueprinting.layerdef")
    def flops_fw(self, inputs: List[TensorDef] = None) -> int:
        if inputs is None:
            inputs = []
        if len(inputs) < 2:
            return 0
        return batch_gemm_flops(inputs[0], inputs[1])

    @property
    @hp.param("blueprinting.layerdef")
    def flops_bw(self, inputs: List[TensorDef] = None) -> int:
        if inputs is None:
            inputs = []
        if len(inputs) < 2:
            return 0
        return 2 * batch_gemm_flops(inputs[0], inputs[1])


"""
    https://automata88.medium.com/how-to-implement-the-softmax-derivative-independently-from-any-loss-function-ae6d44363a9d
    https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html#torch.nn.Softmax
"""


@dataclass
class SoftmaxDef(LayerDef):
    dtype: DType
    dims: int = (
        0  # A dimension along which Softmax will be computed (so every slice along dim will sum to 1).
    )

    def forward(self, *inputs):
        return TensorDef(inputs[0].shape, inputs[0].dtype)

    @property
    @hp.param("blueprinting.layerdef")
    def nbytes_activity(self, inputs: List[TensorDef] = None) -> int:
        if inputs is None:
            inputs = []
        return reduce(mul, inputs[0].shape) * self.dsize

    @property
    @hp.param("blueprinting.layerdef")
    def nbytes_activity_grads(self, inputs: List[TensorDef] = None) -> int:
        if inputs is None:
            inputs = []
        return reduce(mul, inputs[0].shape) * self.dsize

    @property
    @hp.param("blueprinting.layerdef")
    def flops_fw(self, inputs: List[TensorDef] = None) -> int:
        if inputs is None:
            inputs = []
        if not inputs:
            return 0
        return 5 * reduce(mul, inputs[0].shape)

    @property
    @hp.param("blueprinting.layerdef")
    def flops_bw(self, inputs: List[TensorDef] = None) -> int:
        if inputs is None:
            inputs = []
        if not inputs:
            return 0
        return 8 * reduce(mul, inputs[0].shape)


"""
    https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html#torch.nn.SiLU
"""


@dataclass
class SiLUDef(LayerDef):
    """SiLU (Swish) activation: x * sigmoid(x)

    FLOPs计算:
    - Forward: sigmoid(x) 需要 exp + div + 1 = 3 ops，乘法 1 op，共 4 ops per element
    - Backward: d/dx[x * sigmoid(x)] = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
                需要约 6 ops per element
    """

    dtype: DType
    inplace: bool = False

    def forward(self, *inputs):
        return TensorDef(inputs[0].shape, inputs[0].dtype)

    @property
    @hp.param("blueprinting.layerdef")
    def nbytes_activity(self, inputs: List[TensorDef] = None) -> int:
        if inputs is None:
            inputs = []
        return reduce(mul, inputs[0].shape) * self.dsize

    @property
    @hp.param("blueprinting.layerdef")
    def nbytes_input(self, inputs: List[TensorDef] = None):
        if inputs is None:
            inputs = []
        return inputs[0].nbytes

    @property
    @hp.param("blueprinting.layerdef")
    def nbytes_output(self, inputs: List[TensorDef] = None):
        if inputs is None:
            inputs = []
        return self(*inputs).nbytes

    @property
    @hp.param("blueprinting.layerdef")
    def nbytes_activity_grads(self, inputs: List[TensorDef] = None) -> int:
        if inputs is None:
            inputs = []
        if not inputs:
            return 0
        return reduce(mul, inputs[0].shape) * self.dsize

    @property
    @hp.param("blueprinting.layerdef")
    def flops_fw(self, inputs: List[TensorDef] = None) -> int:
        """SiLU forward: x * sigmoid(x), ~4 ops per element"""
        if inputs is None:
            inputs = []
        if not inputs:
            return 0
        return 4 * reduce(mul, inputs[0].shape)

    @property
    @hp.param("blueprinting.layerdef")
    def flops_bw(self, inputs: List[TensorDef] = None) -> int:
        """SiLU backward: ~6 ops per element for gradient computation"""
        if inputs is None:
            inputs = []
        if not inputs:
            return 0
        return 6 * reduce(mul, inputs[0].shape)


"""
    https://pytorch.org/docs/stable/generated/torch.add.html#torch.add
    https://explained.ai/matrix-calculus/#sec:1.4.2
"""


@dataclass
class AddDef(LayerDef):
    """Element-wise addition: torch.add

    无权重层，执行两个张量的逐元素加法。
    """

    alpha: int = 1

    def forward(self, *inputs):
        return TensorDef(inputs[0].shape, inputs[0].dtype)

    @property
    @hp.param("blueprinting.layerdef")
    def nbytes_activity(self, inputs: List[TensorDef] = None) -> int:
        if inputs is None:
            inputs = []
        if len(inputs) < 2:
            return 0
        return (
            reduce(mul, inputs[0].shape) + reduce(mul, inputs[1].shape)
        ) * self.dsize

    @property
    @hp.param("blueprinting.layerdef")
    def nbytes_input(self, inputs: List[TensorDef] = None):
        if inputs is None:
            inputs = []
        if len(inputs) < 2:
            return 0
        return inputs[0].nbytes + inputs[1].nbytes

    @property
    @hp.param("blueprinting.layerdef")
    def nbytes_output(self, inputs: List[TensorDef] = None):
        if inputs is None:
            inputs = []
        if not inputs:
            return 0
        return self.forward(*inputs).nbytes

    @property
    @hp.param("blueprinting.layerdef")
    def flops_fw(self, inputs: List[TensorDef] = None) -> int:
        if inputs is None:
            inputs = []
        if not inputs:
            return 0
        return reduce(mul, inputs[0].shape)

    @property
    @hp.param("blueprinting.layerdef")
    def flops_bw(self, inputs: List[TensorDef] = None) -> int:
        """加法反向传播: dy/dx = 1, 梯度直接传递，无计算"""
        if inputs is None:
            inputs = []
        return 0


@dataclass
class MulDef(LayerDef):
    """Element-wise multiplication: torch.mul

    无权重层，执行两个张量的逐元素乘法。
    用于 SwiGLU 中的 silu(gate) * up 操作。
    """

    def forward(self, *inputs):
        return TensorDef(inputs[0].shape, inputs[0].dtype)

    @property
    @hp.param("blueprinting.layerdef")
    def nbytes_activity(self, inputs: List[TensorDef] = None) -> int:
        if inputs is None:
            inputs = []
        if len(inputs) < 2:
            return 0
        return (
            reduce(mul, inputs[0].shape) + reduce(mul, inputs[1].shape)
        ) * self.dsize

    @property
    @hp.param("blueprinting.layerdef")
    def nbytes_input(self, inputs: List[TensorDef] = None):
        if inputs is None:
            inputs = []
        if len(inputs) < 2:
            return 0
        return inputs[0].nbytes + inputs[1].nbytes

    @property
    @hp.param("blueprinting.layerdef")
    def nbytes_output(self, inputs: List[TensorDef] = None):
        if inputs is None:
            inputs = []
        if not inputs:
            return 0
        return self.forward(*inputs).nbytes

    @property
    @hp.param("blueprinting.layerdef")
    def flops_fw(self, inputs: List[TensorDef] = None) -> int:
        """逐元素乘法: N 次乘法操作"""
        if inputs is None:
            inputs = []
        if not inputs:
            return 0
        return reduce(mul, inputs[0].shape)

    @property
    @hp.param("blueprinting.layerdef")
    def flops_bw(self, inputs: List[TensorDef] = None) -> int:
        """乘法反向传播: d(x*y)/dx = y, d(x*y)/dy = x

        需要计算两个梯度，各需要 N 次乘法
        """
        if inputs is None:
            inputs = []
        if len(inputs) < 2:
            return 0
        return 2 * reduce(mul, inputs[0].shape)


@dataclass
class SequenceParallelAdd(AddDef):
    """Sequence Parallel Add: 序列并行下的加法操作

    在序列并行模式下，序列维度被分割到多个设备上。
    """

    tensor_model_parallel_size: int = 1

    def forward(self, *inputs):
        seq_len = inputs[0].shape[-2] / self.tensor_model_parallel_size
        output = TensorDef(
            inputs[0].shape[:-2] + [seq_len, inputs[0].shape[-1]], inputs[0].dtype
        )
        return output

    def _get_partitioned_inputs(self, inputs):
        """获取序列并行分区后的输入"""
        if len(inputs) < 2:
            return None, None
        seq_len = inputs[0].shape[-2] / self.tensor_model_parallel_size
        input0 = TensorDef(
            inputs[0].shape[:-2] + [seq_len, inputs[0].shape[-1]], inputs[0].dtype
        )
        input1 = TensorDef(
            inputs[1].shape[:-2] + [seq_len, inputs[1].shape[-1]], inputs[1].dtype
        )
        return input0, input1

    @property
    @hp.param("blueprinting.layerdef")
    def nbytes_activity(self, inputs: List[TensorDef] = None) -> int:
        if inputs is None:
            inputs = []
        input0, input1 = self._get_partitioned_inputs(inputs)
        if input0 is None:
            return 0
        return (reduce(mul, input0.shape) + reduce(mul, input1.shape)) * self.dsize

    @property
    @hp.param("blueprinting.layerdef")
    def nbytes_input(self, inputs: List[TensorDef] = None):
        if inputs is None:
            inputs = []
        input0, input1 = self._get_partitioned_inputs(inputs)
        if input0 is None:
            return 0
        return input0.nbytes + input1.nbytes

    @property
    @hp.param("blueprinting.layerdef")
    def flops_fw(self, inputs: List[TensorDef] = None) -> int:
        if inputs is None:
            inputs = []
        input0, _ = self._get_partitioned_inputs(inputs)
        if input0 is None:
            return 0
        return reduce(mul, input0.shape)

    @property
    @hp.param("blueprinting.layerdef")
    def flops_bw(self, inputs: List[TensorDef] = None) -> int:
        """序列并行加法反向传播: dy/dx = 1, 梯度直接传递，无计算"""
        if inputs is None:
            inputs = []
        return 0

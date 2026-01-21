from dataclasses import dataclass
from typing import List, Tuple

import hyperparameter as hp
from sympy import Expr

from blueprinting.types.base import Calculation, DType, TensorDef


@hp.param("sys")
def flops_throughput(dtype, flops, use_matrix_core=True):
    if isinstance(flops, Expr):
        subs = hp.scope.blueprinting.symbolic.subs | []
        flops = int(flops.subs(dict(subs)))
    if use_matrix_core:
        throughput = 1e12 * (hp.scope.sys.matrix.float16.tflops | 0)
        efficiency = hp.scope.sys.matrix.float16.gflops_efficiency | []
    else:
        throughput = 1e12 * (hp.scope.sys.vector.float16.tflops | 0)
        efficiency = hp.scope.sys.vector.float16.gflops_efficiency | []
    efficiency_ratio = 1.0
    for gflops, eff in efficiency:
        if flops > gflops * 1e9:
            efficiency_ratio = eff
            break
    return throughput * efficiency_ratio


@hp.param("sys")
def memrw_throughput(memrw):
    if isinstance(memrw, Expr):
        subs = hp.scope.blueprinting.symbolic.subs | {}
        memrw = int(memrw.subs(dict(subs)))
    throughput = 1e9 * (hp.scope.sys.mem1.GBps | 0)
    efficiency = hp.scope.sys.mem1.MB_efficiency | []
    efficiency_ratio = 1.0
    for mbytes, eff in efficiency:
        if memrw > mbytes * 1e6:
            efficiency_ratio = eff
            break
    return throughput * efficiency_ratio


@hp.param("sys")
def c2c_throughput():
    networks = hp.scope.sys.networks | []
    throughput = 1e9 * (networks[0]["bandwidth"])
    efficiency = networks[0]["efficiency"]
    return throughput * efficiency


@hp.param("sys")
def c2c_nbytes(c2c, comm_type, num_peers):
    if isinstance(c2c, Expr):
        subs = hp.scope.blueprinting.symbolic.subs | {}
        c2c = int(c2c.subs(dict(subs)))
    networks = hp.scope.sys.networks | []
    ops = networks[0]["ops"]
    op_size = 0
    for op, [scalar, offset] in ops.items():
        if comm_type == op:
            op_size = scalar * c2c + offset * (1 / num_peers * c2c)
            break
    return op_size


def c2c_times(comm_type, comm_size, throughput):
    networks = hp.scope.sys.networks | []
    ops = networks[0]["ops"]
    latency = networks[0]["latency"]
    times = comm_size / throughput
    return times + latency if comm_type in ops.keys() else 0.0


@dataclass
class LayerDef:
    dtype: DType

    def forward(self, *inputs) -> TensorDef:
        return TensorDef()

    def __call__(self, *inputs: TensorDef):
        return Calculation(inputs, self.forward(*inputs), self)

    @property
    def nelems(self) -> int:
        return 0

    @property
    @hp.param("blueprinting.layerdef")
    def nbytes_input(self, inputs: List[TensorDef] = []):
        return inputs[0].nbytes

    @property
    @hp.param("blueprinting.layerdef")
    def nbytes_output(self, inputs: List[TensorDef] = []):
        return self(*inputs).nbytes

    @property
    @hp.param("blueprinting.layerdef")
    def nbytes_weight(self) -> int:
        if not hasattr(self, "weight"):
            return 0
        return self.weight.nbytes

    @property
    @hp.param("blueprinting.layerdef")
    def nbytes_weight_grads(self, inputs: List[TensorDef] = []) -> int:
        if not hasattr(self, "weight"):
            return 0
        weight_grads = self.weight.nbytes
        return weight_grads

    @property
    @hp.param("blueprinting.layerdef")
    def nbytes_activity(self, inputs: List[TensorDef] = []):
        return inputs[0].nbytes

    @property
    @hp.param("blueprinting.layerdef")
    def nbytes_activity_grads(self, inputs: List[TensorDef] = []):
        return self(*inputs).nbytes

    @property
    @hp.param("blueprinting.layerdef")
    def flops_fw(self, inputs: List[TensorDef] = []) -> int:
        return 0

    @property
    @hp.param("blueprinting.layerdef")
    def flops_bw(self, inputs: List[TensorDef] = []) -> int:
        return 0

    @property
    @hp.param("blueprinting.layerdef")
    def c2c_fw(self, inputs: List[TensorDef] = []) -> int:
        return 0

    @property
    @hp.param("blueprinting.layerdef")
    def c2c_bw(self, inputs: List[TensorDef] = []) -> int:
        return 0

    @property
    @hp.param("blueprinting.layerdef")
    def time_flops_fw(self, inputs: List[TensorDef] = []) -> int:
        flops = self.flops_fw
        throughput = flops_throughput(self.dtype, flops)
        return flops / throughput

    @property
    @hp.param("blueprinting.layerdef")
    def time_flops_bw(self, inputs: List[TensorDef] = []) -> int:
        flops = self.flops_bw
        throughput = flops_throughput(self.dtype, flops)
        return flops / throughput

    @property
    @hp.param("blueprinting.layerdef")
    def time_memrw_fw(self, inputs: List[TensorDef] = []) -> int:
        memrw = self.memory_fw
        throughput = memrw_throughput(memrw)
        return memrw / throughput

    @property
    @hp.param("blueprinting.layerdef")
    def time_memrw_bw(self, inputs: List[TensorDef] = []) -> int:
        memrw = self.memory_bw
        throughput = memrw_throughput(memrw)
        return memrw / throughput

    @property
    @hp.param("blueprinting.layerdef")
    def time_fw(self, inputs: List[TensorDef] = []) -> int:
        time_c2c_fw = self.time_c2c_fw
        time_flops = self.time_flops_fw
        time_memrw = self.time_memrw_fw
        if hp.scope.sys.processing_mode | "roofline" == "roofline":
            return max(time_flops, time_memrw)
        # hp.scope.sys.processing_mode == "no_overlap"
        return time_flops + time_memrw + time_c2c_fw

    @property
    @hp.param("blueprinting.layerdef")
    def time_bw(self, inputs: List[TensorDef] = []) -> int:
        time_flops = self.time_flops_bw
        time_memrw = self.time_memrw_bw
        if hp.scope.sys.processing_mode | "roofline" == "roofline":
            return max(time_flops, time_memrw)
        # hp.scope.sys.processing_mode == "no_overlap"
        return time_flops + time_memrw

    @property
    @hp.param("blueprinting.layerdef")
    def memory_fw(self, inputs: List[TensorDef] = []) -> int:
        input_memory = self.nbytes_input
        output_memory = self.nbytes_output

        if hasattr(self, "weight"):
            weight_memory = self.nbytes_weight
            return input_memory + weight_memory + output_memory
        else:
            return input_memory + output_memory

    @property
    @hp.param("blueprinting.layerdef")
    def wgrad_memory(self, inputs: List[TensorDef] = []) -> int:
        if not hasattr(self, "weight"):
            return 0
        wgard_mem = self.nbytes_weight_grads + self.nbytes_activity + self.nbytes_activity_grads
        return wgard_mem

    @property
    @hp.param("blueprinting.layerdef")
    def agrad_memory(self, inputs: List[TensorDef] = []) -> int:
        if hasattr(self, "weight"):
            agard_mem = self.nbytes_weight + self.nbytes_activity + self.nbytes_activity_grads
        else:
            agard_mem = self.memory_fw

        return agard_mem

    @property
    @hp.param("blueprinting.layerdef")
    def memory_bw(self, inputs: List[TensorDef] = []) -> int:
        agrad_memory = self.agrad_memory
        wgrad_memory = self.wgrad_memory

        return agrad_memory + wgrad_memory

    @property
    @hp.param("blueprinting.layerdef")
    def time_c2c_fw(self, inputs: List[TensorDef] = []) -> int:
        return 0

    @property
    @hp.param("blueprinting.layerdef")
    def time_c2c_bw(self, inputs: List[TensorDef] = []) -> int:
        return 0

    @property
    @hp.param("blueprinting.layerdef")
    def placement_weight(self, inputs: List[TensorDef] = []) -> Tuple[TensorDef, ...]:
        return [TensorDef([0, 0], dtype=self.dtype)]

    @property
    def dsize(self) -> int:
        return self.dtype.size

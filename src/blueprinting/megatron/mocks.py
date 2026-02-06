from contextlib import nullcontext

import torch
import torch.distributed
import torch.utils
import torch.utils.data
from torch.fx.experimental.proxy_tensor import maybe_disable_fake_tensor_mode

patches = {}
states = {}


def mock(name, state=None):
    if state is not None:
        states[name] = state

    def decorator(func):
        path = name.split(".")

        if len(path) < 2:
            print(f"==== error install monkey patch for {name}")
        curr = globals()[path[0]]
        for p in path[1:-1]:
            curr = getattr(curr, p)
        setattr(curr, path[-1], func)
        return func

    return decorator


@mock("torch.cuda.is_available")
def torch_cuda_is_available():
    return True


@mock("torch.cuda.device_count")
def torch_cuda_device_count():
    return 8


@mock("torch.cuda.set_device")
def torch_cuda_set_device(arg):
    states["torch.cuda.set_device"] = arg


@mock("torch.cuda.current_device")
def torch_cuda_current_device():
    return states["torch.cuda.set_device"]


@mock("torch.cuda.get_rng_state")
def torch_cuda_get_rng_state():
    return None


@mock("torch.cuda.synchronize")
def torch_cuda_synchronize():
    return None


@mock("torch.cuda.is_current_stream_capturing")
def torch_cuda_is_current_stream_capturing():
    return False


@mock("torch.Tensor.item")
def item(*args, **kwargs):
    return 0


@mock("torch.compile")
def mock_torch_compile(func):
    return func


@mock("torch._amp_foreach_non_finite_check_and_unscale_")
def _amp_foreach_non_finite_check_and_unscale_(*args, **kwargs):
    return


torch.cuda.HalfTensor = torch.HalfTensor


class MockDistributedWork:
    def wait(self):
        pass


class MockDistributedBackend:
    default = None

    def __init__(self, backend, world_size, rank, timeout=None):
        self.backend = backend
        self.world_size = world_size
        self.rank = rank
        self.timeout = timeout

    @mock("torch.distributed.init_process_group")
    def init_process_group(backend=None, world_size=None, rank=None, timeout=None):
        get_args().do_train = True
        get_args().num_workers = 0
        MockDistributedBackend.default = MockDistributedBackend(
            backend, world_size, rank, timeout
        )

    @mock("torch.distributed.is_initialized")
    def is_initialized():
        return MockDistributedBackend.default is not None

    @mock("torch.distributed.get_rank")
    def get_rank(group=None):
        if group is not None:
            rank = MockDistributedBackend.get_rank()
            return group.ranks.index(rank)
        return MockDistributedBackend.default.rank

    @mock("torch.distributed.get_world_size")
    def get_world_size(group=None):
        if group is not None:
            return len(group.ranks)
        return MockDistributedBackend.default.world_size

    @mock("torch.distributed.get_backend")
    def get_backend(group=None):
        return MockDistributedBackend.default.backend

    @mock("torch.distributed.distributed_c10d._get_default_group")
    def _get_default_group():
        return MockDistributedBackend.default

    # @mock("torch.distributed.all_reduce")
    def allreduce(*args, **kwargs):
        # print(f"all_redice({args}, {kwargs})")
        return MockDistributedWork()

    @mock("torch.distributed.barrier")
    def barrier(group=None):
        if group is not None:
            return print(f"barrier(group={group})")
        print("barrier()")
        return MockDistributedWork()

    @mock("torch.distributed._all_gather_base")
    def _all_gather_base(*args, **kwargs):
        return MockDistributedWork()

    @mock("torch.distributed.broadcast")
    def broadcast(*args, **kwargs):
        return MockDistributedWork()


class MockDistributedGroup:
    def __init__(self, ranks, timeout=None, pg_options=None, backend=None):
        self.ranks = ranks
        self.timeout = timeout
        self.backend = backend
        self.pg_options = pg_options

    @mock("torch.distributed.new_group")
    def new_group(ranks, timeout=None, pg_options=None, backend=None):
        return MockDistributedGroup(ranks, timeout, pg_options, backend=None)

    # @mock("torch.distributed.all_reduce")
    def allreduce(*args, **kwargs):
        # print(f"all_redice({args}, {kwargs})")
        return MockDistributedWork()

    def allreduce_coalesced(*args, **kwargs):
        return MockDistributedWork()


import megatron
import megatron.core
import megatron.core.jit
import megatron.core.models
import megatron.core.models.gpt
import megatron.core.models.gpt.gpt_layer_specs
import megatron.core.optimizer
import megatron.core.tensor_parallel
import megatron.training

# import megatron.training.tokenizer
from megatron.training import get_args

# from megatron.training import print_rank_0

# @mock("megatron.training.global_vars._build_tokenizer")
# def mock__build_tokenizer(*args, **kwargs):

#     get_args().padded_vocab_size=32000
#     return 12


@mock("megatron.training.get_tokenizer")
def get_tokenizer(*args, **kwargs):
    class Tokenizer:
        def __init__(self):
            self.eod = 42
            self.unique_identifiers = [1, 2, 3, 4]

    return Tokenizer()


from megatron.core.datasets.utils import compile_helpers


@mock("megatron.training.initialize._compile_dependencies")
def _compile_dependencies():
    compile_helpers()


@mock("megatron.core.jit.jit_fuser ")
def jit_fuser(x):
    return x


@mock("megatron.training.training.set_jit_fusion_options")
def set_jit_fusion_options(*args, **kwargs):
    megatron.core.jit.jit_fuser = jit_fuser


megatron.core.tensor_parallel.random.CudaRNGStatesTracker.fork = nullcontext


@mock("megatron.core.optimizer.optimizer.Float16OptimizerWithFloat16Params.__init__")
def Float16OptimizerWithFloat16Params_init(
    self, optimizer, config, grad_scaler, init_state_fn
):
    megatron.core.optimizer.optimizer.MixedPrecisionOptimizer.__init__(
        self, optimizer, config, grad_scaler, init_state_fn
    )
    self.float16_groups = []
    self.fp32_from_float16_groups = []
    self.fp32_from_fp32_groups = []


@mock("megatron.core.optimizer.optimizer._multi_tensor_copy_this_to_that")
def _multi_tensor_copy_this_to_that(*args, **kwargs):
    pass


@mock("megatron.training.global_vars._build_tokenizer")
def mock__build_tokenizer(*args, **kwargs):
    get_args().padded_vocab_size = 32000


@mock("megatron.core.timers.Timers.get_all_timers_string")
def mock_get_all_timers_string(*args, **kwargs):
    return "[time]"


@mock("megatron.core.fusions.fused_softmax.FusedScaleMaskSoftmax.is_kernel_available")
def FusedScaleMaskSoftmax_is_kernel_available(*args, **kwargs):
    return False


import torch.utils.data._utils

ori_worker_loop = torch.utils.data._utils.worker._worker_loop


@mock("torch.utils.data._utils.worker._worker_loop")
def mock_worker_loop(*args, **kwargs):
    with maybe_disable_fake_tensor_mode():
        ori_worker_loop(*args, **kwargs)


@mock("torch.utils.data._utils.pin_memory.pin_memory")
def mock_pin_memory(data, *args):
    return data


@mock("torch.autograd.profiler.record_function")
def mock_torch_autograd_profiler_record_function(*args, **kwargs):
    return nullcontext()


# @mock("megatron.training.training.build_train_valid_test_datasets")
# def build_train_valid_test_datasets(build_train_valid_test_datasets_provider):


# @mock("torch._subclasses.fake_tensor.FakeTensorMode._dispatch_impl")
# def mock_dispatch_impl(self, func, types, args, kwargs):
#     return func(*args, **kwargs)

from contextlib import contextmanager

import torch
from torch.utils._python_dispatch import TorchDispatchMode

optbl = {}
opcnt = {}


def register(name):
    def decorator(func):
        optbl[name] = func
        return func

    return decorator


def _device_handler(args):
    assert len(args) == 1 and isinstance(args[0], TraceTensor)
    return args[0].fake_device


_DISPATCH_META_HANDLERS = {
    torch.ops.prim.device.default: _device_handler,
    torch.ops.aten.size.default: lambda args: tuple(int(s) for s in args[0].size()),
    torch.ops.aten.stride.default: lambda args: tuple(int(s) for s in args[0].stride()),
    torch.ops.aten.storage_offset.default: lambda args: int(args[0].storage_offset()),
}


class TraceTensor(torch.Tensor):
    """Tensor subclass for tracing pytorch computations.

    Clean tracing without decomposing and torch.compile.

    Examples
    --------
    >>> with TraceTensorMode():
    ...     a = torch.ones([10, 1]).cuda()
    ...     b = torch.ones([1, 10]).cuda()
    ...     a * b
    cuda[10, 10]

    >>> with TraceTensorMode():
    ...     torch.matmul(b, a)
    cuda[1, 1]

    >>> with TraceTensorMode():
    ...     print((a * b).expr())
    cuda[10, 10] = aten.mul.Tensor(cuda[10, 1], cuda[1, 10])

    >>> with TraceTensorMode(), torch.autograd.set_multithreading_enabled(False):
    ...     a = torch.ones([10, 1], device="cuda", requires_grad=True)
    ...     b = torch.ones([1, 10], device="cuda", requires_grad=True)
    ...     (a * b)
    cuda[10, 10]

    >>> (a * b).grad_fn  # doctest: +ELLIPSIS
    <MulBackward0 object at ...>
    """

    @staticmethod
    def __new__(cls, elem, device, func=None, args=(), kwargs=None):
        self = torch.Tensor._make_subclass(
            cls,
            elem,
            elem.requires_grad,
            dispatch_device=False,
            device_for_backend_keys=device,
        )
        assert (
            elem.device.type == "meta"
        ), f"create trace tensor from {elem.device.type}, `meta` is expected"
        self.fake_device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )
        self.func = func
        self.args = args
        self.kwargs = kwargs
        return self

    def __init__(self, *args, **kwargs):
        super().__init__()

    def __getitem__(self, index):
        if isinstance(index, slice):
            start = index.start
            stop = index.stop
            if start is None:
                start = 0
            if stop is None:
                stop = self.shape[0]
            if stop == -1:
                stop = self.shape[0] - 1
            return torch.empty(
                [stop - start], dtype=self.dtype, requires_grad=self.requires_grad
            )
        if isinstance(index, (tuple, list)):
            return torch.empty(
                index[0].shape[0], dtype=self.dtype, requires_grad=self.requires_grad
            )
        return torch.empty([1], dtype=self.dtype, requires_grad=self.requires_grad)

    @property
    def device(self) -> torch.device:
        """
        Examples
        --------
        >>> with TraceTensorMode():
        ...     a = torch.ones([10], device="cuda:1")
        ...     print(a.device)
        cuda:1

        >>> with TraceTensorMode() as m:
        ...     with m.in_op_manager():
        ...         print(a.device)
        meta
        """
        if (
            TraceTensorMode.current is not None
            and TraceTensorMode.current.in_op
        ):
            return torch.device("meta")
        return self.fake_device

    def cuda(self, arg=None, **kwargs):
        if arg is not None:
            self.fake_device = torch.device("cuda", arg)
        else:
            self.fake_device = torch.device("cuda")
        return self

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        if func == torch.ops.prim.device.default:
            assert len(args) == 1 and isinstance(args[0], TraceTensor)
            if TraceTensorMode.current is not None and TraceTensorMode.current.in_op:
                return torch.device("meta")
            return args[0].fake_device
        if handler := _DISPATCH_META_HANDLERS.get(func):
            return handler(args)

        name = func.name()
        if name in optbl:
            return optbl[name](*args, **kwargs)
        if TraceTensorMode.current is not None:
            return TraceTensorMode.current.dispatch(func, types, args, kwargs)
        raise "bad trace"

    def __repr__(self):
        shape = self.shape
        if isinstance(shape, torch.Size):
            shape = list(shape)
        return f"{self.device}{shape}"

    def expr(self):
        args = ", ".join([str(arg) for arg in self.args])
        kwargs = (
            ", ".join([f"{k}={v}" for k, v in self.kwargs.items()])
            if self.kwargs is not None
            else None
        )
        if kwargs is None or kwargs == "":
            return f"{self} = {self.func}({args})"
        return f"{self} = {self.func}({args}, {kwargs})"

    def tolist(self):
        out = []
        for s in range(self.shape[0]):
            torch._check_is_size(s)
            torch._check(s >= 2)
            out.append(s)
        return out


class TraceTensorMode(TorchDispatchMode):
    current = None

    def __init__(self):
        self._mode_key = torch._C._TorchDispatchModeKey.FAKE
        self._stack = []
        self.in_op = False
        super().__init__()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        return self.dispatch(func, types, args, kwargs)

    def __enter__(self):
        torch._C._set_only_lift_cpu_tensors(True)
        previous = torch._C._unset_dispatch_mode(self._mode_key)
        if self is not previous:
            self._stack.append((True, previous))
            TraceTensorMode.current = self
            return super().__enter__()
        torch._C._set_dispatch_mode(self)
        self._stack.append((False, None))
        TraceTensorMode.current = self
        return self

    def __exit__(self, a, b, c):
        live, previous = self._stack.pop()
        if live:
            _ = super().__exit__(a, b, c)
            if previous is not None:
                torch._C._set_dispatch_mode(previous)
                TraceTensorMode.current = previous

    @contextmanager
    def in_op_manager(self):
        prev_in_op = self.in_op
        self.in_op = True
        with torch._C._DisableTorchDispatch(), torch._C._PreserveDispatchKeyGuard():
            torch._C._set_meta_in_tls_dispatch_include(True)
            try:
                yield
            finally:
                self.in_op = prev_in_op

    def dispatch(self, func, types, args=(), kwargs=None):
        if handler := _DISPATCH_META_HANDLERS.get(func):
            return handler(args)
        name = func.name()
        if name in optbl:
            opcnt[name] = opcnt.get(name, 0) + 1
            try:
                return optbl[name](*args, **kwargs)
            except:
                raise ValueError(
                    f"bad trace for {name}: {func}[{types}]({args}, {kwargs})"
                )

        # print(f"== unknown op: {name}:", func, types, args, kwargs)
        # flat_args, args_spec = pytree.tree_flatten((args, kwargs))
        # flat_args = [
        #    torch.empty_like(x, device="meta") if isinstance(x, TraceTensor) else x
        #    for x in flat_args
        # ]
        # new_args, new_kwargs = pytree.tree_unflatten(flat_args, args_spec)
        with self.in_op_manager():
            value = func(*args, **kwargs)

        def to_trace_tensor(x):
            if isinstance(x, TraceTensor):
                return x
            if isinstance(x, (list, tuple)):
                return [to_trace_tensor(y) for y in x]

            try:
                device = args[0].device if isinstance(args, list) else args[0][0].device
                ret = TraceTensor(x, device, func=func, args=args, kwargs=kwargs)
                return ret
            except:
                print(f"bad trace for {name}: {func}[{types}]({args}, {kwargs})")
                raise

        return to_trace_tensor(value)


@register("aten::empty.memory_format")
def empty(shape, **kwargs):
    """Mock aten::empty.memory_format

    Examples
    --------
    >>> cnt = opcnt.get("aten::empty.memory_format", 0)
    >>> with TraceTensorMode():
    ...     a = torch.empty([1, 10]).cuda()
    >>> print("aten::empty.memory_format calls:", opcnt.get("aten::empty.memory_format", 0) - cnt)
    aten::empty.memory_format calls: 1

    >>> print(a)
    cuda[1, 10]
    """
    device = kwargs.get("device", "cuda")
    kwargs["device"] = "meta"
    return TraceTensor(
        torch.empty(shape, **kwargs).to("meta"),
        device,
        func=torch.ops.aten.empty,
        kwargs=kwargs,
    )


@register("aten::empty_like")
def empty_like(other, **kwargs):
    """Mock aten::empty_like

    Examples
    --------
    >>> cnt = opcnt.get("aten::empty_like", 0)
    >>> with TraceTensorMode():
    ...     a = torch.ones([1, 10]).cuda()
    ...     a = torch.empty_like(a).cuda()
    >>> print("aten::empty_like calls:", opcnt.get("aten::empty_like", 0) - cnt)
    aten::empty_like calls: 1

    >>> print(a)
    cuda[1, 10]
    """
    device = kwargs.get("device", "cuda")
    kwargs["device"] = "meta"
    return TraceTensor(
        torch.empty(other.shape, **kwargs),
        device,
        func=torch.ops.aten.empty_like,
        kwargs=kwargs,
    )


@register("aten::arange")
def arange(end, **kwargs):
    """Mock aten::arange

    Examples
    --------
    >>> cnt = opcnt.get("aten::arange", 0)
    >>> with TraceTensorMode():
    ...     a = torch.arange([1, 10]).cuda()
    >>> print("aten::arange calls:", opcnt.get("aten::arange", 0) - cnt)
    aten::arange calls: 1

    >>> print(a)
    cuda[1, 10]
    """
    device = kwargs.get("device", "cuda")
    kwargs["device"] = "meta"
    return TraceTensor(
        torch.empty([end], **kwargs), device, func=torch.ops.aten.arange, kwargs=kwargs
    )


@register("aten::arange.start")
def arange(start=0, end=0, step=1, **kwargs):
    """Mock aten::arange

    Examples
    --------
    >>> cnt = opcnt.get("aten::arange", 0)
    >>> with TraceTensorMode():
    ...     a = torch.arange([1, 10]).cuda()
    >>> print("aten::arange calls:", opcnt.get("aten::arange", 0) - cnt)
    aten::arange calls: 1

    >>> print(a)
    cuda[1, 10]
    """
    device = kwargs.get("device", "cuda")
    kwargs["device"] = "meta"
    length = len(range(start, end, step))
    return TraceTensor(
        torch.empty([length], **kwargs),
        device,
        func=torch.ops.aten.arange,
        kwargs=kwargs,
    )


@register("aten::scalar_tensor")
def scalar_tensor(value, **kwargs):
    """Mock aten::scalar_tensor

    Examples
    --------
    >>> cnt = opcnt.get("aten::scalar_tensor", 0)
    >>> with TraceTensorMode():
    ...     a = torch.scalar_tensor(0).cuda()
    >>> print("aten::scalar_tensor calls:", opcnt.get("aten::scalar_tensor", 0) - cnt)
    aten::scalar_tensor calls: 1

    >>> print(a)
    cuda[1, 10]
    """
    device = kwargs.get("device", "cuda")
    kwargs["device"] = "meta"
    return TraceTensor(
        torch.empty([1], **kwargs),
        device,
        func=torch.ops.aten.scalar_tensor,
        kwargs=kwargs,
    )


@register("aten::ones")
def ones(shape, **kwargs):
    """Mock aten::ones

    Examples
    --------
    >>> cnt = opcnt.get("aten::ones", 0)
    >>> with TraceTensorMode():
    ...     a = torch.ones([1, 10]).cuda()
    >>> print("aten::ones calls:", opcnt.get("aten::ones", 0) - cnt)
    aten::ones calls: 1

    >>> print(a)
    cuda[1, 10]
    """
    device = kwargs.get("device", "cuda")
    kwargs["device"] = "meta"
    return TraceTensor(
        torch.ones(shape, **kwargs), device, func=torch.ops.aten.ones, kwargs=kwargs
    )


@register("aten::zeros")
@register("aten::zeros_")
def zeros(shape, **kwargs):
    """Mock aten::zeros

    Examples
    --------
    >>> cnt = opcnt.get("aten::zeros", 0)
    >>> with TraceTensorMode():
    ...     a = torch.zeros([1, 10])
    >>> print("aten::zeros calls:", opcnt.get("aten::zeros", 0) - cnt)
    aten::zeros calls: 1
    """
    device = kwargs.get("device", "cuda")
    kwargs["device"] = "meta"
    return TraceTensor(
        torch.zeros(shape, **kwargs), device, func=torch.ops.aten.zeros, kwargs=kwargs
    )


@register("aten::zero_")
def zero_(self):
    return self


@register("aten::normal_")
def normal_(self, *args, **kwargs):
    return self


@register("aten::fill_")
@register("aten::fill_.Scalar")
def fill_(self, *args, **kwargs):
    return self


@register("aten::_to_copy")
def _to_copy(self, *args, **kwargs):
    return self


@register("aten::lift_fresh")
def lift_fresh(arg, **kwargs):
    device = kwargs.get("device", "cuda")
    kwargs["device"] = "meta"
    return TraceTensor(
        torch.empty(arg.shape, **kwargs).to("meta"),
        device,
        func=torch.ops.aten.lift_fresh,
        kwargs=kwargs,
    )


@register("aten::detach")
def detach(self):
    return TraceTensor(
        torch.empty(self.shape, requires_grad=self.requires_grad).to("meta"),
        self.device,
        func=torch.ops.aten.detach,
    )


@register("aten::embedding")
def embedding(weight, input):
    value = torch.ops.aten.embedding.default(
        torch.empty(
            weight.shape,
            dtype=weight.dtype,
            device="meta",
            requires_grad=weight.requires_grad,
        ),
        torch.empty(
            input.shape,
            dtype=torch.int64,
            device="meta",
            requires_grad=input.requires_grad,
        ),
    )
    return TraceTensor(
        value, input.device, func=torch.ops.aten.embedding.default, args=(weight, input)
    )

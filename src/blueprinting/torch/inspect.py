import gc
import os
import time
import traceback
from typing import List

import torch
import torch.distributed


__all__ = ["install_tensorboard_hook"]

WRITER = None
WRITER_INTERVAL = 1
GLOBAL_STEP = 0

MEGATRON_HOOK = None


def install_megatron_hook():
    global MEGATRON_HOOK
    if MEGATRON_HOOK is not None:
        return

    MEGATRON_HOOK = True
    try:
        import megatron
    except:
        return

    try:
        import megatron
        import megatron.global_vars

        megatron.global_vars._GLOBAL_TENSORBOARD_WRITER = get_writer()
        args = megatron.global_vars.get_args()
        args.tensorboard_log_interval = 10
    except:
        print("failed to install megatron hook")
        MEGATRON_HOOK = False


def get_writer(name=None, interval=10):
    global WRITER
    global WRITER_INTERVAL
    if WRITER is None:
        install_megatron_hook()
        rank = torch.distributed.get_rank()
        if name is None:
            name = os.environ.get("EXP_NAME", None)
        if name is None:
            name = time.strftime("%Y%m%d%H%M")
        name = f"logs/{name}/{rank}/"

        from torch.utils.tensorboard import SummaryWriter

        WRITER = SummaryWriter(name)
        WRITER_INTERVAL = interval
    return WRITER


def inspect_weights(m: torch.nn.Module):
    global WRITER
    global WRITER_INTERVAL
    global GLOBAL_STEP

    if GLOBAL_STEP % WRITER_INTERVAL != 0:
        return

    if GLOBAL_STEP == 0:
        return

    if isinstance(m, list):
        for mm in m:
            inspect_weights(mm)
        return

    def find_longest_prefix(data):
        if not data:
            return ""
        shortest_word = min(data, key=len)

        for prefix_slice_end in range(len(shortest_word), 0, -1):
            if all(i.startswith(shortest_word[0:prefix_slice_end]) for i in data):
                return shortest_word[0:prefix_slice_end]

        return ""

    def xxh64(x):
        import xxhash

        return xxhash.xxh64(x).intdigest()

    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    cnt = 0
    params = {id(x.data): x for x in m.parameters()}

    prefix_remove = find_longest_prefix([x for x, _ in m.named_parameters()])

    for name, weight in m.named_parameters():
        cnt += 1
        if weight is None:
            continue
        if name is None:
            name = f"unamed_{cnt}"

        if world_size > 1 and abs(xxh64(name)) % world_size != rank:
            print(f"skipping {name} @{GLOBAL_STEP} {abs(xxh64(name))} {rank}/{world_size}")
            continue
        print(f"logging {name} @{GLOBAL_STEP} ")
        wrt = get_writer()
        # write_hist(wrt, name, weight.data, global_step=global_step)
        name = name[len(prefix_remove) :]
        try:
            print(f"\t{name} @{GLOBAL_STEP} to tensorboard")
            w = weight.data
            get_writer().add_histogram(name, w, global_step=GLOBAL_STEP)
            get_writer().add_scalar(f"{name}/norm", (w * w).sum(), global_step=GLOBAL_STEP)
        except:
            traceback.print_exc()

        did = id(weight.data)

        grad = None
        if did in params and hasattr(params[did], "grad") and params[did].grad is not None:
            grad = params[did].grad
        elif hasattr(weight, "grad") and weight.grad is not None:
            grad = weight.grad
        try:
            if grad is not None:
                print(f"\t{name}/grad @{GLOBAL_STEP} to tensorboard")
                # write_hist(wrt, name, weight.grad, global_step=global_step)
                get_writer().add_histogram(f"{name}/grad", grad, global_step=GLOBAL_STEP)
                get_writer().add_scalar(f"{name}/grad/norm", (grad * grad).sum(), global_step=GLOBAL_STEP)
        except:
            traceback.print_exc()

        main_grad = None
        if did in params and hasattr(params[did], "main_grad") and params[did].main_grad is not None:
            main_grad = params[did].main_grad
        elif hasattr(weight, "main_grad") and weight.main_grad is not None:
            main_grad = weight.main_grad
        try:
            if main_grad is not None:
                print(f"\t{name}/main_grad @{GLOBAL_STEP} to tensorboard")
                # write_hist(wrt, name, weight.main_grad, global_step=global_step)
                get_writer().add_histogram(f"{name}/main_grad", main_grad, global_step=GLOBAL_STEP)
                get_writer().add_scalar(
                    f"{name}/main_grad/norm",
                    (main_grad * main_grad).sum(),
                    global_step=GLOBAL_STEP,
                )
        except:
            traceback.print_exc()


def step_writer():
    global GLOBAL_STEP
    GLOBAL_STEP = GLOBAL_STEP + 1


def write_hist(writer, name: str, values: torch.Tensor, global_step: int = 0):
    counts, limits = torch.histogram(values, 500)
    counts, limits = counts.cpu().detach(), limits.cpu().detach()
    writer.add_histogram_raw(
        name,
        min=values.min(),
        max=values.max(),
        num=values.nelement(),
        sum=values.sum(),
        sum_squares=values.square().sum(),
        bucket_limits=limits[1:].tolist(),  # <- note here.
        bucket_counts=counts.tolist(),
        global_step=global_step,
    )


toplevel_model = None


def get_top_level_modules() -> List:
    objs = gc.get_objects()
    objs = [obj for obj in objs if isinstance(obj, torch.nn.Module)]
    children = set()

    def walk(obj):
        if hasattr(obj, "children"):
            cnt = 0
            for child in obj.children():
                children.add(id(child))
                walk(child)
                cnt += 1
            if cnt == 0:
                children.add(id(obj))
        else:
            children.add(id(obj))

    for obj in objs:
        walk(obj)
    return [obj for obj in objs if id(obj) not in children]


def optimizer_step_pre_hook(optim, *args, **kwargs):
    global toplevel_model
    if toplevel_model is None:
        toplevel_model = get_top_level_modules()
    if toplevel_model is not None:
        inspect_weights(toplevel_model)


def optimizer_step_post_hook(optim, *args, **kwargs):
    step_writer()


def install_tensorboard_hook():
    from torch.optim.optimizer import (
        register_optimizer_step_pre_hook,
        register_optimizer_step_post_hook,
    )

    register_optimizer_step_pre_hook(optimizer_step_pre_hook)
    register_optimizer_step_post_hook(optimizer_step_post_hook)

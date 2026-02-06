import gc
import os
import time
import traceback
from typing import List

import torch
import torch.distributed
from torch.utils.tensorboard import SummaryWriter

__all__ = ["install_tensorboard_hook"]

writer = None
writer_interval = 1
global_step = 0


def get_writer(name=None, interval=10):
    global writer
    global writer_interval
    if writer is None:
        rank = torch.distributed.get_rank()
        if name is None:
            name = os.environ.get("EXP_NAME", None)
        if name is None:
            name = time.strftime("%Y%m%d%H%M")
        name = f"logs/{name}/{rank}/"
        writer = SummaryWriter(name)
        writer_interval = interval
    return writer


def inspect_weights(m: torch.nn.Module):
    global writer
    global writer_interval
    global global_step

    if global_step % writer_interval != 0:
        return

    if isinstance(m, list):
        for mm in m:
            inspect_weights(mm)
        return

    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    cnt = 0
    params = {id(x.data): x for x in m.parameters()}
    for name, weight in m.named_parameters():
        cnt += 1
        if weight is None:
            continue
        if name is None:
            name = f"unamed_{cnt}"
        if world_size > 1 and cnt % world_size != rank:
            continue
        get_writer()
        # write_hist(wrt, name, weight.data, global_step=global_step)
        try:
            print(f"logging {name} @{global_step} to tensorboard")
            get_writer().add_histogram(name, weight.data, global_step=global_step)
        except:
            traceback.print_exc()

        did = id(weight.data)
        try:
            if (
                did in params
                and hasattr(params[did], "grad")
                and params[did].grad is not None
            ):
                print(f"logging {name}/grad @{global_step} to tensorboard")
                # write_hist(wrt, name, weight.grad, global_step=global_step)
                get_writer().add_histogram(
                    f"{name}/grad", params[did].grad, global_step=global_step
                )
        except:
            traceback.print_exc()

        try:
            if (
                did in params
                and hasattr(params[did], "main_grad")
                and params[did].main_grad is not None
            ):
                print(f"logging {name}/main_grad @{global_step} to tensorboard")
                # write_hist(wrt, name, weight.main_grad, global_step=global_step)
                get_writer().add_histogram(
                    f"{name}/main_grad", params[did].main_grad, global_step=global_step
                )
        except:
            traceback.print_exc()


def step_writer():
    global global_step
    global_step = global_step + 1


def write_hist(
    writer: SummaryWriter, name: str, values: torch.Tensor, global_step: int = 0
):
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
    from torch.optim.optimizer import register_optimizer_step_pre_hook

    register_optimizer_step_pre_hook(optimizer_step_pre_hook)

"""Blueprinting: Heterogeneous Computing and Large-Scale Distributed Computing Simulators"""

import logging
import sys
from typing import List, Optional

import fire
import fire.decorators
import hyperparameter as hp

from . import io
from .types import (
    CommCounter,
    DType,
    Execution,
    Memory,
    Model,
    ModelComm,
    ModelFlops,
    ModelParams,
    Network,
    Processor,
    System,
    TensorDef,
)

__all__ = [
    "io",
    "Execution",
    "Model",
    "ModelParams",
    "ModelFlops",
    "ModelComm",
    "System",
    "Memory",
    "Processor",
    "Network",
    "CommCounter",
    "DType",
    "TensorDef",
]


class LLM:
    """support for large language models"""

    @fire.decorators.SetParseFns(define=lambda x: x)
    def train(
        model,
        execution,
        system,
        stats=None,
        peers=False,
        layers=False,
        define: Optional[List] = None,
    ):
        """analysis llm training"""
        if define is None:
            define = []
        print(f"blueprinting train {model}", define, type(define), len(define))
        app_json = io.read_json_file(model)
        exe_json = io.read_json_file(execution)
        sys_json = io.read_json_file(system)

        logger = logging.getLogger()
        logger.addHandler(logging.StreamHandler(stream=sys.stdout))
        logger.setLevel("INFO")
        with hp.scope(app=app_json, sys=sys_json, exe=exe_json) as ps, hp.scope(*define) as ps:
            app = Model(ps.app)
            Execution(ps.exe)
            syst = System(ps.sys)

            # TODO: Implement blueprinting's own Llm simulator
            # For now, this is a placeholder
            print(f"Model: {app.hidden}x{app.num_blocks} blocks")
            print(f"System: {syst.proc_mode} mode")

        if stats is not None and io.is_json_extension(stats):
            # TODO: Implement stats collection
            pass

    def infer(self):
        """analysis llm inference"""
        print("blueprinting infer")

    def megatron(self, *args, **kwargs):
        from blueprinting.megatron import execute_megatron_worker

        execute_megatron_worker()


def __main__():
    from .patch import _ParseKeywordArgs

    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.core._ParseKeywordArgs = _ParseKeywordArgs
    fire.Fire(
        {"llm": LLM, "train": LLM.train, "infer": LLM.infer, "megatron": LLM.megatron}
    )

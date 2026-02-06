"""LLM CLI module for blueprinting."""

import logging
import sys
from typing import List, Optional

import fire
import fire.decorators
import hyperparameter as hp

from .. import io
from ..types import Execution, Model, System


class LLM:
    """Support for large language models."""

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
        """Analysis LLM training."""
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
            exe = Execution(ps.exe)
            syst = System(ps.sys)

            # TODO: Implement blueprinting's own Llm simulator
            print(f"Model: {app.hidden}x{app.num_blocks} blocks")
            print(
                f"Execution: TP={exe.tensor_par}, PP={exe.pipeline_par}, DP={exe.data_par}"
            )
            print(f"System: {syst.proc_mode} mode")

        if stats is not None and io.is_json_extension(stats):
            # TODO: Implement stats collection
            pass

        if peers:
            # TODO: Implement peers output
            pass

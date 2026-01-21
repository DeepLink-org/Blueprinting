"""Validation case for seqsel figure 1.

NOTE: This validation case intentionally uses calculon for comparison purposes.
It validates blueprinting's results against calculon's reference implementation.
"""

import logging

import pandas as pd
import hyperparameter as hp

# Calculon is used here for validation comparison only
import calculon
from calculon.llm import Llm, System as CalculonSystem

from blueprinting import Execution, Model, io

kProfile = {
    "megatron-22B": {
        "none": {"par_opt": 45.5625, "act": 59.25},
        "seqsel": {"par_opt": 45.5625, "act": 9.5625},
    },
    "gpt3-175B": {
        "none": {"par_opt": 45.5625, "act": 66.84375},
        "seqsel": {"par_opt": 45.5625, "act": 12.3515625},
    },
    "turing-530B": {
        "none": {"par_opt": 31.640625, "act": 114.0234375},
        "seqsel": {"par_opt": 31.640625, "act": 23.076171875},
    },
    "megatron-1T": {
        "none": {"par_opt": 32.958984375, "act": 131.25},
        "seqsel": {"par_opt": 32.958984375, "act": 26.5625},
    },
}

mem_usage = pd.DataFrame.from_records(
    [
        {
            "model": model,
            "system": "a100_80e",
            "mode": mode,
            "w+opt mem(GiB)": kProfile[model][mode]["par_opt"],
            "act mem(GiB)": kProfile[model][mode]["act"],
        }
        for model in kProfile
        for mode in kProfile[model]
    ]
)


def seqsel_fig1(show=False):
    models = mem_usage["model"].unique()
    systems = mem_usage["system"].unique()
    modes = mem_usage["mode"].unique()

    records = []

    for m, s, e in [(m, s, e) for m in models for s in systems for e in modes]:
        app_json = f"data/models/{m}.json"
        sys_json = f"data/systems/{s}.json"
        exe_json = f"data/validation/seqsel/fig1/{m}_{e}.json"

        logger = logging.getLogger()

        app_json = io.read_json_file(app_json)
        sys_json = io.read_json_file(sys_json)
        exe_json = io.read_json_file(exe_json)

        with hp.scope(app=app_json, sys=sys_json, exe=exe_json) as ps:
            app = Model(ps.app)
            exe = Execution(ps.exe)
            # Use calculon's System for validation
            syst = CalculonSystem(sys_json)

            model = Llm(app, logger)
            model.compile(syst, exe)
            model.run(syst)
            stats = model.get_stats_json(False)
            act_par_opt = (stats["weight_space"] + stats["weight_grad_space"] + stats["optimizer_space"]) / (1024**3)
            act_act = stats["act_space"] / (1024**3)
            records += [
                {
                    "model": m,
                    "system": s,
                    "mode": e,
                    "w+opt mem(GiB)": act_par_opt,
                    "act mem(GiB)": act_act,
                }
            ]
    df = pd.DataFrame.from_records(records)
    result = (
        mem_usage.set_index(["model", "system", "mode"])
        .join(
            df.set_index(["model", "system", "mode"]),
            on=["model", "system", "mode"],
            lsuffix="[act]",
            rsuffix="[pred]",
        )
        .reset_index()
    )

    result["w+opt mem/rtol"] = (result["w+opt mem(GiB)[act]"] - result["w+opt mem(GiB)[pred]"]).abs() / result[
        "w+opt mem(GiB)[act]"
    ]
    result["act mem/rtol"] = (result["act mem(GiB)[act]"] - result["act mem(GiB)[pred]"]).abs() / result[
        "act mem(GiB)[act]"
    ]

    return (
        result.style.format(
            {
                "w+opt mem/rtol": lambda x: "%.2f%%" % (100 * x),
                "act mem/rtol": lambda x: "%.2f%%" % (100 * x),
            }
        )
        if show
        else result
    )

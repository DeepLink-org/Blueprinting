"""Validation case for seqsel table 5.

NOTE: This validation case intentionally uses calculon for comparison purposes.
It validates blueprinting's results against calculon's reference implementation.
"""

import logging

import hyperparameter as hp
import pandas as pd

from blueprinting import Execution, Model, io

# Calculon is used here for validation comparison only
from calculon.llm import Llm
from calculon.llm import System as CalculonSystem

kProfile = {
    "megatron-22B": {"full": 1.42, "seqsel": 1.10},
    "gpt3-175B": {"full": 18.13, "seqsel": 13.75},
    "turing-530B": {"full": 49.05, "seqsel": 37.83},
    "megatron-1T": {"full": 94.42, "seqsel": 71.49},
}

iter_time = pd.DataFrame.from_records(
    [
        {
            "model": model,
            "system": "a100_80g",
            "mode": mode,
            "iter time(s)": kProfile[model][mode],
        }
        for model in kProfile
        for mode in kProfile[model]
    ]
)


def seqsel_tab5(show=False):
    models = iter_time["model"].unique()
    systems = iter_time["system"].unique()
    modes = iter_time["mode"].unique()

    records = []

    for m, s, e in [(m, s, e) for m in models for s in systems for e in modes]:
        app_json = f"data/models/{m}.json"
        sys_json = f"data/systems/{s}.json"
        exe_json = f"data/validation/seqsel/tab5/{m}_{e}.json"

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
            act_act = stats["total_time"]
            records += [
                {
                    "model": m,
                    "system": s,
                    "mode": e,
                    "iter time(s)": act_act,
                }
            ]
    df = pd.DataFrame.from_records(records)
    result = (
        iter_time.set_index(["model", "system", "mode"])
        .join(
            df.set_index(["model", "system", "mode"]),
            on=["model", "system", "mode"],
            lsuffix="[act]",
            rsuffix="[pred]",
        )
        .reset_index()
    )

    result["iter time/rtol"] = (
        result["iter time(s)[act]"] - result["iter time(s)[pred]"]
    ).abs() / result["iter time(s)[act]"]

    return (
        result.style.format({"iter time/rtol": lambda x: "%.2f%%" % (100 * x)})
        if show
        else result
    )

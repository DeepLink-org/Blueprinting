"""Legacy Calculon reproduction of SeqSel figure 7.

This compatibility check does not exercise Blueprinting's canonical derivation
path; the strict production gate lives in :mod:`blueprinting.validation.calculon`.
"""

import logging

import hyperparameter as hp
import pandas as pd

import blueprinting.io as io
from blueprinting.types import Execution, Model

# Calculon is used here for validation comparison only
from calculon.llm import Llm
from calculon.llm import System as CalculonSystem

kProfile = {
    "megatron-22B": {
        "none": 100.00,
        "seq": 66.84,
        "sel": 49.42,
        "seqsel": 16.18,
        "full": 7.64,
    },
    "gpt3-175B": {
        "none": 100.00,
        "seq": 62.04,
        "sel": 56.53,
        "seqsel": 18.49,
        "full": 8.71,
    },
    "turing-530B": {
        "none": 100.00,
        "seq": 58.31,
        "sel": 62.04,
        "seqsel": 20.27,
        "full": 9.42,
    },
    "megatron-1T": {
        "none": 100.00,
        "seq": 58.31,
        "sel": 62.04,
        "seqsel": 20.27,
        "full": 9.42,
    },
}


mem_usage = pd.DataFrame.from_records(
    [
        {
            "model": model,
            "system": "a100_80e",
            "mode": mode,
            "act mem(%)": kProfile[model][mode],
        }
        for model in kProfile
        for mode in kProfile[model]
    ]
)


def seqsel_fig7(show=False):
    models = mem_usage["model"].unique()
    systems = mem_usage["system"].unique()
    modes = mem_usage["mode"].unique()

    records = []

    for m, s, e in [(m, s, e) for m in models for s in systems for e in modes]:
        app_json = f"data/models/{m}.json"
        sys_json = f"data/systems/{s}.json"
        exe_json = f"data/validation/seqsel/fig7/{m}_{e}.json"

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
            act_act = stats["act_space"] + stats["act_checkpoint_size"]
            records += [
                {
                    "model": m,
                    "system": s,
                    "mode": e,
                    "act mem(%)": act_act,
                }
            ]
    df = pd.DataFrame.from_records(records)
    selected = df[df["mode"] == "none"]
    for _, row in df.iterrows():
        x = selected[(selected.model == row.model) & (selected.system == row.system)]
        df.loc[
            (df.model == row.model) & (df.system == row.system) & (df["mode"] == row["mode"]),
            "act mem(%)",
        ] = 100 * row["act mem(%)"] / x.iloc[0]["act mem(%)"]
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

    result["act mem/rtol"] = (result["act mem(%)[act]"] - result["act mem(%)[pred]"]).abs() / result["act mem(%)[act]"]

    return (
        result.style.format(
            {
                "act mem/rtol": lambda x: "%.2f%%" % (100 * x),
                "act mem(%)[act]": "{:.2f}%",
                "act mem(%)[pred]": "{:.2f}%",
            }
        )
        if show
        else result
    )

"""Convert a simulated episode to runtime-format inputs (dicts of strings)."""
import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd
import hashlib

SUBJ = ["normalized_name", "provider", "release_date", "access_date", "harness",
        "reasoning_effort", "harness_version", "subject_features_extra"]
_T = {}


def tables():
    if not _T:
        cs = pd.read_parquet("data/proc/code_subjects.parquet")
        S = pd.read_parquet("data/proc/subjects.parquet")
        subj = cs.merge(S, on=["bench", "subject_id"], how="left").set_index("s").sort_index()
        rows = pd.read_parquet("data/proc/code_rows.parquet").set_index("r").sort_index()
        items = pd.read_parquet("data/proc/code_items.parquet").set_index("i").sort_index()
        _T.update(subj=subj, rows=rows, items=items)
    return _T


def anon(bench):
    return "benchmark_" + str(int(hashlib.sha1(bench.encode()).hexdigest(), 16) % 900000 + 100000)


def to_input(s, r):
    t = tables()
    sj = t["subj"].loc[s]
    subject = {k: ("" if pd.isna(sj[k]) else str(sj[k])) for k in SUBJ}
    row = t["rows"].loc[r]
    it = t["items"].loc[row.i]
    item = {"item_content": str(it.content), "item_features": str(it.item_features),
            "interactors": str(row.interactors), "benchmark_id": anon(str(row.bench))}
    return [subject, item]

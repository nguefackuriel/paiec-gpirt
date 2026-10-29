"""Build unified training tables from measurement-db.

Outputs (data/proc/):
  obs.parquet      one row per (bench, subject, item, interactors): n trials, k correct
  items.parquet    (bench, item_id, content, item_features)
  subjects.parquet (bench, subject_id, runtime subject fields)
"""
import glob, os, sys
from concurrent.futures import ProcessPoolExecutor
import pandas as pd

ROOT = "data/mdb"
OUT = "data/proc"
SUBJ_KEYS = ["normalized_name", "provider", "release_date", "access_date", "harness",
             "reasoning_effort", "harness_version", "subject_features_extra"]


def find(d, t):
    p = glob.glob(f"{ROOT}/{d}/**/{t}.parquet", recursive=True)
    return p[0] if p else None


def one(d):
    r = pd.read_parquet(find(d, "responses") or find(d, "response"),
                        columns=["subject_id", "item_id", "interactors", "response"])
    r = r[r.response.notna()]
    r = r[r.response.isin([0.0, 1.0])]
    if r.empty:
        return None
    r["interactors"] = r["interactors"].fillna("").astype(str)
    g = r.groupby(["subject_id", "item_id", "interactors"], observed=True).response.agg(["size", "sum"]).reset_index()
    g.columns = ["subject_id", "item_id", "interactors", "n", "k"]
    g["k"] = g["k"].astype("int32"); g["n"] = g["n"].astype("int32")
    g.insert(0, "bench", d)
    it = pd.read_parquet(find(d, "items"), columns=["item_id", "content", "item_features"])
    it = it[it.item_id.isin(set(g.item_id))].copy()
    it["content"] = it["content"].fillna("").astype(str)
    it["item_features"] = it["item_features"].fillna("").astype(str)
    it.insert(0, "bench", d)
    s = pd.read_parquet(find(d, "subjects"))
    s = s[s.subject_id.isin(set(g.subject_id))].copy()
    for k in SUBJ_KEYS + ["display_name"]:
        s[k] = s[k].fillna("").astype(str) if k in s else ""
    s = s[["subject_id", "display_name"] + SUBJ_KEYS]
    s.insert(0, "bench", d)
    return g, it, s


if __name__ == "__main__":
    cat = pd.read_csv("data/benchmarks_catalog.csv")
    elig = cat[(cat.response_type == "binary") & (cat.granularity == "item") & (cat.release == "public")].dir.tolist()
    print(len(elig), "eligible", flush=True)
    os.makedirs(OUT, exist_ok=True)
    G, I, S = [], [], []
    with ProcessPoolExecutor(32) as ex:
        for d, res in zip(elig, ex.map(one, elig)):
            if res is None:
                print("empty", d); continue
            G.append(res[0]); I.append(res[1]); S.append(res[2])
    G = pd.concat(G, ignore_index=True); I = pd.concat(I, ignore_index=True); S = pd.concat(S, ignore_index=True)
    for c in ["bench", "subject_id", "item_id", "interactors"]:
        G[c] = G[c].astype("category")
    G.to_parquet(f"{OUT}/obs.parquet"); I.to_parquet(f"{OUT}/items.parquet"); S.to_parquet(f"{OUT}/subjects.parquet")
    print(G.shape, I.shape, S.shape, G.n.sum(), G.k.sum() / G.n.sum())

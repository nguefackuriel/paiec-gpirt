"""Train the budget-0 prior: P(correct | subject fields, item text) across benchmarks.

For CV, one LightGBM per fold is trained on the other folds' benchmarks and
used to produce out-of-fold logits for every (subject, input row) in the
eligible data: data/proc/b0_oof.npy aligned with codes.parquet rows.
"""
import os, sys, json
sys.path.insert(0, '.')
import numpy as np, pandas as pd, lightgbm as lgb
from paiec import features as F

SUBJ = ["normalized_name", "provider", "release_date", "access_date", "harness",
        "reasoning_effort", "harness_version", "subject_features_extra"]


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))


def name_key(s):
    return (s or "").strip().lower()


def name_abilities(o, subj, groups_mask):
    """Relative logit ability per normalized name from the given training rows."""
    g = o[groups_mask].groupby("s").agg(b=("b", "first"), n=("n", "sum"), k=("k", "sum"))
    g["lg"] = logit((g.k + 0.5) / (g.n + 1))
    g["rel"] = g.lg - g.groupby("b").lg.transform("mean")
    g["name"] = subj.loc[g.index, "normalized_name"].map(name_key)
    g = g[g.name != ""]
    # weight each benchmark appearance equally; shrink toward 0
    a = g.groupby("name").rel.agg(["sum", "size"])
    return (a["sum"] / (a["size"] + 1.0)).to_dict()


def main():
    o = pd.read_parquet("data/proc/codes.parquet")
    cs = pd.read_parquet("data/proc/code_subjects.parquet")
    S = pd.read_parquet("data/proc/subjects.parquet")
    subj = cs.merge(S, on=["bench", "subject_id"], how="left").set_index("s").sort_index()
    for k in SUBJ:
        subj[k] = subj[k].fillna("")
    rows = pd.read_parquet("data/proc/code_rows.parquet").set_index("r").sort_index()
    items = pd.read_parquet("data/proc/code_items.parquet").set_index("i").sort_index()
    E = np.load("data/proc/emb_e5small.npy").astype(np.float32)
    folds = pd.read_csv("data/proc/folds.csv"); bfold = dict(zip(folds.bench, folds.fold))
    o["fold"] = cs.set_index("s").bench.map(bfold).loc[o.s].to_numpy()

    # item-row features
    R = np.stack([F.item_stats(items.content.iat[i], items.item_features.iat[i], rows.interactors.at[r])
                  for r, i in zip(rows.index, rows.i)])
    row_i = rows.i.to_numpy()

    oof = np.full(len(o), np.nan, np.float32)
    rng = np.random.default_rng(0)
    for f in range(5):
        tr_mask = (o.fold != f).to_numpy()
        abil = name_abilities(o, subj, tr_mask)
        SV = np.stack([F.subject_vector(subj.loc[s, SUBJ].to_dict(), abil.get(name_key(subj.at[s, "normalized_name"]), np.nan))
                       for s in subj.index])

        def featurize(idx):
            r = o.r.to_numpy()[idx]; s = o.s.to_numpy()[idx]
            return np.concatenate([SV[s], R[r], E[row_i[r]]], 1)

        tr = np.flatnonzero(tr_mask)
        # subsample: equal total weight per benchmark, cap rows per bench
        tb = o.b.to_numpy()[tr]
        cnt = np.bincount(tb)
        keep_p = np.minimum(1.0, 40000 / np.maximum(cnt[tb], 1))
        tr = tr[rng.random(len(tr)) < keep_p]
        tb = o.b.to_numpy()[tr]
        w = 1.0 / np.bincount(tb)[tb]
        y = (o.k.to_numpy()[tr] / o.n.to_numpy()[tr]).astype(np.float32)
        X = featurize(tr)
        ds = lgb.Dataset(X, y, weight=w * len(np.unique(tb)) * 1000 / w.sum() * 0 + w / w.mean())
        params = dict(objective="cross_entropy", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
                      feature_fraction=0.5, bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0,
                      verbose=-1, num_threads=24)
        model = lgb.train(params, ds, num_boost_round=400)
        te = np.flatnonzero(~tr_mask)
        for chunk in np.array_split(te, max(1, len(te) // 200000)):
            oof[chunk] = model.predict(featurize(chunk), raw_score=True)
        print(f"fold {f}: train rows {len(tr)}", flush=True)
        model.save_model(f"data/proc/b0_fold{f}.txt")
    np.save("data/proc/b0_oof.npy", oof)
    # quick eval: Brier per group then mean (budget-0 style)
    p = 1 / (1 + np.exp(-oof))
    n, k = o.n.to_numpy(float), o.k.to_numpy(float)
    se = k * (1 - p) ** 2 + (n - k) * p ** 2
    g = o.s.to_numpy()
    br = np.bincount(g, se) / np.maximum(np.bincount(g, n), 1)
    gb = o.groupby("s").b.first()
    per_bench = pd.Series(br[gb.index], index=gb.index).groupby(gb.values).mean()
    print("B0 brier mean over benches:", per_bench.mean(), " const0.5: 0.25")


if __name__ == "__main__":
    main()

"""Offline simulator of the PAIEC streaming evaluation protocol.

Reproduces: group eligibility (>=80 distinct items), the ~1000-distinct-item
draw with preference for under-represented benchmarks/subjects, the per-group
50/50 acquisition/evaluation split, nested random acquisition at caps
0,1,3,7,15,31 (labels shared across all groups), and Brier ALC scoring.

Data are integer coded: obs has columns g (group), b (bench), s (subject,
unique per bench), i (item, unique per bench), r (input row: item+interactors),
n, k. Groups are (b, s) pairs, so g == s.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BUDGETS = (0, 1, 3, 7, 15, 31)
WEIGHTS = np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1])
MIN_ITEMS = 80
DRAW_LIMIT = 1000


def encode(obs: pd.DataFrame) -> pd.DataFrame:
    """Integer-code a string obs table (bench, subject_id, item_id, interactors, n, k)."""
    bench = obs.bench.astype(str)
    o = pd.DataFrame({"n": obs.n.to_numpy(np.int32), "k": obs.k.to_numpy(np.int32)})
    o["b"] = pd.factorize(bench)[0].astype(np.int32)
    o["s"] = pd.factorize(bench + "\x00" + obs.subject_id.astype(str))[0].astype(np.int32)
    o["i"] = pd.factorize(bench + "\x00" + obs.item_id.astype(str))[0].astype(np.int32)
    o["r"] = pd.factorize(o.i.astype(str) + "\x00" + obs.interactors.astype(str))[0].astype(np.int32)
    o["g"] = o["s"]
    return o


def group_table(o: pd.DataFrame) -> pd.DataFrame:
    g = o.groupby("g").agg(b=("b", "first"), n_items=("i", "nunique")).reset_index()
    return g[(g.n_items >= MIN_ITEMS) & (g.n_items <= DRAW_LIMIT)].reset_index(drop=True)


def draw_pairs(groups: pd.DataFrame, rng: np.random.Generator, limit: int = DRAW_LIMIT,
               subj_key: dict | None = None) -> list[int]:
    """Greedy draw where the limit counts subject-item pairs (sum of group sizes).

    This matches the formative feedback (e.g. 9 groups of 88-152 items, mostly one
    subject per benchmark). Ordering as in `draw`.
    """
    cand = groups.g.to_numpy(); bench = groups.b.to_numpy(); size = groups.n_items.to_numpy()
    skey = np.array([subj_key.get(g, g) for g in cand], dtype=object) if subj_key else cand
    tie = rng.random(len(cand))
    alive = np.ones(len(cand), bool)
    used = 0
    bcount: dict = {}; scount: dict = {}
    chosen = []
    while alive.any():
        idx = np.flatnonzero(alive)
        bc = np.array([bcount.get(x, 0) for x in bench[idx]])
        sc = np.array([scount.get(x, 0) for x in skey[idx]])
        order = idx[np.lexsort((tie[idx], sc, bc))]
        picked = None
        for j in order:
            if used + size[j] <= limit:
                picked = j
                break
            alive[j] = False
        if picked is None:
            break
        alive[picked] = False
        chosen.append(int(cand[picked]))
        used += size[picked]
        bcount[bench[picked]] = bcount.get(bench[picked], 0) + 1
        scount[skey[picked]] = scount.get(skey[picked], 0) + 1
    return chosen


def draw(groups: pd.DataFrame, group_items: dict, rng: np.random.Generator,
         limit: int = DRAW_LIMIT, subj_key: dict | None = None) -> list[int]:
    """Greedy draw of complete groups under a distinct-item limit.

    Candidates are ordered by (#groups of the same bench already drawn,
    #groups of the same subject identity already drawn, random tie-break);
    the first candidate that fits the remaining item budget is added.
    """
    cand = groups.g.to_numpy(); bench = groups.b.to_numpy()
    skey = np.array([subj_key.get(g, g) for g in cand]) if subj_key else cand
    tie = rng.random(len(cand))
    alive = np.ones(len(cand), bool)
    items: set = set()
    bcount: dict = {}; scount: dict = {}
    chosen = []
    while alive.any():
        idx = np.flatnonzero(alive)
        bc = np.array([bcount.get(x, 0) for x in bench[idx]])
        sc = np.array([scount.get(x, 0) for x in skey[idx]])
        order = idx[np.lexsort((tie[idx], sc, bc))]
        picked = None
        for j in order:
            new = group_items[cand[j]] - items
            if len(items) + len(new) <= limit:
                picked = j
                break
            alive[j] = False
        if picked is None:
            break
        alive[picked] = False
        chosen.append(int(cand[picked]))
        items |= group_items[cand[picked]]
        bcount[bench[picked]] = bcount.get(bench[picked], 0) + 1
        scount[skey[picked]] = scount.get(skey[picked], 0) + 1
    return chosen


def build_episode(o: pd.DataFrame, chosen, rng: np.random.Generator):
    """Split each chosen group 50/50 and build the nested acquisition stream."""
    sub = o[o.g.isin(chosen)]
    gi = sub[["g", "i"]].drop_duplicates()
    gi = gi.assign(u=rng.random(len(gi))).sort_values(["g", "u"])
    gi["rank"] = gi.groupby("g").cumcount()
    gi["cnt"] = gi.groupby("g").i.transform("size")
    gi["acq"] = gi["rank"] < (gi["cnt"] + 1) // 2
    sub = sub.merge(gi[["g", "i", "acq", "rank"]], on=["g", "i"])
    ev = sub[~sub.acq].drop(columns=["acq", "rank"]).reset_index(drop=True)
    aq = sub[sub.acq & (sub["rank"] < max(BUDGETS))]
    # one input row per acquired item (if several interactor variants), one sampled trial
    aq = aq.assign(u=rng.random(len(aq))).sort_values("u").drop_duplicates(["g", "i"])
    aq = aq.sort_values(["g", "rank"]).reset_index(drop=True)
    aq["y"] = (rng.random(len(aq)) < aq.k / aq.n).astype(np.int8)
    return ev, aq.drop(columns=["acq", "u"])


def score_episode(ev: pd.DataFrame, preds: dict) -> pd.DataFrame:
    n, k = ev.n.to_numpy(float), ev.k.to_numpy(float)
    g = ev.g.to_numpy()
    nsum = np.bincount(g, n)
    keep = nsum > 0
    out = {}
    for b in BUDGETS:
        p = np.clip(preds[b], 0, 1)
        se = k * (1 - p) ** 2 + (n - k) * p ** 2
        out[f"B{b}"] = np.bincount(g, se, minlength=len(nsum))[keep] / nsum[keep]
    df = pd.DataFrame(out, index=np.flatnonzero(keep))
    df["ALC"] = df[[f"B{b}" for b in BUDGETS]].to_numpy() @ WEIGHTS
    return df


def run_episode(model, o, chosen, rng):
    ev, aq = build_episode(o, chosen, rng)
    preds = {b: np.asarray(model.predict_batch(ev, aq[aq["rank"] < b]), float) for b in BUDGETS}
    return score_episode(ev, preds)


CAPS = (1, 3, 7, 15, 31)


def build_stream_episode(o: pd.DataFrame, chosen, rng: np.random.Generator):
    """Like build_episode but returns the full acquisition pool in stream order."""
    sub = o[o.g.isin(chosen)]
    gi = sub[["g", "i", "b"]].drop_duplicates(["g", "i"])
    gi = gi.assign(u=rng.random(len(gi))).sort_values(["g", "u"])
    gi["rank"] = gi.groupby("g").cumcount()
    gi["cnt"] = gi.groupby("g").i.transform("size")
    gi["acq"] = gi["rank"] < (gi["cnt"] + 1) // 2
    sub = sub.merge(gi[["g", "i", "acq"]], on=["g", "i"])
    ev = sub[~sub.acq].drop(columns=["acq"]).reset_index(drop=True)
    pool = sub[sub.acq]
    pool = pool.assign(u=rng.random(len(pool))).sort_values("u").drop_duplicates(["g", "i"])
    # stream order: random permutation within group (fresh random key)
    pool = pool.assign(order=rng.random(len(pool))).sort_values(["g", "order"]).reset_index(drop=True)
    pool["pos"] = pool.groupby("g").cumcount()
    pool["y"] = (rng.random(len(pool)) < pool.k / pool.n).astype(np.int8)
    return ev, pool.drop(columns=["acq", "u", "order"])


def run_stream(pool: pd.DataFrame, policy, rng: np.random.Generator):
    """Simulate streaming acquisition; returns acquired rows with 'rank' (0-based order within group).

    policy(state, row, context) -> bool. `state` is a dict the policy may use for
    bookkeeping; it receives the acquired rows so far via state['acquired'].
    """
    groups = pool.g.unique()
    by_g = {g: d for g, d in pool.groupby("g", sort=False)}
    pos = {g: 0 for g in groups}
    n_acq = {g: 0 for g in groups}
    acquired = []
    state = {"acquired": acquired}
    for cap in CAPS:
        for g in rng.permutation(groups):
            d = by_g[g]
            while n_acq[g] < cap and pos[g] < len(d):
                row = d.iloc[pos[g]]
                ctx = {"labels_remaining": 31 - n_acq[g], "labels_acquired": n_acq[g], "max_labels": 31,
                       "items_remaining": len(d) - pos[g], "cap": cap}
                pos[g] += 1
                if policy(state, row, ctx):
                    r = row.copy(); r["rank"] = n_acq[g]
                    acquired.append(r)
                    n_acq[g] += 1
    return pd.DataFrame(acquired)


def run_stream_episode(model, o, chosen, rng, policy):
    ev, pool = build_stream_episode(o, chosen, rng)
    aq = run_stream(pool, policy, rng)
    preds = {b: np.asarray(model.predict_batch(ev, aq[aq["rank"] < b] if len(aq) else aq), float) for b in BUDGETS}
    return score_episode(ev, preds)

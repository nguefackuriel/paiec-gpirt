"""Stacking features from the online model, computable at runtime.

`detail(model, ev, lab)` runs the GPBMA6 model and returns, per eval row, the
BMA predictive probability plus diagnostics (BMA-mean logit mean / variance,
evidence-weighted prior scales, label counts, empirical rates, similarity of the
target item to labeled items). These are the inputs of the meta-model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import gpirt
from .models import load_emb, _STATS

FEATURES = ["p_gp", "mu", "logvar", "n_s", "k_s", "rate_s", "n_b", "rate_b", "n_subj_b", "n_items_L",
            "item_lab_n", "item_lab_rate", "max_cos", "mean_cos", "sa_w", "sd_w", "p_beta", "budget_cap",
            "len", "nlen_z"]


def detail(model, ev: pd.DataFrame, lab: pd.DataFrame, budget: int) -> pd.DataFrame:
    E = load_emb(model.emb)
    if "data/proc/item_stats.npy" not in _STATS:
        _STATS["data/proc/item_stats.npy"] = np.load("data/proc/item_stats.npy")
    ST = _STATS["data/proc/item_stats.npy"]
    out = pd.DataFrame(index=ev.index, columns=FEATURES, dtype=float)
    lb = lab.b.to_numpy() if len(lab) else np.zeros(0, int)
    lp = model.log_prior()
    grid = [(sa, sd) for sa in model.s_a_grid for sd in model.s_d_grid]
    for b, idx in ev.groupby("b").indices.items():
        e = ev.iloc[idx]
        l = lab[lb == b]
        f = out.iloc[idx].copy()
        f["budget_cap"] = budget
        f["len"] = ST[e.i.to_numpy(), 0]
        f["n_b"] = len(l)
        f["n_subj_b"] = l.s.nunique() if len(l) else 0
        f["rate_b"] = (l.y.mean() if len(l) else 0.5)
        ns = l.groupby("s").y.agg(["size", "sum"]) if len(l) else pd.DataFrame(columns=["size", "sum"])
        f["n_s"] = e.s.map(ns["size"]).fillna(0).to_numpy()
        f["k_s"] = e.s.map(ns["sum"]).fillna(0).to_numpy()
        f["rate_s"] = (f.k_s + 0.5) / (f.n_s + 1)
        f["p_beta"] = (f.k_s + 1) / (f.n_s + 2)
        li = l.groupby("i").y.agg(["size", "mean"]) if len(l) else pd.DataFrame(columns=["size", "mean"])
        f["item_lab_n"] = e.i.map(li["size"]).fillna(0).to_numpy()
        f["item_lab_rate"] = e.i.map(li["mean"]).fillna(-1).to_numpy()
        if len(l) == 0:
            f["p_gp"] = 0.5; f["mu"] = 0.0; f["logvar"] = np.log(model.s_m ** 2 + 4); f["n_items_L"] = 0
            f["max_cos"] = 0; f["mean_cos"] = 0; f["sa_w"] = 0; f["sd_w"] = 0; f["nlen_z"] = 0
            out.iloc[idx] = f.to_numpy(); continue
        subj = pd.Index(np.unique(np.concatenate([e.s.to_numpy(), l.s.to_numpy()])))
        Litems = np.unique(l.i.to_numpy()); Lidx = pd.Index(Litems)
        ti, tinv = np.unique(e.i.to_numpy(), return_inverse=True)
        K_LL, K_tL, k_tt = model.kernels(Litems, ti)
        ls, ll, y = subj.get_indexer(l.s), Lidx.get_indexer(l.i), l.y.to_numpy(float)
        ts = subj.get_indexer(e.s)
        fits = [gpirt.fit(ls, ll, y, np.zeros(len(l)), K_LL, len(subj), model.s_m, sa, sd) for sa, sd in grid]
        w = gpirt.bma_weights(fits, lp)
        p = np.zeros(len(e)); mu_m = np.zeros(len(e)); var_m = np.zeros(len(e)); tot = 0
        for wf, ft in zip(w, fits):
            if wf < 1e-4:
                continue
            mu, var = gpirt.predict_moments(ft, ts, K_tL[tinv], k_tt[tinv], np.zeros(len(e)))
            p += wf * gpirt.predictive(mu, var, model.var_scale); mu_m += wf * mu; var_m += wf * var; tot += wf
        f["p_gp"] = p / tot; f["mu"] = mu_m / tot; f["logvar"] = np.log(var_m / tot + 1e-9)
        f["sa_w"] = float(sum(wf * np.log(sa) for wf, (sa, sd) in zip(w, grid)))
        f["sd_w"] = float(sum(wf * np.log(sd) for wf, (sa, sd) in zip(w, grid)))
        f["n_items_L"] = len(Litems)
        C = E[ti] @ E[Litems].T
        f["max_cos"] = C.max(1)[tinv]; f["mean_cos"] = C.mean(1)[tinv]
        Ls = ST[Litems, 0]
        f["nlen_z"] = (f["len"] - Ls.mean()) / (Ls.std() + 1e-6)
        out.iloc[idx] = f.to_numpy()
    return out

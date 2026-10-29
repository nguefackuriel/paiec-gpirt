"""Acquisition policies for the streaming simulator (cheap, prediction-free)."""
from __future__ import annotations

import numpy as np


def _must(ctx):
    return ctx["items_remaining"] <= ctx["labels_remaining"]


class RandomPolicy:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def __call__(self, state, row, ctx):
        return bool(self.rng.random() < min(1.0, ctx["labels_remaining"] / ctx["items_remaining"]))


class OverlapPolicy:
    """Prefer items already labeled for other subjects (mode='overlap') or new items ('novel').

    Accept preferred items; accept non-preferred with probability `p_other`,
    and always when the remaining stream is needed to fill the budget.
    """
    def __init__(self, mode="overlap", p_other=0.1, slack=1.0, seed=0):
        self.mode, self.p_other, self.slack = mode, p_other, slack
        self.rng = np.random.default_rng(seed)
        self.seen = {}

    def __call__(self, state, row, ctx):
        acq = state["acquired"]
        # incremental index of labeled items per bench
        n_done = state.setdefault("_n", 0)
        for r in acq[n_done:]:
            self.seen.setdefault(int(r["i"]), 0)
            self.seen[int(r["i"])] += 1
        state["_n"] = len(acq)
        labeled_before = self.seen.get(int(row["i"]), 0) > 0
        preferred = labeled_before if self.mode == "overlap" else not labeled_before
        need = ctx["labels_remaining"] * self.slack
        if ctx["items_remaining"] <= need:
            return True
        return bool(preferred or self.rng.random() < self.p_other)


class DiversityPolicy:
    """Accept a candidate when its kernel similarity to already-acquired items of the
    same subject is below `tau`; fall back to filling the budget when needed.

    The threshold relaxes linearly as the stream runs out: accept if
    max_sim < tau or items_remaining <= labels_remaining * slack.
    """
    def __init__(self, tau=0.5, ell=0.05, slack=1.5, emb="data/proc/emb_e5small.npy"):
        self.tau, self.ell, self.slack, self.emb = tau, ell, slack, emb
        self.acq_by_g = {}

    def __call__(self, state, row, ctx):
        from .models import load_emb
        E = load_emb(self.emb)
        g = int(row["g"])
        prev = self.acq_by_g.setdefault(g, [])
        if ctx["items_remaining"] <= ctx["labels_remaining"] * self.slack:
            ok = True
        elif not prev:
            ok = True
        else:
            c = E[prev] @ E[int(row["i"])]
            ok = float(np.exp((c.max() - 1) / self.ell)) < self.tau
        if ok:
            prev.append(int(row["i"]))
        return bool(ok)


class VariancePolicy:
    """Accept a candidate whose prior predictive variance given already-acquired items of
    the same subject is high (information-greedy), using the model's linear+RBF kernel on
    unit embeddings. Falls back to filling the budget when the stream runs short.

    var(t | A) = k_tt - k_tA K_AA^-1 k_At with K = rho_le * centered-linear + rho * RBF + rest * I.
    The linear part is centered on the running mean of candidates seen for the subject.
    """
    def __init__(self, tau=0.6, slack=1.5, rho=0.365, rho_le=0.5, ell=0.12, emb="data/proc/emb_e5small.npy"):
        self.tau, self.slack, self.rho, self.rho_le, self.ell, self.emb = tau, slack, rho, rho_le, ell, emb
        self.acq, self.seen = {}, {}

    def _k(self, E, a, b, mu):
        za, zb = E[a] - mu, E[b] - mu
        lin = za @ zb.T
        rbf = np.exp((E[a] @ E[b].T - 1) / self.ell)
        same = (np.asarray(a)[:, None] == np.asarray(b)[None, :]).astype(float)
        return self.rho_le * lin + self.rho * rbf + (1 - self.rho - self.rho_le) * same

    def __call__(self, state, row, ctx):
        from .models import load_emb
        E = load_emb(self.emb)
        g, i = int(row["g"]), int(row["i"])
        seen = self.seen.setdefault(g, []); seen.append(i)
        A = self.acq.setdefault(g, [])
        if ctx["items_remaining"] <= ctx["labels_remaining"] * self.slack or len(A) == 0:
            ok = True
        else:
            mu = E[seen].mean(0)
            q = max(1e-9, float(((E[seen] - mu) ** 2).sum(1).mean()))
            Ea = E  # scale linear part by q via centering below
            KAA = self._k(E, A, A, mu); KAA[np.diag_indices(len(A))] += 1e-6
            lin_scale = 1.0 / q
            KAA = KAA + (lin_scale - 1) * self.rho_le * ((E[A] - mu) @ (E[A] - mu).T)
            kt = self._k(E, [i], A, mu)[0] + (lin_scale - 1) * self.rho_le * ((E[i] - mu) @ (E[A] - mu).T)
            ktt = self.rho_le * lin_scale * float((E[i] - mu) @ (E[i] - mu)) + self.rho + (1 - self.rho - self.rho_le)
            var = ktt - kt @ np.linalg.solve(KAA, kt)
            ok = var / ktt >= self.tau
        if ok:
            A.append(i)
        return bool(ok)

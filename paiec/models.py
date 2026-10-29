"""Batch predictors used by the simulator (same information as the runtime).

ev / lab frames carry integer codes b (bench), s (subject), i (item), r (input
row); lab also has y. Predictors must not read ev.n / ev.k.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import irt


class Constant:
    def __init__(self, p=0.5):
        self.p = p

    def predict_batch(self, ev, lab):
        return np.full(len(ev), self.p)


class GroupMean:
    """Beta-shrunk mean of the target group's labels."""
    def __init__(self, prior=0.54, strength=2.0):
        self.prior, self.strength = prior, strength

    def predict_batch(self, ev, lab):
        n = np.bincount(lab.s, minlength=ev.s.max() + 1) if len(lab) else 0
        k = np.bincount(lab.s, lab.y, minlength=ev.s.max() + 1) if len(lab) else 0
        n = np.zeros(ev.s.max() + 1) + np.pad(n, (0, 0)) if len(lab) else np.zeros(ev.s.max() + 1)
        k = np.zeros(ev.s.max() + 1) + k
        n = n[:ev.s.max() + 1]; k = k[:ev.s.max() + 1]
        return ((k + self.prior * self.strength) / (n + self.strength))[ev.s.to_numpy()]


class IRT:
    """Online hierarchical IRT with optional fixed prior offsets."""
    def __init__(self, s_m=1.5, s_a=1.0, s_d=1.0, m0=0.0, use_items=True):
        self.s_m, self.s_a, self.s_d, self.m0, self.use_items = s_m, s_a, s_d, m0, use_items

    def base(self, df):
        return np.full(len(df), self.m0)

    def predict_batch(self, ev, lab):
        out = np.empty(len(ev))
        ev_base = self.base(ev)
        lab_base = self.base(lab) if len(lab) else np.zeros(0)
        lb = lab.b.to_numpy() if len(lab) else np.zeros(0, int)
        for b, idx in ev.groupby("b").indices.items():
            e = ev.iloc[idx]
            lmask = lb == b
            l = lab[lmask]
            subj = pd.Index(np.unique(np.concatenate([e.s.to_numpy(), l.s.to_numpy()])))
            items = pd.Index(np.unique(np.concatenate([e.i.to_numpy(), l.i.to_numpy()])))
            s_d = self.s_d if self.use_items else 1e-3
            out[idx] = irt.fit_predict(
                subj.get_indexer(l.s), items.get_indexer(l.i), l.y.to_numpy(float), lab_base[lmask],
                subj.get_indexer(e.s), items.get_indexer(e.i), ev_base[idx],
                len(subj), len(items), self.s_m, self.s_a, s_d)
        return out


_EMB = {}


def load_emb(path="data/proc/emb_e5small.npy"):
    if path not in _EMB:
        _EMB[path] = np.load(path).astype(np.float32)
    return _EMB[path]


class GPIRT:
    """IRT with GP prior on item effects from text-embedding similarity.

    Kernel: K = rho * exp((cos - 1) / ell) + (1 - rho) * [same item].
    """
    def __init__(self, s_m=1.5, s_a=1.0, s_d=1.0, m0=0.0, rho=0.5, ell=0.1, emb="data/proc/emb_e5small.npy"):
        self.s_m, self.s_a, self.s_d, self.m0, self.rho, self.ell, self.emb = s_m, s_a, s_d, m0, rho, ell, emb

    def kern(self, ia, ib):
        E = load_emb(self.emb)
        c = E[ia] @ E[ib].T
        return self.rho * np.exp((c - 1.0) / self.ell) + (1 - self.rho) * (ia[:, None] == ib[None, :])

    def predict_batch(self, ev, lab):
        out = np.empty(len(ev))
        lb = lab.b.to_numpy() if len(lab) else np.zeros(0, int)
        for b, idx in ev.groupby("b").indices.items():
            e = ev.iloc[idx]
            l = lab[lb == b]
            subj = pd.Index(np.unique(np.concatenate([e.s.to_numpy(), l.s.to_numpy()])))
            Litems = np.unique(l.i.to_numpy())
            Lidx = pd.Index(Litems)
            # predict per distinct target item, then broadcast
            ti, tinv = np.unique(e.i.to_numpy(), return_inverse=True)
            K_LL = self.kern(Litems, Litems) if len(Litems) else np.zeros((0, 0))
            K_tL = self.kern(ti, Litems)[tinv] if len(Litems) else np.zeros((len(e), 0))
            out[idx] = irt.fit_predict_gp(
                subj.get_indexer(l.s), Lidx.get_indexer(l.i), l.y.to_numpy(float), np.full(len(l), self.m0), K_LL,
                subj.get_indexer(e.s), K_tL, np.ones(len(e)), np.full(len(e), self.m0), len(subj),
                self.s_m, self.s_a, self.s_d)
        return out


from . import gpirt


class GPBMA:
    """GP-IRT with Bayesian model averaging over a (s_a, s_d) grid per benchmark."""
    def __init__(self, s_m=2.0, s_a_grid=(0.6, 1.2, 2.4), s_d_grid=(0.8, 1.6, 3.2), rho=0.85, ell=0.05,
                 var_scale=1.0, emb="data/proc/emb_e5small.npy", bma=True, temp=1.0):
        self.s_m, self.s_a_grid, self.s_d_grid = s_m, s_a_grid, s_d_grid
        self.rho, self.ell, self.var_scale, self.emb, self.bma, self.temp = rho, ell, var_scale, emb, bma, temp

    def kern(self, ia, ib):
        E = load_emb(self.emb)
        c = E[ia] @ E[ib].T
        return self.rho * np.exp((c - 1.0) / self.ell) + (1 - self.rho) * (ia[:, None] == ib[None, :])

    def predict_batch(self, ev, lab):
        out = np.empty(len(ev))
        lb = lab.b.to_numpy() if len(lab) else np.zeros(0, int)
        for b, idx in ev.groupby("b").indices.items():
            e = ev.iloc[idx]
            l = lab[lb == b]
            subj = pd.Index(np.unique(np.concatenate([e.s.to_numpy(), l.s.to_numpy()])))
            Litems = np.unique(l.i.to_numpy()); Lidx = pd.Index(Litems)
            ti, tinv = np.unique(e.i.to_numpy(), return_inverse=True)
            K_LL = self.kern(Litems, Litems) if len(Litems) else np.zeros((0, 0))
            K_tL = self.kern(ti, Litems) if len(Litems) else np.zeros((len(ti), 0))
            ls, ll, y = subj.get_indexer(l.s), Lidx.get_indexer(l.i), l.y.to_numpy(float)
            ts = subj.get_indexer(e.s)
            fits = [gpirt.fit(ls, ll, y, np.zeros(len(l)), K_LL, len(subj), self.s_m, sa, sd)
                    for sa in self.s_a_grid for sd in self.s_d_grid]
            w = gpirt.bma_weights(fits) if self.bma else np.full(len(fits), 1 / len(fits))
            if self.temp != 1.0:
                le = np.array([f.log_evidence for f in fits]) * self.temp
                w = np.exp(le - le.max()); w /= w.sum()
            p = np.zeros(len(e))
            for wf, f in zip(w, fits):
                if wf < 1e-4:
                    continue
                mu, var = gpirt.predict_moments(f, ts, K_tL[tinv], np.ones(len(e)), np.zeros(len(e)))
                p += wf * gpirt.predictive(mu, var, self.var_scale)
            out[idx] = p / w[w >= 1e-4].sum()
        return out


import pickle
from . import tokens as TK

_TOK = {}


def load_tokens(path="data/proc/item_tokens.pkl"):
    if path not in _TOK:
        _TOK[path] = pickle.load(open(path, "rb"))
    return _TOK[path]


class GPBMA2(GPBMA):
    """GPBMA + item-feature token kernel + log-normal hyperprior on the grid."""
    def __init__(self, rho_f=0.0, hp=0.0, s_a0=1.2, s_d0=1.6, **kw):
        super().__init__(**kw)
        self.rho_f, self.hp, self.s_a0, self.s_d0 = rho_f, hp, s_a0, s_d0

    def kernels(self, Litems, ti):
        E = load_emb(self.emb)
        rho_e, rho_f = self.rho, self.rho_f
        cLL = E[Litems] @ E[Litems].T; ctL = E[ti] @ E[Litems].T
        K_LL = rho_e * np.exp((cLL - 1) / self.ell) + (1 - rho_e - rho_f) * np.eye(len(Litems))
        K_tL = rho_e * np.exp((ctL - 1) / self.ell) + (1 - rho_e - rho_f) * (ti[:, None] == Litems[None, :])
        k_tt = np.full(len(ti), 1.0 - rho_f)
        if rho_f > 0:
            toks = load_tokens()
            XL, XT = TK.token_matrix([toks[i] for i in Litems], [toks[i] for i in ti])
            if XL.shape[1]:
                K_LL = K_LL + rho_f * TK.cosine_kernel(XL, XL)
                K_tL = K_tL + rho_f * TK.cosine_kernel(XT, XL)
                k_tt = k_tt + rho_f * (XT.sum(1) > 0)
            # items without tokens: give back the variance to the identity part
            if XL.shape[1]:
                noL = XL.sum(1) == 0
                K_LL[noL, noL] += rho_f
                same = (ti[:, None] == Litems[None, :])
                K_tL += rho_f * same * noL[None, :]
                k_tt = np.where(XT.sum(1) > 0, k_tt, k_tt + rho_f) if XT is not None else k_tt
            else:
                K_LL = K_LL + rho_f * np.eye(len(Litems))
                K_tL = K_tL + rho_f * (ti[:, None] == Litems[None, :])
                k_tt = k_tt + rho_f
        return K_LL, K_tL, k_tt

    def log_prior(self):
        return np.array([-self.hp * ((np.log(sa / self.s_a0)) ** 2 + (np.log(sd / self.s_d0)) ** 2)
                         for sa in self.s_a_grid for sd in self.s_d_grid])

    def predict_batch(self, ev, lab):
        out = np.empty(len(ev))
        lb = lab.b.to_numpy() if len(lab) else np.zeros(0, int)
        lp = self.log_prior()
        for b, idx in ev.groupby("b").indices.items():
            e = ev.iloc[idx]
            l = lab[lb == b]
            if len(l) == 0:
                out[idx] = 0.5
                continue
            subj = pd.Index(np.unique(np.concatenate([e.s.to_numpy(), l.s.to_numpy()])))
            Litems = np.unique(l.i.to_numpy()); Lidx = pd.Index(Litems)
            ti, tinv = np.unique(e.i.to_numpy(), return_inverse=True)
            K_LL, K_tL, k_tt = self.kernels(Litems, ti)
            ls, ll, y = subj.get_indexer(l.s), Lidx.get_indexer(l.i), l.y.to_numpy(float)
            ts = subj.get_indexer(e.s)
            fits = [gpirt.fit(ls, ll, y, np.zeros(len(l)), K_LL, len(subj), self.s_m, sa, sd)
                    for sa in self.s_a_grid for sd in self.s_d_grid]
            w = gpirt.bma_weights(fits, lp)
            p = np.zeros(len(e)); tot = 0.0
            for wf, f in zip(w, fits):
                if wf < 1e-4:
                    continue
                mu, var = gpirt.predict_moments(f, ts, K_tL[tinv], k_tt[tinv], np.zeros(len(e)))
                p += wf * gpirt.predictive(mu, var, self.var_scale); tot += wf
            out[idx] = p / tot
        return out


class GPBMA3(GPBMA2):
    """GPBMA2 + subject feature-token effects (beta ~ N(0, s_b^2))."""
    def __init__(self, s_b=0.0, sub_norm=True, **kw):
        super().__init__(**kw)
        self.s_b, self.sub_norm = s_b, sub_norm

    def phi(self, subj_codes, lab_subj_codes):
        if self.s_b <= 0:
            return None
        toks = load_tokens("data/proc/subject_tokens.pkl")
        XL, XT = TK.token_matrix([toks[s] for s in lab_subj_codes], [toks[s] for s in subj_codes])
        if XT.shape[1] == 0:
            return None
        if self.sub_norm:
            n = np.sqrt(XT.sum(1, keepdims=True)); XT = np.divide(XT, n, out=np.zeros_like(XT), where=n > 0)
        return XT

    def predict_batch(self, ev, lab):
        out = np.empty(len(ev))
        lb = lab.b.to_numpy() if len(lab) else np.zeros(0, int)
        lp = self.log_prior()
        for b, idx in ev.groupby("b").indices.items():
            e = ev.iloc[idx]
            l = lab[lb == b]
            if len(l) == 0:
                out[idx] = 0.5
                continue
            subj = pd.Index(np.unique(np.concatenate([e.s.to_numpy(), l.s.to_numpy()])))
            Litems = np.unique(l.i.to_numpy()); Lidx = pd.Index(Litems)
            ti, tinv = np.unique(e.i.to_numpy(), return_inverse=True)
            K_LL, K_tL, k_tt = self.kernels(Litems, ti)
            ls, ll, y = subj.get_indexer(l.s), Lidx.get_indexer(l.i), l.y.to_numpy(float)
            ts = subj.get_indexer(e.s)
            Phi = self.phi(subj.to_numpy(), np.unique(l.s.to_numpy()))
            fits = [gpirt.fit(ls, ll, y, np.zeros(len(l)), K_LL, len(subj), self.s_m, sa, sd, Phi=Phi, s_b=self.s_b)
                    for sa in self.s_a_grid for sd in self.s_d_grid]
            w = gpirt.bma_weights(fits, lp)
            p = np.zeros(len(e)); tot = 0.0
            for wf, f in zip(w, fits):
                if wf < 1e-4:
                    continue
                mu, var = gpirt.predict_moments(f, ts, K_tL[tinv], k_tt[tinv], np.zeros(len(e)))
                p += wf * gpirt.predictive(mu, var, self.var_scale); tot += wf
            out[idx] = p / tot
        return out


_METAK = {}


def load_metak(path):
    if path not in _METAK:
        _METAK[path] = np.load(path, allow_pickle=True).item()
    return _METAK[path]


class GPBMA4(GPBMA2):
    """GPBMA2 with a meta-learned item kernel exp(-|W e_i - W e_j|^2 / 2) (W per CV fold)."""
    def __init__(self, fold=-1, metak="data/proc/metak_e5s_d64.npy", mix=1.0, **kw):
        super().__init__(**kw)
        self.fold, self.metak, self.mix = fold, metak, mix
        self._Z = None

    def Z(self):
        if self._Z is None:
            W = load_metak(self.metak)[self.fold]["W"]
            self._Z = (load_emb(self.emb) @ W).astype(np.float32)
        return self._Z

    def kernels(self, Litems, ti):
        Z = self.Z()
        E = load_emb(self.emb)

        def k(a, b):
            za, zb = Z[a], Z[b]
            d2 = (za ** 2).sum(1)[:, None] + (zb ** 2).sum(1)[None, :] - 2 * za @ zb.T
            km = np.exp(-0.5 * np.maximum(d2, 0))
            if self.mix < 1.0:
                ke = np.exp((E[a] @ E[b].T - 1) / self.ell)
                km = self.mix * km + (1 - self.mix) * ke
            return km
        rho, rho_f = self.rho, self.rho_f
        K_LL = rho * k(Litems, Litems) + (1 - rho - rho_f) * np.eye(len(Litems))
        K_tL = rho * k(ti, Litems) + (1 - rho - rho_f) * (ti[:, None] == Litems[None, :])
        k_tt = np.full(len(ti), 1.0 - rho_f)
        if rho_f > 0:
            toks = load_tokens()
            XL, XT = TK.token_matrix([toks[i] for i in Litems], [toks[i] for i in ti])
            if XL.shape[1]:
                K_LL = K_LL + rho_f * TK.cosine_kernel(XL, XL)
                K_tL = K_tL + rho_f * TK.cosine_kernel(XT, XL)
                k_tt = k_tt + rho_f * (XT.sum(1) > 0)
                noL = XL.sum(1) == 0
                K_LL[noL, noL] += rho_f
                K_tL += rho_f * (ti[:, None] == Litems[None, :]) * noL[None, :]
                k_tt = np.where(XT.sum(1) > 0, k_tt, k_tt + rho_f)
            else:
                K_LL = K_LL + rho_f * np.eye(len(Litems))
                K_tL = K_tL + rho_f * (ti[:, None] == Litems[None, :])
                k_tt = k_tt + rho_f
        return K_LL, K_tL, k_tt


_STATS = {}


class GPBMA5(GPBMA2):
    """GPBMA2 + linear kernel on generic item statistics standardized within the benchmark."""
    def __init__(self, rho_l=0.1, stats="data/proc/item_stats.npy", **kw):
        super().__init__(**kw)
        self.rho_l, self.stats = rho_l, stats

    def kernels(self, Litems, ti):
        K_LL, K_tL, k_tt = super().kernels(Litems, ti)
        if self.rho_l <= 0:
            return K_LL, K_tL, k_tt
        if self.stats not in _STATS:
            _STATS[self.stats] = np.load(self.stats)
        S = _STATS[self.stats]
        SL, ST = S[Litems], S[ti]
        mu, sd = SL.mean(0), SL.std(0)
        keep = sd > 1e-6
        if not keep.any() or len(Litems) < 3:
            return K_LL, K_tL, k_tt
        ZL = (SL[:, keep] - mu[keep]) / sd[keep]; ZT = np.clip((ST[:, keep] - mu[keep]) / sd[keep], -5, 5)
        q = keep.sum()
        s = 1 - self.rho_l
        return (s * K_LL + self.rho_l * ZL @ ZL.T / q, s * K_tL + self.rho_l * ZT @ ZL.T / q,
                s * k_tt + self.rho_l * (ZT ** 2).sum(1) / q)


class GPBMA6(GPBMA2):
    """GPBMA2 + linear kernels on (a) item statistics and (b) benchmark-centered embeddings."""
    def __init__(self, rho_l=0.15, rho_le=0.0, stats="data/proc/item_stats2.npy", **kw):
        super().__init__(**kw)
        self.rho_l, self.rho_le, self.stats = rho_l, rho_le, stats

    @staticmethod
    def _lin(XL, XT, standardize=True):
        mu = XL.mean(0)
        sd = XL.std(0) if standardize else np.ones(XL.shape[1])
        keep = sd > 1e-6
        ZL = (XL[:, keep] - mu[keep]) / sd[keep]; ZT = (XT[:, keep] - mu[keep]) / sd[keep]
        if standardize:
            ZT = np.clip(ZT, -5, 5)
        q = max(1e-9, (ZL ** 2).sum(1).mean())
        return ZL @ ZL.T / q, ZT @ ZL.T / q, (ZT ** 2).sum(1) / q

    def kernels(self, Litems, ti):
        K_LL, K_tL, k_tt = super().kernels(Litems, ti)
        if len(Litems) < 3:
            return K_LL, K_tL, k_tt
        s = 1 - self.rho_l - self.rho_le
        K_LL, K_tL, k_tt = s * K_LL, s * K_tL, s * k_tt
        if self.rho_l > 0:
            if self.stats not in _STATS:
                _STATS[self.stats] = np.load(self.stats)
            S = _STATS[self.stats]
            a, b, c = self._lin(S[Litems], S[ti])
            K_LL, K_tL, k_tt = K_LL + self.rho_l * a, K_tL + self.rho_l * b, k_tt + self.rho_l * c
        if self.rho_le > 0:
            E = load_emb(self.emb)
            a, b, c = self._lin(E[Litems], E[ti], standardize=False)
            K_LL, K_tL, k_tt = K_LL + self.rho_le * a, K_tL + self.rho_le * b, k_tt + self.rho_le * c
        return K_LL, K_tL, k_tt


class Ensemble:
    """Average of sub-model predictions (in probability or logit space)."""
    def __init__(self, members, logit=False):
        self.members, self.logit = members, logit

    def predict_batch(self, ev, lab):
        P = np.stack([m.predict_batch(ev, lab) for m in self.members])
        if self.logit:
            P = np.clip(P, 1e-6, 1 - 1e-6)
            return gpirt.sigmoid(np.log(P / (1 - P)).mean(0))
        return P.mean(0)


class EnsFactory:
    def __init__(self, configs, logit=False):
        self.configs, self.logit = configs, logit

    def __call__(self, f):
        return Ensemble([GPBMA6(**c) for c in self.configs], self.logit)


class MultiSourceBMA:
    """Joint BMA over (embedding source x prior-scale grid) per benchmark, with tempered evidence."""
    def __init__(self, embs, temp=1.0, mix=0.0, **kw):
        self.members = [GPBMA7(emb=e, **kw) for e in embs]
        self.temp, self.mix = temp, mix

    def predict_batch(self, ev, lab):
        m0 = self.members[0]
        out = np.empty(len(ev))
        lb = lab.b.to_numpy() if len(lab) else np.zeros(0, int)
        for b, idx in ev.groupby("b").indices.items():
            e = ev.iloc[idx]
            l = lab[lb == b]
            tb = m0.zb(e.i.to_numpy())
            if len(l) == 0:
                out[idx] = gpirt.predictive(tb, np.full(len(e), m0.s_m ** 2 + 1.0), m0.var_scale) if m0.zpath else 0.5
                continue
            subj = pd.Index(np.unique(np.concatenate([e.s.to_numpy(), l.s.to_numpy()])))
            Litems = np.unique(l.i.to_numpy()); Lidx = pd.Index(Litems)
            ti, tinv = np.unique(e.i.to_numpy(), return_inverse=True)
            ls, ll, y = subj.get_indexer(l.s), Lidx.get_indexer(l.i), l.y.to_numpy(float)
            ts = subj.get_indexer(e.s)
            lbase = m0.zb(l.i.to_numpy())
            preds, les = [], []
            for m in self.members:
                K_LL, K_tL, k_tt = m.kernels(Litems, ti)
                for sa in m.s_a_grid:
                    for sd in m.s_d_grid:
                        f = gpirt.fit(ls, ll, y, lbase, K_LL, len(subj), m.s_m, sa, sd)
                        mu, var = gpirt.predict_moments(f, ts, K_tL[tinv], k_tt[tinv], tb)
                        preds.append(gpirt.predictive(mu, var, m.var_scale)); les.append(f.log_evidence)
            les = np.array(les) * self.temp
            w = np.exp(les - les.max()); w /= w.sum()
            # optional mixing with the uniform-over-sources average
            nsrc = len(self.members); per = len(preds) // nsrc
            wu = np.concatenate([np.exp(les[k * per:(k + 1) * per] - les[k * per:(k + 1) * per].max()) for k in range(nsrc)])
            for k in range(nsrc):
                wu[k * per:(k + 1) * per] /= wu[k * per:(k + 1) * per].sum() * nsrc
            w = (1 - self.mix) * w + self.mix * wu
            out[idx] = np.tensordot(w, np.stack(preds), 1)
        return out


class MultiCfgBMA:
    """Joint evidence-weighted BMA over several kernel configurations (GPBMA7 kwargs) x prior-scale grid."""
    def __init__(self, cfgs, temp=1.0):
        self.members = [GPBMA7(**c) for c in cfgs]
        self.temp = temp

    def predict_batch(self, ev, lab):
        m0 = self.members[0]
        out = np.empty(len(ev))
        lb = lab.b.to_numpy() if len(lab) else np.zeros(0, int)
        for b, idx in ev.groupby("b").indices.items():
            e = ev.iloc[idx]
            l = lab[lb == b]
            tb = m0.zb(e.i.to_numpy()) + m0.sb(e.s.to_numpy())
            if len(l) == 0:
                out[idx] = gpirt.predictive(tb, np.full(len(e), m0.s_m ** 2 + 1.0), m0.var_scale) if (m0.zpath or m0.spath) else 0.5
                continue
            subj = pd.Index(np.unique(np.concatenate([e.s.to_numpy(), l.s.to_numpy()])))
            Litems = np.unique(l.i.to_numpy()); Lidx = pd.Index(Litems)
            ti, tinv = np.unique(e.i.to_numpy(), return_inverse=True)
            ls, ll, y = subj.get_indexer(l.s), Lidx.get_indexer(l.i), l.y.to_numpy(float)
            ts = subj.get_indexer(e.s)
            lbase = m0.zb(l.i.to_numpy()) + m0.sb(l.s.to_numpy())
            preds, les = [], []
            for m in self.members:
                K_LL, K_tL, k_tt = m.kernels(Litems, ti)
                for sa in m.s_a_grid:
                    for sd in m.s_d_grid:
                        f = gpirt.fit(ls, ll, y, lbase, K_LL, len(subj), m.s_m, sa, sd)
                        mu, var = gpirt.predict_moments(f, ts, K_tL[tinv], k_tt[tinv], tb)
                        preds.append(gpirt.predictive(mu, var, m.var_scale)); les.append(f.log_evidence)
            les = np.array(les) * self.temp
            w = np.exp(les - les.max()); w /= w.sum()
            out[idx] = np.tensordot(w, np.stack(preds), 1)
        return out


class MCFactory:
    def __init__(self, cfgs, temp=1.0):
        self.cfgs, self.temp = cfgs, temp

    def __call__(self, f):
        return MultiCfgBMA(self.cfgs, self.temp)


class MSFactory:
    def __init__(self, embs, temp=1.0, mix=0.0, **kw):
        self.embs, self.temp, self.mix, self.kw = embs, temp, mix, kw

    def __call__(self, f):
        return MultiSourceBMA(self.embs, self.temp, self.mix, **self.kw)


class EnsFactory7:
    def __init__(self, configs, logit=False):
        self.configs, self.logit = configs, logit

    def __call__(self, f):
        return Ensemble([GPBMA7(**c) for c in self.configs], self.logit)


_Z = {}


class GPBMA7(GPBMA6):
    """GPBMA6 with an out-of-fold prior offset base = T * z_item (cross-benchmark accuracy regression)."""
    def __init__(self, zpath=None, T=0.4, zpath2=None, T2=0.0, spath=None, Ts=0.0, **kw):
        super().__init__(**kw)
        self.zpath, self.T, self.zpath2, self.T2 = zpath, T, zpath2, T2
        self.spath, self.Ts = spath, Ts

    def sb(self, subjects):
        if self.spath is None or self.Ts == 0:
            return np.zeros(len(subjects))
        if self.spath not in _Z:
            _Z[self.spath] = np.load(self.spath).astype(np.float64)
        return self.Ts * _Z[self.spath][subjects]

    def zb(self, items):
        out = np.zeros(len(items))
        for p, t in ((self.zpath, self.T), (self.zpath2, self.T2)):
            if p is None or t == 0:
                continue
            if p not in _Z:
                _Z[p] = np.load(p).astype(np.float64)
            out = out + t * _Z[p][items]
        return out

    def predict_batch(self, ev, lab):
        out = np.empty(len(ev))
        lb = lab.b.to_numpy() if len(lab) else np.zeros(0, int)
        lp = self.log_prior()
        for b, idx in ev.groupby("b").indices.items():
            e = ev.iloc[idx]
            l = lab[lb == b]
            tb = self.zb(e.i.to_numpy()) + self.sb(e.s.to_numpy())
            if len(l) == 0:
                out[idx] = gpirt.predictive(tb, np.full(len(e), self.s_m ** 2 + 1.0), self.var_scale) if (self.zpath or self.spath) else 0.5
                continue
            subj = pd.Index(np.unique(np.concatenate([e.s.to_numpy(), l.s.to_numpy()])))
            Litems = np.unique(l.i.to_numpy()); Lidx = pd.Index(Litems)
            ti, tinv = np.unique(e.i.to_numpy(), return_inverse=True)
            K_LL, K_tL, k_tt = self.kernels(Litems, ti)
            ls, ll, y = subj.get_indexer(l.s), Lidx.get_indexer(l.i), l.y.to_numpy(float)
            ts = subj.get_indexer(e.s)
            fits = [gpirt.fit(ls, ll, y, self.zb(l.i.to_numpy()) + self.sb(l.s.to_numpy()), K_LL, len(subj), self.s_m, sa, sd)
                    for sa in self.s_a_grid for sd in self.s_d_grid]
            w = gpirt.bma_weights(fits, lp)
            p = np.zeros(len(e)); tot = 0.0
            for wf, f in zip(w, fits):
                if wf < 1e-4:
                    continue
                mu, var = gpirt.predict_moments(f, ts, K_tL[tinv], k_tt[tinv], tb)
                p += wf * gpirt.predictive(mu, var, self.var_scale); tot += wf
            out[idx] = p / tot
        return out


class GPBMA8(GPBMA7):
    """GPBMA7 + (i) benchmark-level prior from the mean Qwen score of labeled items,
    (ii) label-count-dependent variance scale, (iii) BMA over a grid of benchmark-shift scales s_m.

    base_i = T1 * zbar + T2 * (z_i - zbar), zbar = mean z over labeled items of the benchmark
    (at budget 0 zbar = z_i). var_scale = vs_lo if the target subject has <= n_lo labels else vs_hi.
    """
    def __init__(self, T1=None, T2=None, vs_lo=None, vs_hi=None, n_lo=3, s_m_grid=None, sm_logprior=None,
                 sm_min_lab=0, sm_max_items=None, **kw):
        super().__init__(**kw)
        self.T1 = self.T if T1 is None else T1
        self.T2 = self.T if T2 is None else T2
        self.vs_lo = self.var_scale if vs_lo is None else vs_lo
        self.vs_hi = self.var_scale if vs_hi is None else vs_hi
        self.n_lo = n_lo
        self.s_m_grid = tuple(s_m_grid) if s_m_grid else (self.s_m,)
        self.sm_logprior = np.zeros(len(self.s_m_grid)) if sm_logprior is None else np.asarray(sm_logprior, float)
        self.sm_min_lab, self.sm_max_items = sm_min_lab, sm_max_items

    def zraw(self, items):
        if self.zpath is None:
            return np.zeros(len(items))
        if self.zpath not in _Z:
            _Z[self.zpath] = np.load(self.zpath).astype(np.float64)
        return _Z[self.zpath][items]

    def predict_batch(self, ev, lab):
        out = np.empty(len(ev))
        lb = lab.b.to_numpy() if len(lab) else np.zeros(0, int)
        for b, idx in ev.groupby("b").indices.items():
            e = ev.iloc[idx]
            l = lab[lb == b]
            zt = self.zraw(e.i.to_numpy())
            if len(l) == 0:
                tb = self.T1 * zt + self.sb(e.s.to_numpy())
                out[idx] = gpirt.predictive(tb, np.full(len(e), self.s_m ** 2 + 1.0), self.vs_lo) if (self.zpath or self.spath) else 0.5
                continue
            zl = self.zraw(l.i.to_numpy())
            zbar = self.zraw(np.unique(l.i.to_numpy())).mean() if self.zpath else 0.0
            tb = self.T1 * zbar + self.T2 * (zt - zbar) + self.sb(e.s.to_numpy())
            lbase = self.T1 * zbar + self.T2 * (zl - zbar) + self.sb(l.s.to_numpy())
            subj = pd.Index(np.unique(np.concatenate([e.s.to_numpy(), l.s.to_numpy()])))
            Litems = np.unique(l.i.to_numpy()); Lidx = pd.Index(Litems)
            ti, tinv = np.unique(e.i.to_numpy(), return_inverse=True)
            K_LL, K_tL, k_tt = self.kernels(Litems, ti)
            ls, ll, y = subj.get_indexer(l.s), Lidx.get_indexer(l.i), l.y.to_numpy(float)
            ts = subj.get_indexer(e.s)
            nlab = np.bincount(ls, minlength=len(subj))[ts]
            vs = np.where(nlab <= self.n_lo, self.vs_lo, self.vs_hi)
            fits, lps = [], []
            use_grid = len(l) >= self.sm_min_lab and (self.sm_max_items is None or len(ti) <= self.sm_max_items)
            for k, sm in enumerate(self.s_m_grid if use_grid else (self.s_m,)):
                for sa in self.s_a_grid:
                    for sd in self.s_d_grid:
                        fits.append(gpirt.fit(ls, ll, y, lbase, K_LL, len(subj), sm, sa, sd)); lps.append(self.sm_logprior[k] if use_grid else 0.0)
            w = gpirt.bma_weights(fits, np.array(lps))
            p = np.zeros(len(e)); tot = 0.0
            for wf, f in zip(w, fits):
                if wf < 1e-4:
                    continue
                mu, var = gpirt.predict_moments(f, ts, K_tL[tinv], k_tt[tinv], tb)
                p += wf * gpirt.predictive(mu, var, vs); tot += wf
            out[idx] = p / tot
        return out


class EnsFactory8:
    def __init__(self, configs):
        self.configs = configs

    def __call__(self, f):
        return Ensemble([GPBMA8(**c) for c in self.configs])


_ML = {}


class GPBMA9(GPBMA8):
    """GPBMA8 whose linear kernel mixes the isotropic centred-embedding kernel with a meta-learned
    low-rank metric: lin = (1-alpha) * Zc Zc^T / q1 + alpha * (Zc M)(Zc M)^T / q2 (M per CV fold)."""
    def __init__(self, fold=-1, metalin=None, alpha=0.5, **kw):
        super().__init__(**kw)
        self.fold, self.metalin, self.alpha = fold, metalin, alpha

    def kernels(self, Litems, ti):
        K_LL, K_tL, k_tt = GPBMA2.kernels(self, Litems, ti)
        if len(Litems) < 3 or self.rho_le <= 0:
            return K_LL, K_tL, k_tt
        E = load_emb(self.emb)
        mu = E[Litems].mean(0)
        ZL, ZT = E[Litems] - mu, E[ti] - mu
        q = max(1e-9, float((ZL ** 2).sum(1).mean()))
        LL, TL, tt = ZL @ ZL.T / q, ZT @ ZL.T / q, (ZT ** 2).sum(1) / q
        if self.metalin and self.alpha > 0:
            if self.metalin not in _ML:
                _ML[self.metalin] = np.load(self.metalin, allow_pickle=True).item()
            M = _ML[self.metalin][self.fold]["M"]
            AL, AT = ZL @ M, ZT @ M
            q2 = max(1e-9, float((AL ** 2).sum(1).mean()))
            LL = (1 - self.alpha) * LL + self.alpha * AL @ AL.T / q2
            TL = (1 - self.alpha) * TL + self.alpha * AT @ AL.T / q2
            tt = (1 - self.alpha) * tt + self.alpha * (AT ** 2).sum(1) / q2
        s = 1 - self.rho_le
        return s * K_LL + self.rho_le * LL, s * K_tL + self.rho_le * TL, s * k_tt + self.rho_le * tt


class EnsFactory9:
    def __init__(self, configs):
        self.configs = configs

    def __call__(self, f):
        return Ensemble([GPBMA9(fold=f, **c) if c.get('metalin') else GPBMA8(**{k: v for k, v in c.items() if k not in ('metalin', 'alpha')})
                         for c in self.configs])


class GPBMA10(GPBMA8):
    """GPBMA8 + linear kernel on numeric item metadata standardized within the benchmark (weight rho_n)."""
    def __init__(self, rho_n=0.1, numfeat="data/proc/item_numfeat.pkl", **kw):
        super().__init__(**kw)
        self.rho_n, self.numfeat = rho_n, numfeat

    def kernels(self, Litems, ti):
        K_LL, K_tL, k_tt = super().kernels(Litems, ti)
        if self.rho_n <= 0 or len(Litems) < 3:
            return K_LL, K_tL, k_tt
        nf = load_tokens(self.numfeat)
        XL, XT = TK.numeric_matrix([nf[i] for i in Litems], [nf[i] for i in ti])
        if XL.shape[1] == 0:
            return K_LL, K_tL, k_tt
        q = XL.shape[1]
        s = 1 - self.rho_n
        return (s * K_LL + self.rho_n * XL @ XL.T / q, s * K_tL + self.rho_n * XT @ XL.T / q,
                s * k_tt + self.rho_n * (XT ** 2).sum(1) / q)


class EnsFactory10:
    def __init__(self, configs):
        self.configs = configs

    def __call__(self, f):
        return Ensemble([GPBMA10(**c) for c in self.configs])


class GPBMA11(GPBMA8):
    """GPBMA8 with an empirical prior on the benchmark shift m: a mixture of Gaussians
    m ~ sum_k pi_k N(mu_k, s_k^2) learned from training groups (m_mix = [(mu, s, pi), ...]).
    Each component is one more axis of the BMA grid, with log pi_k as its log prior weight."""
    def __init__(self, m_mix=((0.0, 1.5, 1.0),), fold=-1, grid_lp=None, record=None, adapt_tau=None,
                 adapt_s=1.8, **kw):
        super().__init__(**kw)
        self.m_mix = [tuple(c) for c in m_mix]
        # adapt_tau: the centre of the shift prior is itself uncertain, N(mu0, tau^2), and is updated from the
        # label rates of the OTHER benchmarks of the episode (each a noisy view with between-benchmark sd adapt_s)
        self.adapt_tau, self.adapt_s = adapt_tau, adapt_s
        # grid_lp: path to {fold: log prior over the (component, s_a, s_d) grid}, learned on the other folds
        self.fold, self.grid_lp, self.record = fold, grid_lp, record
        self.glp = None
        if grid_lp:
            if grid_lp not in _ML:
                _ML[grid_lp] = np.load(grid_lp, allow_pickle=True).item()
            self.glp = np.asarray(_ML[grid_lp][fold], float)
        self.recorded = []

    def predict_batch(self, ev, lab):
        out = np.empty(len(ev))
        lb = lab.b.to_numpy() if len(lab) else np.zeros(0, int)
        pis = np.array([c[2] for c in self.m_mix], float); pis = pis / pis.sum()
        bstats = None
        if self.adapt_tau and len(lab):
            st = lab.groupby("b").y.agg(["sum", "size"])
            bstats = (st.index.to_numpy(), st["sum"].to_numpy(float), st["size"].to_numpy(float))
        mix_all = self.m_mix
        for b, idx in ev.groupby("b").indices.items():
            e = ev.iloc[idx]
            l = lab[lb == b]
            zt = self.zraw(e.i.to_numpy())
            self.m_mix = mix_all if bstats is None else self._adapted(mix_all, bstats, b)
            if len(l) == 0:
                tb = self.T1 * zt + self.sb(e.s.to_numpy())
                out[idx] = sum(pk * gpirt.predictive(tb + mu, np.full(len(e), sm ** 2 + 1.0), self.vs_lo)
                               for (mu, sm, _), pk in zip(self.m_mix, pis))
                continue
            zl = self.zraw(l.i.to_numpy())
            zbar = self.zraw(np.unique(l.i.to_numpy())).mean() if self.zpath else 0.0
            tb = self.T1 * zbar + self.T2 * (zt - zbar) + self.sb(e.s.to_numpy())
            lbase = self.T1 * zbar + self.T2 * (zl - zbar) + self.sb(l.s.to_numpy())
            subj = pd.Index(np.unique(np.concatenate([e.s.to_numpy(), l.s.to_numpy()])))
            Litems = np.unique(l.i.to_numpy()); Lidx = pd.Index(Litems)
            ti, tinv = np.unique(e.i.to_numpy(), return_inverse=True)
            K_LL, K_tL, k_tt = self.kernels(Litems, ti)
            ls, ll, y = subj.get_indexer(l.s), Lidx.get_indexer(l.i), l.y.to_numpy(float)
            ts = subj.get_indexer(e.s)
            nlab = np.bincount(ls, minlength=len(subj))[ts]
            vs = np.where(nlab <= self.n_lo, self.vs_lo, self.vs_hi)
            fits, lps, mus = [], [], []
            for (mu, sm, _), pk in zip(self.m_mix, pis):
                for sa in self.s_a_grid:
                    for sd in self.s_d_grid:
                        fits.append(gpirt.fit(ls, ll, y, lbase + mu, K_LL, len(subj), sm, sa, sd))
                        lps.append(np.log(pk)); mus.append(mu)
            lps = np.array(lps) + (self.glp if self.glp is not None else 0.0)
            w = gpirt.bma_weights(fits, lps)
            if self.record is not None and len(l) >= self.record:
                self.recorded.append((int(b), w))
            p = np.zeros(len(e)); tot = 0.0
            for wf, f, mu in zip(w, fits, mus):
                if wf < 1e-4:
                    continue
                m_, var = gpirt.predict_moments(f, ts, K_tL[tinv], k_tt[tinv], tb + mu)
                p += wf * gpirt.predictive(m_, var, vs); tot += wf
            out[idx] = p / tot
        self.m_mix = mix_all
        return out

    def _adapted(self, mix, bstats, b):
        """Shift every component centre by the posterior move of the shared centre, from other benchmarks."""
        bs, k, n = bstats
        keep = bs != b
        if not keep.any():
            return mix
        k, n = k[keep], n[keep]
        p = (k + 0.5) / (n + 1.0)
        lg = np.log(p / (1 - p))
        v = self.adapt_s ** 2 + 1.0 / ((n + 1.0) * p * (1 - p))
        mu0 = mix[0][0]
        prec = 1.0 / self.adapt_tau ** 2 + (1.0 / v).sum()
        mu = (mu0 / self.adapt_tau ** 2 + (lg / v).sum()) / prec
        return [(c[0] + mu - mu0, c[1], c[2]) for c in mix]


class EnsFactory11:
    def __init__(self, configs):
        self.configs = configs

    def __call__(self, f):
        return Ensemble([GPBMA11(fold=f, **c) for c in self.configs])


class BayesOwn:
    """Reference: Bayes posterior mean of the group accuracy from the group's own labels only, with the
    accuracies of training groups (80-200 items, other folds, each benchmark weighted equally) as the prior."""
    def __init__(self, fold=-1, path="data/group_acc.csv", max_items=200):
        g = pd.read_csv(path)
        g = g[(g.n_items <= max_items) & (g.fold != fold)]
        self.q = g.q.clip(1e-4, 1 - 1e-4).to_numpy()
        w = 1.0 / g.groupby("b").q.transform("size").to_numpy()
        self.w = w / w.sum()

    def predict_batch(self, ev, lab):
        out = np.empty(len(ev))
        k = lab.groupby("g").y.sum() if len(lab) else pd.Series(dtype=float)
        n = lab.groupby("g").y.size() if len(lab) else pd.Series(dtype=float)
        lq, l1q = np.log(self.q), np.log1p(-self.q)
        for g, idx in ev.groupby("g").indices.items():
            kk, nn = float(k.get(g, 0)), float(n.get(g, 0))
            ll = kk * lq + (nn - kk) * l1q
            post = self.w * np.exp(ll - ll.max())
            out[idx] = (post * self.q).sum() / post.sum()
        return out

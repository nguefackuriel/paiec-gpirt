"""GP-IRT: online hierarchical logistic model with a GP prior on item effects.

logit P(y=1 | s, t) = base + m + a_s + phi_s . beta + d_t
  m    ~ N(0, s_m^2)            benchmark-level shift
  a_s  ~ N(mu_a[s], s_a^2)      subject shift within the benchmark
  beta ~ N(0, s_b^2 I)          effects of subject feature tokens phi_s (optional)
  d    ~ GP(0, s_d^2 K)         item effects; K from text/feature similarity

`fit` uses only revealed labels (labeled items L), so it can be cached and
reused for every target. `predict_moments` handles any target item through the
GP conditional d_t | d_L. Laplace approximation; the diagonal subject block is
eliminated with a Schur complement onto the c = (m, beta, d_L) block, so the
cost is O(L * r + n_s * n_c + n_c^3). `log_evidence` is the Laplace marginal
likelihood, used for per-benchmark hyperparameter averaging. numpy only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def predictive(mean, var, var_scale=1.0):
    return sigmoid(mean / np.sqrt(1.0 + np.pi * var_scale * np.maximum(var, 0.0) / 8.0))


@dataclass
class Fit:
    s_m: float
    s_a: float
    s_d: float
    s_b: float = 1.0
    r: int = 0
    m: float = 0.0
    a: np.ndarray = None
    beta: np.ndarray = None
    d: np.ndarray = None
    mu_a: np.ndarray = None
    Phi: np.ndarray = None
    Kinv: np.ndarray = None
    Q: np.ndarray = None
    M: np.ndarray = None
    var_a: np.ndarray = None
    log_evidence: float = 0.0
    empty: bool = False


def fit(lab_s, lab_l, y, base, K_LL, n_s, s_m=1.5, s_a=1.0, s_d=1.0, mu_a=None,
        iters=30, tol=1e-6, Phi=None, s_b=1.0) -> Fit:
    lab_s = np.asarray(lab_s, int); lab_l = np.asarray(lab_l, int)
    y = np.asarray(y, float); base = np.asarray(base, float)
    mu_a = np.zeros(n_s) if mu_a is None else np.asarray(mu_a, float)
    Phi = np.zeros((n_s, 0)) if Phi is None else np.asarray(Phi, float)
    r = Phi.shape[1]
    n_L = K_LL.shape[0]
    f = Fit(s_m, s_a, s_d, s_b, r, mu_a=mu_a, Phi=Phi)
    if len(y) == 0 or n_L == 0:
        f.empty = True
        f.a = mu_a.copy(); f.var_a = np.full(n_s, s_a ** 2)
        return f
    pa = np.full(n_s, 1.0 / s_a ** 2); pm = 1.0 / s_m ** 2; pb = 1.0 / s_b ** 2 if r else 1.0
    cL = np.linalg.cholesky(K_LL + 1e-6 * np.eye(n_L))
    cinv = np.linalg.solve(cL, np.eye(n_L))
    Kinv = cinv.T @ cinv
    Pd = Kinv / s_d ** 2
    o_d = 1 + r                     # offset of item effects in the c block
    nc = o_d + n_L
    Pc = np.zeros((nc, nc)); Pc[0, 0] = pm
    Pc[np.arange(1, o_d), np.arange(1, o_d)] = pb
    Pc[o_d:, o_d:] = Pd
    a = mu_a.copy(); c = np.zeros(nc)
    PhiL = Phi[lab_s]                # L x r
    rows = np.arange(len(y))

    def eta_of(a, c):
        return base + c[0] + a[lab_s] + PhiL @ c[1:o_d] + c[o_d + lab_l]

    def blocks(w):
        D = np.bincount(lab_s, w, n_s) + pa
        # dense C = sum_l w x_c x_c^T with x_c = [1, phi_s, e_l]
        C = Pc.copy()
        wl = np.bincount(lab_l, w, n_L)
        C[0, 0] += w.sum()
        C[0, o_d:] += wl; C[o_d:, 0] += wl
        C[np.arange(o_d, nc), np.arange(o_d, nc)] += wl
        if r:
            wPhi = PhiL * w[:, None]
            C[1:o_d, 1:o_d] += PhiL.T @ wPhi
            sb = wPhi.sum(0); C[0, 1:o_d] += sb; C[1:o_d, 0] += sb
            Bdl = np.zeros((n_L, r)); np.add.at(Bdl, lab_l, wPhi)
            C[o_d:, 1:o_d] += Bdl; C[1:o_d, o_d:] += Bdl.T
        B = np.zeros((n_s, nc))
        B[:, 0] = np.bincount(lab_s, w, n_s)
        if r:
            ws = np.bincount(lab_s, w, n_s)
            B[:, 1:o_d] = Phi * ws[:, None]
        np.add.at(B, (lab_s, o_d + lab_l), w)
        return D, C, B

    for _ in range(iters):
        p = sigmoid(eta_of(a, c)); w = p * (1 - p); res = y - p
        ga = np.bincount(lab_s, res, n_s) - pa * (a - mu_a)
        gc = -Pc @ c
        gc[0] += res.sum()
        if r:
            gc[1:o_d] += PhiL.T @ res
        gc[o_d:] += np.bincount(lab_l, res, n_L)
        D, C, B = blocks(w)
        BD = B / D[:, None]
        dc = np.linalg.solve(C - B.T @ BD, gc - BD.T @ ga)
        da = (ga - B @ dc) / D
        c += dc; a += da
        if max(np.abs(dc).max(), np.abs(da).max()) < tol:
            break
    eta = eta_of(a, c)
    p = sigmoid(eta); w = p * (1 - p)
    D, C, B = blocks(w)
    BD = B / D[:, None]
    S = C - B.T @ BD
    Q = np.linalg.inv(S)
    M = BD @ Q
    f.m, f.a, f.beta, f.d = c[0], a, c[1:o_d], c[o_d:]
    f.Kinv, f.Q, f.M = Kinv, Q, M
    f.var_a = 1.0 / D + np.einsum("sc,sc->s", M, BD)
    loglik = np.sum(y * eta - np.logaddexp(0, eta))
    quad = c @ Pc @ c + np.sum(pa * (a - mu_a) ** 2)
    logdet_prior = (-np.log(pm) - r * np.log(pb) - np.sum(np.log(pa))
                    + 2 * np.sum(np.log(np.diag(cL))) + n_L * np.log(s_d ** 2))
    logdet_H = np.sum(np.log(D)) + np.linalg.slogdet(S)[1]
    f.log_evidence = float(loglik - 0.5 * quad - 0.5 * logdet_prior - 0.5 * logdet_H)
    return f


def predict_moments(f: Fit, tgt_s, K_tL, k_tt, tgt_base, Phi_t=None):
    """Mean and variance of the target logits.

    tgt_s indexes subjects of the fit; Phi_t (n_t x r) gives the targets'
    subject features (defaults to f.Phi[tgt_s]).
    """
    tgt_s = np.asarray(tgt_s, int)
    k_tt = np.asarray(k_tt, float)
    if f.empty:
        mean = tgt_base + f.mu_a[tgt_s]
        var = f.s_m ** 2 + f.s_a ** 2 + f.s_d ** 2 * k_tt
        if f.r:
            Pt = f.Phi[tgt_s] if Phi_t is None else Phi_t
            var = var + f.s_b ** 2 * (Pt ** 2).sum(1)
        return mean, var
    r = f.r; o_d = 1 + r
    W = K_tL @ f.Kinv
    var_eps = f.s_d ** 2 * np.maximum(k_tt - np.einsum("tl,tl->t", W, K_tL), 0)
    # u_t = [1, phi_t, W_t] in c-space
    Pt = (f.Phi[tgt_s] if Phi_t is None else Phi_t) if r else np.zeros((len(tgt_s), 0))
    U = np.concatenate([np.ones((len(tgt_s), 1)), Pt, W], 1)
    Q = f.Q
    var_x = np.einsum("tc,tc->t", U @ Q, U)
    cov_ax = -np.einsum("tc,tc->t", f.M[tgt_s], U)
    var = f.var_a[tgt_s] + var_x + 2 * cov_ax + var_eps
    mean = tgt_base + f.m + f.a[tgt_s] + (Pt @ f.beta if r else 0.0) + W @ f.d
    return mean, var


def bma_weights(fits, log_prior=None):
    le = np.array([f.log_evidence for f in fits])
    if log_prior is not None:
        le = le + np.asarray(log_prior)
    w = np.exp(le - le.max())
    return w / w.sum()

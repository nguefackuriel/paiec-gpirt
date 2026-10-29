"""Online hierarchical logistic (IRT) model fit on revealed labels.

logit P(y=1 | s, i) = base[s,i] + m + a_s + d_i
  base: fixed prior offset (e.g. from an offline feature model), default 0
  m   ~ N(0, s_m^2)  benchmark-level shift
  a_s ~ N(0, s_a^2)  subject (group) shift, optionally with per-subject prior mean
  d_i ~ N(0, s_d^2)  item shift (shared across subjects of the benchmark)

MAP by Newton's method. The subject block of the Hessian is diagonal, so it is
eliminated with a Schur complement onto the (m, items) block, whose size is at
most ~1000. The Laplace posterior gives the predictive variance of each
target's logit, and the probit approximation gives its predictive mean.
Only numpy is required.
"""
from __future__ import annotations

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def predictive(mean, var):
    return sigmoid(mean / np.sqrt(1.0 + np.pi * np.maximum(var, 0.0) / 8.0))


def fit_predict(lab_s, lab_i, lab_y, lab_base, tgt_s, tgt_i, tgt_base, n_s, n_i,
                s_m=1.5, s_a=1.0, s_d=1.0, iters=25, tol=1e-6, return_state=False):
    """Fit on labels of one benchmark; return predictive P(correct) for targets.

    Subjects/items are integer coded in [0, n_s) and [0, n_i). s_a and s_d may be
    scalars or per-subject / per-item arrays.
    """
    lab_s = np.asarray(lab_s, int); lab_i = np.asarray(lab_i, int)
    y = np.asarray(lab_y, float); base = np.asarray(lab_base, float)
    tgt_s = np.asarray(tgt_s, int); tgt_i = np.asarray(tgt_i, int)
    pa = 1.0 / np.broadcast_to(np.asarray(s_a, float), (n_s,)) ** 2
    pd_ = 1.0 / np.broadcast_to(np.asarray(s_d, float), (n_i,)) ** 2
    pm = 1.0 / s_m ** 2
    L = len(y)
    if L == 0:
        var = s_m ** 2 + (1 / pa)[tgt_s] + (1 / pd_)[tgt_i]
        return predictive(tgt_base, var)

    a = np.zeros(n_s); d = np.zeros(n_i); m = 0.0
    nc = n_i + 1  # C block: index 0 = m, 1..n_i = items
    ci = 1 + lab_i
    for _ in range(iters):
        eta = base + m + a[lab_s] + d[lab_i]
        p = sigmoid(eta)
        w = p * (1 - p)
        r = y - p
        # gradients of the log posterior
        ga = np.bincount(lab_s, r, n_s) - pa * a
        gc = np.empty(nc)
        gc[0] = r.sum() - pm * m
        gc[1:] = np.bincount(lab_i, r, n_i) - pd_ * d
        # Hessian blocks (negative log posterior): D (subjects, diagonal), B (subj x C), C
        D = np.bincount(lab_s, w, n_s) + pa
        C = np.zeros((nc, nc))
        C[0, 0] = w.sum() + pm
        wi = np.bincount(lab_i, w, n_i)
        C[0, 1:] = wi; C[1:, 0] = wi
        C[np.arange(1, nc), np.arange(1, nc)] += wi + pd_
        # B[s, c] = sum of w over labels with subject s and C-index c (plus m column)
        Bm = np.bincount(lab_s, w, n_s)  # column 0 (m)
        Bi = np.zeros((n_s, n_i))
        np.add.at(Bi, (lab_s, lab_i), w)
        B = np.concatenate([Bm[:, None], Bi], 1)
        BD = B / D[:, None]
        S = C - B.T @ BD
        rhs = gc - BD.T @ ga
        dc = np.linalg.solve(S, rhs)
        da = (ga - B @ dc) / D
        m += dc[0]; d += dc[1:]; a += da
        if max(np.abs(dc).max(), np.abs(da).max()) < tol:
            break
    # posterior covariance pieces at the mode
    eta = base + m + a[lab_s] + d[lab_i]
    p = sigmoid(eta); w = p * (1 - p)
    D = np.bincount(lab_s, w, n_s) + pa
    C = np.zeros((nc, nc)); C[0, 0] = w.sum() + pm
    wi = np.bincount(lab_i, w, n_i); C[0, 1:] = wi; C[1:, 0] = wi
    C[np.arange(1, nc), np.arange(1, nc)] += wi + pd_
    Bi = np.zeros((n_s, n_i)); np.add.at(Bi, (lab_s, lab_i), w)
    B = np.concatenate([np.bincount(lab_s, w, n_s)[:, None], Bi], 1)
    BD = B / D[:, None]
    Q = np.linalg.inv(C - B.T @ BD)          # Cov of C block
    M = BD @ Q                               # n_s x nc
    var_a = 1.0 / D + np.einsum("sc,sc->s", M, BD)
    # x = m + d_i  -> u = e_0 + e_{1+i}
    ti = 1 + tgt_i
    var_x = Q[0, 0] + Q[ti, ti] + 2 * Q[0, ti]
    cov_ax = -(M[tgt_s, 0] + M[tgt_s, ti])
    var_t = var_a[tgt_s] + var_x + 2 * cov_ax
    mean_t = tgt_base + m + a[tgt_s] + d[tgt_i]
    out = predictive(mean_t, var_t)
    if return_state:
        return out, dict(m=m, a=a, d=d, var_t=var_t, mean_t=mean_t)
    return out


def fit_predict_gp(lab_s, lab_l, lab_y, lab_base, K_LL, tgt_s, K_tL, k_tt, tgt_base, n_s,
                   s_m=1.5, s_a=1.0, s_d=1.0, mu_a=None, iters=25, tol=1e-6):
    """IRT with a Gaussian-process prior on item effects.

    Labeled items are indexed 0..n_L-1 (lab_l). Item effects d_L ~ N(0, s_d^2 K_LL).
    A target item t (labeled or not) has d_t = W_t d_L + eps, W = K_tL K_LL^-1,
    Var(eps) = s_d^2 (k_tt - W_t K_tL^T). Subjects 0..n_s-1, a ~ N(mu_a, s_a^2).
    The posterior over labeled-item effects is independent of the targets, so
    the fit can be cached and reused for every target input.
    """
    lab_s = np.asarray(lab_s, int); lab_l = np.asarray(lab_l, int)
    y = np.asarray(lab_y, float); base = np.asarray(lab_base, float)
    tgt_s = np.asarray(tgt_s, int)
    mu_a = np.zeros(n_s) if mu_a is None else np.asarray(mu_a, float)
    pa = np.full(n_s, 1.0 / s_a ** 2)
    pm = 1.0 / s_m ** 2
    n_L = K_LL.shape[0]
    if len(y) == 0 or n_L == 0:
        var = s_m ** 2 + s_a ** 2 + s_d ** 2 * np.asarray(k_tt, float)
        return predictive(tgt_base + mu_a[tgt_s], var)
    cL = np.linalg.cholesky(K_LL + 1e-8 * np.eye(n_L))
    Kinv = np.linalg.solve(cL.T, np.linalg.solve(cL, np.eye(n_L)))
    Pd = Kinv / s_d ** 2
    nc = n_L + 1
    a = mu_a.copy(); d = np.zeros(n_L); m = 0.0

    def blocks(w):
        D = np.bincount(lab_s, w, n_s) + pa
        C = np.zeros((nc, nc)); C[0, 0] = w.sum() + pm
        wl = np.bincount(lab_l, w, n_L); C[0, 1:] = wl; C[1:, 0] = wl
        C[1:, 1:] = Pd; C[np.arange(1, nc), np.arange(1, nc)] += wl
        Bi = np.zeros((n_s, n_L)); np.add.at(Bi, (lab_s, lab_l), w)
        B = np.concatenate([np.bincount(lab_s, w, n_s)[:, None], Bi], 1)
        return D, C, B

    for _ in range(iters):
        p = sigmoid(base + m + a[lab_s] + d[lab_l]); w = p * (1 - p); r = y - p
        ga = np.bincount(lab_s, r, n_s) - pa * (a - mu_a)
        gc = np.empty(nc); gc[0] = r.sum() - pm * m; gc[1:] = np.bincount(lab_l, r, n_L) - Pd @ d
        D, C, B = blocks(w)
        BD = B / D[:, None]
        dc = np.linalg.solve(C - B.T @ BD, gc - BD.T @ ga)
        da = (ga - B @ dc) / D
        m += dc[0]; d += dc[1:]; a += da
        if max(np.abs(dc).max(), np.abs(da).max()) < tol:
            break
    p = sigmoid(base + m + a[lab_s] + d[lab_l]); w = p * (1 - p)
    D, C, B = blocks(w)
    BD = B / D[:, None]
    Q = np.linalg.inv(C - B.T @ BD)
    M = BD @ Q
    var_a = 1.0 / D + np.einsum("sc,sc->s", M, BD)
    W = K_tL @ Kinv                                  # n_t x n_L
    var_eps = s_d ** 2 * np.maximum(np.asarray(k_tt, float) - np.einsum("tl,tl->t", W, K_tL), 0)
    Qdd = Q[1:, 1:]
    var_x = Q[0, 0] + 2 * W @ Q[0, 1:] + np.einsum("tl,tl->t", W @ Qdd, W)
    cov_ax = -(M[tgt_s, 0] + np.einsum("tl,tl->t", M[tgt_s, 1:], W))
    var_t = var_a[tgt_s] + var_x + 2 * cov_ax + var_eps
    mean_t = tgt_base + m + a[tgt_s] + W @ d
    return predictive(mean_t, var_t)

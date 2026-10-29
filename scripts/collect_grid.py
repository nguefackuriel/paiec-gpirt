"""Empirical-Bayes prior over the BMA grid: average posterior grid weights on training benchmarks.

For every simulated episode, fit the e5 member at budget 31 and keep the normalised BMA weights of each
benchmark (labels from all its groups). The prior for held-out fold f is the benchmark-averaged weight
over the other folds, mixed with the uniform prior: lp = log((1-eps) * wbar + eps / n_grid).
Usage: PAIEC_MAX_GROUP=200 python scripts/collect_grid.py <n_per_fold> <out.npy> [eps]
"""
import os, sys; sys.path.insert(0, '.')
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from scripts.cv import load, episodes
from paiec import sim, models
load()
N = int(sys.argv[1]); OUT = sys.argv[2]
EPS = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
W = int(os.environ.get('W', 26))
Z = 'data/proc/emb_qwen7b_L21_z_b0z.npy'
MIX = [(-0.6, 1.0, 1), (-0.6, 2.0, 1), (-0.6, 4.0, 1)]
cfg = dict(s_m=1.5, rho=0.73, rho_f=0.17, ell=0.12, var_scale=0.35, s_a_grid=(0.4, 0.8, 1.6), s_d_grid=(0.8, 1.6, 3.2),
           rho_l=0.0, rho_le=0.4, zpath=Z, T=0.3, vs_lo=0.35, vs_hi=0.2, m_mix=MIX)


def one(a):
    f, s, ch = a
    c = load()
    ev, aq = sim.build_episode(c['o'], ch, np.random.default_rng(s))
    m = models.GPBMA11(record=1, **cfg)
    m.predict_batch(ev, aq[aq['rank'] < 31])
    return [(f, b, w) for b, w in m.recorded]


with ProcessPoolExecutor(W) as ex:
    rec = [r for rs in ex.map(one, episodes(N)) for r in rs]
folds = np.array([r[0] for r in rec]); bench = np.array([r[1] for r in rec]); Wt = np.stack([r[2] for r in rec])
out = {}
for f in range(5):
    m = folds != f
    # benchmark-averaged: each benchmark gets equal weight
    wb = np.stack([Wt[m & (bench == b)].mean(0) for b in np.unique(bench[m])])
    wbar = wb.mean(0)
    out[f] = np.log((1 - EPS) * wbar + EPS / len(wbar))
    print('fold', f, 'benches', len(wb), 'top grid weights', np.round(np.sort(wbar)[::-1][:5], 3), flush=True)
np.save(OUT, out)
print('grid order: (component, s_a, s_d) nested;', np.round(wbar.reshape(len(MIX), 3, 3), 3))

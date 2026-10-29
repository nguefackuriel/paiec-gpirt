"""Refine the empirical prior on the benchmark shift m (centre, scales, data-fitted mixtures, Qwen weight)."""
import os, sys; sys.path.insert(0, '.')
import numpy as np
from scripts.cv import evaluate, fmt, load
from paiec import models
load()
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
W = int(os.environ.get('W', 26))
ONLY = os.environ.get('ONLY')
Z = 'data/proc/emb_qwen7b_L21_z_b0z.npy'
Q = 'data/proc/emb_qwen7b_L21_z.npy'
v7 = dict(s_m=1.5, rho=0.73, rho_f=0.17, ell=0.12, var_scale=0.35, s_a_grid=(0.4, 0.8, 1.6), s_d_grid=(0.8, 1.6, 3.2),
          rho_l=0.0, rho_le=0.4, zpath=Z, T=0.3, s_m_grid=(1.0, 2.0, 4.0), vs_lo=0.35, vs_hi=0.2)


def ens11(mix, **kw):
    a = dict(v7, m_mix=mix, **kw)
    return models.EnsFactory11([a, dict(a, emb=Q)])


SC = lambda mu, sc=(1.0, 2.0, 4.0): [(mu, s, 1) for s in sc]
GMM3 = [(-1.75, 1.2, .45), (0.54, 1.3, .50), (-5.7, 1.0, .05)]
GMM2 = [(-2.13, 2.0, .37), (-0.02, 1.5, .63)]
rows = [('centre -0.6', SC(-0.6), {}),
        ('centre -0.6 sc(.8,1.6,3.2)', SC(-0.6, (0.8, 1.6, 3.2)), {}),
        ('centre -0.6 sc(1.5,3)', SC(-0.6, (1.5, 3.0)), {}),
        ('centre -0.6 vs_lo .25', SC(-0.6), dict(vs_lo=0.25)),
        ('centre -0.6 T .5', SC(-0.6), dict(T=0.5)),
        ('GMM3 (fit on train groups)', GMM3, {}),
        ('GMM2 (fit on train groups)', GMM2, {})]
if ONLY:
    rows = [r for r in rows if any(o in r[0] for o in ONLY.split(','))]
ref = evaluate(models.EnsFactory8([v7, dict(v7, emb=Q)]), n_per_fold=N, seed=SEED, workers=W)
print(f'{"v7":34s}', fmt(ref), flush=True)
for name, mix, kw in rows:
    df = evaluate(ens11(mix, **kw), n_per_fold=N, seed=SEED, workers=W)
    d = df.ALC.to_numpy() - ref.ALC.to_numpy()
    print(f'{name:34s}', fmt(df), f'dALC={d.mean():+.5f}±{d.std()/np.sqrt(len(d)):.5f}', flush=True)

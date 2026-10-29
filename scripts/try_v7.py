"""v7 candidates (realistic regime): benchmark-level Qwen prior, label-count variance scale, s_m mixture."""
import sys; sys.path.insert(0, '.')
import numpy as np
from scripts.cv import evaluate, fmt, load
from paiec import models
load()
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
W = 26
Z = 'data/proc/emb_qwen7b_L21_z_b0z.npy'
Q = 'data/proc/emb_qwen7b_L21_z.npy'
v6 = dict(s_m=1.5, rho=0.73, rho_f=0.17, ell=0.12, var_scale=0.35, s_a_grid=(0.4, 0.8, 1.6), s_d_grid=(0.8, 1.6, 3.2),
          rho_l=0.0, rho_le=0.4, zpath=Z, T=0.3)


def ens(**kw):
    a = {**v6, **kw}
    return models.EnsFactory8([a, {**a, 'emb': Q}])


ref = evaluate(ens(), n_per_fold=N, seed=SEED, workers=W)
print(f'{"v6 (GPBMA8 equiv)":34s}', fmt(ref), flush=True)
CANDS = [
    ('T1=.5 T2=.3', dict(T1=0.5, T2=0.3)),
    ('T1=.8 T2=.3', dict(T1=0.8, T2=0.3)),
    ('T1=.5 T2=.15', dict(T1=0.5, T2=0.15)),
    ('vs_lo=.5 vs_hi=.25', dict(vs_lo=0.5, vs_hi=0.25)),
    ('vs_lo=.25 vs_hi=.5', dict(vs_lo=0.25, vs_hi=0.5)),
    ('vs_lo=.35 vs_hi=.2', dict(vs_lo=0.35, vs_hi=0.2)),
    ('s_m grid (.8,1.5,3)', dict(s_m_grid=(0.8, 1.5, 3.0))),
    ('s_m grid (1,2,4)', dict(s_m_grid=(1.0, 2.0, 4.0))),
]
for name, kw in CANDS:
    df = evaluate(ens(**kw), n_per_fold=N, seed=SEED, workers=W)
    d = df.ALC.to_numpy() - ref.ALC.to_numpy()
    print(f'{name:34s}', fmt(df), f'dALC={d.mean():+.5f}±{d.std()/np.sqrt(len(d)):.5f}', flush=True)

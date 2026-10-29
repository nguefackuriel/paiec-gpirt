"""Combine the small v7 gains (heavy-tailed s_m mixture, label-count variance) on fresh episodes."""
import sys; sys.path.insert(0, '.')
import numpy as np
from scripts.cv import evaluate, fmt, load
from paiec import models
load()
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 1
W = 26
Z = 'data/proc/emb_qwen7b_L21_z_b0z.npy'
Q = 'data/proc/emb_qwen7b_L21_z.npy'
v6 = dict(s_m=1.5, rho=0.73, rho_f=0.17, ell=0.12, var_scale=0.35, s_a_grid=(0.4, 0.8, 1.6), s_d_grid=(0.8, 1.6, 3.2),
          rho_l=0.0, rho_le=0.4, zpath=Z, T=0.3)


def ens(**kw):
    a = {**v6, **kw}
    return models.EnsFactory8([a, {**a, 'emb': Q}])


ref = evaluate(ens(), n_per_fold=N, seed=SEED, workers=W)
print(f'{"v6":36s}', fmt(ref), flush=True)
for name, kw in [('sm(1,2,4)', dict(s_m_grid=(1.0, 2.0, 4.0))),
                 ('sm(1,2,4,8)', dict(s_m_grid=(1.0, 2.0, 4.0, 8.0))),
                 ('vs_hi .2', dict(vs_lo=0.35, vs_hi=0.2)),
                 ('sm(1,2,4) + vs_hi .2', dict(s_m_grid=(1.0, 2.0, 4.0), vs_lo=0.35, vs_hi=0.2)),
                 ('sm(1,2,4) + vs_hi .15', dict(s_m_grid=(1.0, 2.0, 4.0), vs_lo=0.35, vs_hi=0.15)),
                 ('sm(1,2,4) + vs_hi .2 n_lo 7', dict(s_m_grid=(1.0, 2.0, 4.0), vs_lo=0.35, vs_hi=0.2, n_lo=7))]:
    df = evaluate(ens(**kw), n_per_fold=N, seed=SEED, workers=W)
    d = df.ALC.to_numpy() - ref.ALC.to_numpy()
    print(f'{name:36s}', fmt(df), f'dALC={d.mean():+.5f}±{d.std()/np.sqrt(len(d)):.5f}', flush=True)

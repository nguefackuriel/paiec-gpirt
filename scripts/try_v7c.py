"""Regime-robust variants of the v7 s_m mixture (conditional on labels / group size, narrower grid)."""
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


V7 = dict(s_m_grid=(1.0, 2.0, 4.0), vs_lo=0.35, vs_hi=0.2)
ref = evaluate(ens(vs_lo=0.35, vs_hi=0.2), n_per_fold=N, seed=SEED, workers=W)
print(f'{"v6 + vs_hi .2":36s}', fmt(ref), flush=True)
for name, kw in [('v7', V7),
                 ('v7 grid if labels>=4', dict(V7, sm_min_lab=4)),
                 ('v7 grid if labels>=8', dict(V7, sm_min_lab=8)),
                 ('v7 grid if items<=200', dict(V7, sm_max_items=200)),
                 ('grid (1,1.5,2.5)', dict(V7, s_m_grid=(1.0, 1.5, 2.5))),
                 ('grid (1.5,3) prior (0,-1)', dict(V7, s_m_grid=(1.5, 3.0), sm_logprior=(0.0, -1.0)))]:
    df = evaluate(ens(**kw), n_per_fold=N, seed=SEED, workers=W)
    d = df.ALC.to_numpy() - ref.ALC.to_numpy()
    print(f'{name:36s}', fmt(df), f'dALC={d.mean():+.5f}±{d.std()/np.sqrt(len(d)):.5f}', flush=True)

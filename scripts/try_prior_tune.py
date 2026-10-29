"""Tune prior strength T and benchmark-shift scale s_m for the Qwen-prior ensemble (v4)."""
import sys; sys.path.insert(0, '.')
import numpy as np
from scripts.cv import evaluate, fmt, load
from paiec import models
load()
N, W = 100, 26
Z = 'data/proc/emb_qwen7b_L21_z_b0z.npy'
Q = 'data/proc/emb_qwen7b_L21_z.npy'


def ens(T, s_m, **kw):
    base = dict(s_m=s_m, rho=0.73, rho_f=0.17, ell=0.12, var_scale=0.5, s_d_grid=(0.8, 1.6, 3.2), rho_l=0.0, rho_le=0.5, **kw)
    return models.EnsFactory7([{**base, 'zpath': Z, 'T': T}, {**base, 'zpath': Z, 'T': T, 'emb': Q}])


ref = evaluate(ens(0.3, 1.0), n_per_fold=N, workers=W)
print(f'{"v4 T.3 sm1":24s}', fmt(ref), flush=True)
for T, s_m in [(0.3, 0.8), (0.3, 1.3), (0.45, 0.8), (0.45, 1.0), (0.2, 1.0)]:
    df = evaluate(ens(T, s_m), n_per_fold=N, workers=W)
    d = df.ALC.to_numpy() - ref.ALC.to_numpy()
    print(f'T={T} s_m={s_m:<14}', fmt(df), f'dALC={d.mean():+.5f}±{d.std()/np.sqrt(len(d)):.5f}', flush=True)

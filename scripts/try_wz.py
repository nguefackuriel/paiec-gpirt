"""Add a within-benchmark item-difficulty prior (ridge on centered Qwen-L14 states) to v4."""
import sys; sys.path.insert(0, '.')
import numpy as np
from scripts.cv import evaluate, fmt, load
from paiec import models
load()
N, W = 100, 26
Z = 'data/proc/emb_qwen7b_L21_z_b0z.npy'
Q = 'data/proc/emb_qwen7b_L21_z.npy'
WZ = 'data/proc/emb_qwen7b_L14_z_wz.npy'


def ens(T2):
    base = dict(s_m=1.0, rho=0.73, rho_f=0.17, ell=0.12, var_scale=0.5, s_d_grid=(0.8, 1.6, 3.2), rho_l=0.0, rho_le=0.5,
                zpath=Z, T=0.3, zpath2=WZ, T2=T2)
    return models.EnsFactory7([base, {**base, 'emb': Q}])


ref = evaluate(ens(0.0), n_per_fold=N, workers=W)
print(f'{"v4":12s}', fmt(ref), flush=True)
for T2 in [0.5, 1.0, 2.0]:
    df = evaluate(ens(T2), n_per_fold=N, workers=W)
    d = df.ALC.to_numpy() - ref.ALC.to_numpy()
    print(f'T2={T2:<8}', fmt(df), f'dALC={d.mean():+.5f}±{d.std()/np.sqrt(len(d)):.5f}', flush=True)

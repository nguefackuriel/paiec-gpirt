"""Cumulative gain: Qwen cross-benchmark prior + ensemble of e5-small / Qwen-L21 item kernels."""
import sys; sys.path.insert(0, '.')
import numpy as np
from scripts.cv import evaluate, fmt, load, Factory
from paiec import models
load()
N, W = 100, 26
base = dict(s_m=1.0, rho=0.73, rho_f=0.17, ell=0.12, var_scale=0.5, s_d_grid=(0.8, 1.6, 3.2), rho_l=0.0, rho_le=0.5)
Z = 'data/proc/emb_qwen7b_L21_z_b0z.npy'
Q = 'data/proc/emb_qwen7b_L21_z.npy'
ref = evaluate(Factory(models.GPBMA6, **base), n_per_fold=N, workers=W)
print(f'{"v3":26s}', fmt(ref), flush=True)
for name, mk in [
    ('prior T.3', Factory(models.GPBMA7, zpath=Z, T=0.3, **base)),
    ('prior T.3 + ens(s,q21)', models.EnsFactory7([{**base, 'zpath': Z, 'T': 0.3}, {**base, 'zpath': Z, 'T': 0.3, 'emb': Q}])),
    ('prior T.3 + ens logit', models.EnsFactory7([{**base, 'zpath': Z, 'T': 0.3}, {**base, 'zpath': Z, 'T': 0.3, 'emb': Q}], logit=True)),
    ('prior T.3 emb q21 only', Factory(models.GPBMA7, zpath=Z, T=0.3, **{**base, 'emb': Q})),
]:
    df = evaluate(mk, n_per_fold=N, workers=W)
    d = df.ALC.to_numpy() - ref.ALC.to_numpy()
    print(f'{name:26s}', fmt(df), f'dALC={d.mean():+.5f}±{d.std()/np.sqrt(len(d)):.5f}', flush=True)

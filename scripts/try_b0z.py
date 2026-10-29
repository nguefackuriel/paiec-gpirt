"""Does an out-of-fold cross-benchmark accuracy prior (ridge on item embeddings) help the online model?"""
import sys; sys.path.insert(0, '.')
import numpy as np
from scripts.cv import evaluate, fmt, load, Factory
from paiec import models
load()
N, W = 100, 26
base = dict(s_m=1.0, rho=0.73, rho_f=0.17, ell=0.12, var_scale=0.5, s_d_grid=(0.8, 1.6, 3.2), rho_l=0.0, rho_le=0.5)
ref = evaluate(Factory(models.GPBMA6, **base), n_per_fold=N, workers=W)
print(f'{"v3":22s}', fmt(ref), flush=True)
for zp in ['data/proc/emb_e5small_b0z.npy', 'data/proc/emb_qwen7b_L21_z_b0z.npy']:
    for T in [0.2, 0.4, 0.7]:
        df = evaluate(Factory(models.GPBMA7, zpath=zp, T=T, **base), n_per_fold=N, workers=W)
        d = df.ALC.to_numpy() - ref.ALC.to_numpy()
        print(f'{zp.split("/")[-1][:14]} T={T:<4}', fmt(df), f'dALC={d.mean():+.5f}±{d.std()/np.sqrt(len(d)):.5f}', flush=True)

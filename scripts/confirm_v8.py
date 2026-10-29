"""Check v8 against v7 on fresh episodes (another seed), paired."""
import os, sys; sys.path.insert(0, '.')
import numpy as np
from scripts.cv import evaluate, fmt, load
from paiec import models
load()
N = int(sys.argv[1]); SEED = int(sys.argv[2])
W = int(os.environ.get('W', 26))
Z = 'data/proc/emb_qwen7b_L21_z_b0z.npy'
Q = 'data/proc/emb_qwen7b_L21_z.npy'
v7 = dict(s_m=1.5, rho=0.73, rho_f=0.17, ell=0.12, var_scale=0.35, s_a_grid=(0.4, 0.8, 1.6), s_d_grid=(0.8, 1.6, 3.2),
          rho_l=0.0, rho_le=0.4, zpath=Z, T=0.3, s_m_grid=(1.0, 2.0, 4.0), vs_lo=0.35, vs_hi=0.2)
v8 = dict(v7, m_mix=[(-0.6, 0.8, 1), (-0.6, 1.6, 1), (-0.6, 3.2, 1)])
ref = evaluate(models.EnsFactory8([v7, dict(v7, emb=Q)]), n_per_fold=N, seed=SEED, workers=W)
print(f'{"v7":10s}', fmt(ref), flush=True)
df = evaluate(models.EnsFactory11([v8, dict(v8, emb=Q)]), n_per_fold=N, seed=SEED, workers=W)
d = df.ALC.to_numpy() - ref.ALC.to_numpy()
print(f'{"v8":10s}', fmt(df), f'dALC={d.mean():+.5f}±{d.std()/np.sqrt(len(d)):.5f}', flush=True)
for b in ('B0', 'B1', 'B3', 'B7'):
    x = df[b].to_numpy() - ref[b].to_numpy()
    print(f'  d{b}={x.mean():+.5f}±{x.std()/np.sqrt(len(x)):.5f}', flush=True)

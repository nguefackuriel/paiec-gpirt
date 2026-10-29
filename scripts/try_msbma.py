"""Joint evidence-weighted BMA across embedding sources vs uniform ensemble (v4)."""
import sys; sys.path.insert(0, '.')
import numpy as np
from scripts.cv import evaluate, fmt, load
from paiec import models
load()
N, W = 100, 26
Z = 'data/proc/emb_qwen7b_L21_z_b0z.npy'
S, Q, L = 'data/proc/emb_e5small.npy', 'data/proc/emb_qwen7b_L21_z.npy', 'data/proc/emb_e5large.npy'
base = dict(s_m=1.0, rho=0.73, rho_f=0.17, ell=0.12, var_scale=0.5, s_d_grid=(0.8, 1.6, 3.2), rho_l=0.0, rho_le=0.5, zpath=Z, T=0.3)
ref = evaluate(models.MSFactory([S, Q], mix=1.0, **base), n_per_fold=N, workers=W)
print(f'{"v4 (uniform over src)":24s}', fmt(ref), flush=True)
for name, mk in [('joint BMA', models.MSFactory([S, Q], **base)),
                 ('joint BMA t.5', models.MSFactory([S, Q], temp=0.5, **base)),
                 ('mix .5', models.MSFactory([S, Q], mix=0.5, **base)),
                 ('3 src uniform', models.MSFactory([S, Q, L], mix=1.0, **base)),
                 ('3 src joint', models.MSFactory([S, Q, L], **base))]:
    df = evaluate(mk, n_per_fold=N, workers=W)
    d = df.ALC.to_numpy() - ref.ALC.to_numpy()
    print(f'{name:24s}', fmt(df), f'dALC={d.mean():+.5f}±{d.std()/np.sqrt(len(d)):.5f}', flush=True)

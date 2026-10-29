"""Compare Qwen2.5-14B representations against the 7B ones inside the v4 recipe."""
import sys; sys.path.insert(0, '.')
import numpy as np
from scripts.cv import evaluate, fmt, load
from paiec import models
load()
N, W = 100, 26
S = 'data/proc/emb_e5small.npy'
v3 = dict(s_m=1.0, rho=0.73, rho_f=0.17, ell=0.12, var_scale=0.5, s_d_grid=(0.8, 1.6, 3.2), rho_l=0.0, rho_le=0.5)


def v4(zp, q, T=0.3):
    return models.EnsFactory7([{**v3, 'zpath': zp, 'T': T}, {**v3, 'zpath': zp, 'T': T, 'emb': q}])


ref = evaluate(v4('data/proc/emb_qwen7b_L21_z_b0z.npy', 'data/proc/emb_qwen7b_L21_z.npy'), n_per_fold=N, workers=W)
print(f'{"v4 (7B L21)":28s}', fmt(ref), flush=True)
for name, zp, q in [('14B L36 prior+kernel', 'data/proc/emb_qwen14b_L36_z_b0z.npy', 'data/proc/emb_qwen14b_L36_z.npy'),
                    ('14B L24 prior+kernel', 'data/proc/emb_qwen14b_L24_z_b0z.npy', 'data/proc/emb_qwen14b_L24_z.npy'),
                    ('14B L36 prior, 7B kernel', 'data/proc/emb_qwen14b_L36_z_b0z.npy', 'data/proc/emb_qwen7b_L21_z.npy')]:
    df = evaluate(v4(zp, q), n_per_fold=N, workers=W)
    d = df.ALC.to_numpy() - ref.ALC.to_numpy()
    print(f'{name:28s}', fmt(df), f'dALC={d.mean():+.5f}±{d.std()/np.sqrt(len(d)):.5f}', flush=True)

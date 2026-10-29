"""Meta-learned linear metric (e5 member) inside v6, realistic regime."""
import sys; sys.path.insert(0, '.')
import numpy as np
from scripts.cv import evaluate, fmt, load
from paiec import models
load()
N, SEED, W = 100, 0, 26
Z = 'data/proc/emb_qwen7b_L21_z_b0z.npy'
Q = 'data/proc/emb_qwen7b_L21_z.npy'
ML = sys.argv[1] if len(sys.argv) > 1 else 'data/proc/metalin_e5s_r32.npy'
v6 = dict(s_m=1.5, rho=0.73, rho_f=0.17, ell=0.12, var_scale=0.35, s_a_grid=(0.4, 0.8, 1.6), s_d_grid=(0.8, 1.6, 3.2),
          rho_l=0.0, rho_le=0.4, zpath=Z, T=0.3)
ref = evaluate(models.EnsFactory9([v6, {**v6, 'emb': Q}]), n_per_fold=N, seed=SEED, workers=W)
print(f'{"v6":26s}', fmt(ref), flush=True)
for name, kw in [('metalin a=.5', dict(alpha=0.5)), ('metalin a=1', dict(alpha=1.0)),
                 ('metalin a=.5 le=.6', dict(alpha=0.5, rho_le=0.6)), ('metalin a=1 le=.6', dict(alpha=1.0, rho_le=0.6))]:
    df = evaluate(models.EnsFactory9([{**v6, 'metalin': ML, **kw}, {**v6, 'emb': Q}]), n_per_fold=N, seed=SEED, workers=W)
    d = df.ALC.to_numpy() - ref.ALC.to_numpy()
    print(f'{name:26s}', fmt(df), f'dALC={d.mean():+.5f}±{d.std()/np.sqrt(len(d)):.5f}', flush=True)

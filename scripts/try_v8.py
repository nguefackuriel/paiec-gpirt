"""v8 candidates on top of v7 (realistic regime): numeric-metadata kernel, text-statistics kernel."""
import sys; sys.path.insert(0, '.')
import numpy as np
from scripts.cv import evaluate, fmt, load
from paiec import models
load()
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
W = 26
Z = 'data/proc/emb_qwen7b_L21_z_b0z.npy'
Q = 'data/proc/emb_qwen7b_L21_z.npy'
v7 = dict(s_m=1.5, rho=0.73, rho_f=0.17, ell=0.12, var_scale=0.35, s_a_grid=(0.4, 0.8, 1.6), s_d_grid=(0.8, 1.6, 3.2),
          rho_l=0.0, rho_le=0.4, zpath=Z, T=0.3, s_m_grid=(1.0, 2.0, 4.0), vs_lo=0.35, vs_hi=0.2,
          stats='data/proc/item_stats.npy')


def ens(**kw):
    a = {**v7, **kw}
    return models.EnsFactory10([a, {**a, 'emb': Q}])


ref = evaluate(ens(rho_n=0.0), n_per_fold=N, seed=SEED, workers=W)
print(f'{"v7":26s}', fmt(ref), flush=True)
for name, kw in [('numeric .1', dict(rho_n=0.1)), ('numeric .25', dict(rho_n=0.25)),
                 ('stats .05', dict(rho_n=0.0, rho_l=0.05)), ('stats .1', dict(rho_n=0.0, rho_l=0.1)),
                 ('numeric .1 + stats .05', dict(rho_n=0.1, rho_l=0.05))]:
    df = evaluate(ens(**kw), n_per_fold=N, seed=SEED, workers=W)
    d = df.ALC.to_numpy() - ref.ALC.to_numpy()
    print(f'{name:26s}', fmt(df), f'dALC={d.mean():+.5f}±{d.std()/np.sqrt(len(d)):.5f}', flush=True)

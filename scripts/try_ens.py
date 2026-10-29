import sys, time; sys.path.insert(0, '.')
import numpy as np
from scripts.cv import evaluate, fmt, load, Factory
from paiec import models
load()
N, W = 60, 24
base = dict(s_m=1.0, rho=0.73, rho_f=0.17, ell=0.12, var_scale=0.5, s_d_grid=(0.8, 1.6, 3.2), rho_l=0.0, rho_le=0.5)
E = {'s': 'data/proc/emb_e5small.npy', 'l': 'data/proc/emb_e5large.npy', 'b': 'data/proc/emb_e5base.npy',
     'q14': 'data/proc/emb_qwen7b_L14_z.npy', 'q21': 'data/proc/emb_qwen7b_L21_z.npy'}
ref = evaluate(Factory(models.GPBMA6, **base), n_per_fold=N, workers=W)
print(f'{"e5small":16s}', fmt(ref), flush=True)
for name, keys in [('s+l', 's l'), ('s+q14', 's q14'), ('s+l+q14', 's l q14'), ('s+b+l', 's b l'), ('s+b+l+q14+q21', 's b l q14 q21')]:
    cfgs = [{**base, 'emb': E[k]} for k in keys.split()]
    df = evaluate(models.EnsFactory(cfgs), n_per_fold=N, workers=W)
    d = df.ALC.to_numpy() - ref.ALC.to_numpy()
    print(f'{name:16s}', fmt(df), f'dALC={d.mean():+.5f}±{d.std()/np.sqrt(len(d)):.5f}', flush=True)
# kernel mixture variant: s_d BMA with also rho_le alternatives
for name, kw in [('le.5 l.1 ens s+l', None)]:
    cfgs = [{**base, 'emb': E['s'], 'rho_l': 0.1, 'stats': 'data/proc/item_stats.npy'}, {**base, 'emb': E['l']}]
    df = evaluate(models.EnsFactory(cfgs), n_per_fold=N, workers=W)
    d = df.ALC.to_numpy() - ref.ALC.to_numpy()
    print(f'{name:16s}', fmt(df), f'dALC={d.mean():+.5f}±{d.std()/np.sqrt(len(d)):.5f}', flush=True)

"""var_scale x rho_le grid for v4 under the current draw regime (set PAIEC_MAX_GROUP); prints draw composition."""
import os, sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd
from scripts.cv import evaluate, fmt, load, episodes
from paiec import models
c = load()
N, W = 100, 26
eps = episodes(N)
gs = c['groups'].set_index('g')
comp = pd.DataFrame([{'groups': len(ch), 'benches': gs.loc[ch].b.nunique(), 'items': gs.loc[ch].n_items.mean()} for _, _, ch in eps])
print('MAX_GROUP', os.environ.get('PAIEC_MAX_GROUP'), 'draw composition (mean):', comp.mean().round(2).to_dict(), flush=True)
Z = 'data/proc/emb_qwen7b_L21_z_b0z.npy'
Q = 'data/proc/emb_qwen7b_L21_z.npy'
v3 = dict(s_m=1.0, rho=0.73, rho_f=0.17, ell=0.12, var_scale=0.5, s_d_grid=(0.8, 1.6, 3.2), rho_l=0.0, rho_le=0.5)


def v4(**kw):
    a = {**v3, 'zpath': Z, 'T': 0.3, **kw}
    return models.EnsFactory7([a, {**a, 'emb': Q}])


ref = evaluate(v4(), n_per_fold=N, workers=W)
print(f'{"v4 (vs.5 le.5)":20s}', fmt(ref), flush=True)
for vs in [0.25, 0.35, 0.5]:
    for le in [0.3, 0.5]:
        if vs == 0.5 and le == 0.5:
            continue
        df = evaluate(v4(var_scale=vs, rho_le=le), n_per_fold=N, workers=W)
        d = df.ALC.to_numpy() - ref.ALC.to_numpy()
        print(f'vs={vs} le={le:<10}', fmt(df), f'dALC={d.mean():+.5f}±{d.std()/np.sqrt(len(d)):.5f}', flush=True)

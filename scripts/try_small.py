"""Robustness: model ranking when groups are small (<= PAIEC_MAX_GROUP items), as in formative feedback."""
import sys; sys.path.insert(0, '.')
import numpy as np
from scripts.cv import evaluate, fmt, load, Factory
from paiec import models
load()
N, W = 100, 26
Z = 'data/proc/emb_qwen7b_L21_z_b0z.npy'
Q = 'data/proc/emb_qwen7b_L21_z.npy'
v2 = dict(s_m=1.0, rho=0.73, rho_f=0.17, ell=0.12, var_scale=0.5, s_d_grid=(0.8, 1.6, 3.2))
v3 = dict(v2, rho_l=0.0, rho_le=0.5)


def v4(**kw):
    a = {**v3, 'zpath': Z, 'T': 0.3, **kw}
    return models.EnsFactory7([a, {**a, 'emb': Q}])


ref = evaluate(v4(), n_per_fold=N, workers=W)
print(f'{"v4":24s}', fmt(ref), flush=True)
for name, mk in [('group mean', Factory(models.GroupMean, prior=0.5, strength=2.0)),
                 ('v2', Factory(models.GPBMA2, **v2)), ('v3', Factory(models.GPBMA6, **v3)),
                 ('v4 rho_le .3', v4(rho_le=0.3)), ('v4 rho_le .7', v4(rho_le=0.7)),
                 ('v4 vs .35', v4(var_scale=0.35)), ('v4 T .45', v4(T=0.45))]:
    df = evaluate(mk, n_per_fold=N, workers=W)
    d = df.ALC.to_numpy() - ref.ALC.to_numpy()
    print(f'{name:24s}', fmt(df), f'dALC={d.mean():+.5f}±{d.std()/np.sqrt(len(d)):.5f}', flush=True)

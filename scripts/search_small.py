"""Random search of v4 hyperparameters in the realistic small-group regime (PAIEC_MAX_GROUP=200)."""
import sys; sys.path.insert(0, '.')
import numpy as np
from scripts.cv import evaluate, fmt, load
from paiec import models
load()
N, W = 100, 26
Z = 'data/proc/emb_qwen7b_L21_z_b0z.npy'
Q = 'data/proc/emb_qwen7b_L21_z.npy'
v3 = dict(s_m=1.0, rho=0.73, rho_f=0.17, ell=0.12, var_scale=0.5, s_d_grid=(0.8, 1.6, 3.2), rho_l=0.0, rho_le=0.5)


def v4(**kw):
    a = {**v3, 'zpath': Z, 'T': 0.3, **kw}
    return models.EnsFactory7([a, {**a, 'emb': Q}])


ref = evaluate(v4(var_scale=0.25, rho_le=0.3), n_per_fold=N, workers=W)
print(f'{"ref vs.25 le.3":20s}', fmt(ref), flush=True)
rng = np.random.default_rng(7)
res = []
for t in range(30):
    kw = dict(var_scale=float(rng.choice([0.0, 0.1, 0.15, 0.2, 0.25, 0.3])),
              rho_le=float(rng.choice([0.15, 0.2, 0.3, 0.4])),
              s_m=float(rng.choice([0.7, 1.0, 1.5])),
              T=float(rng.choice([0.2, 0.3, 0.45])),
              rho=float(rng.choice([0.5, 0.73, 0.85])),
              s_a_grid=[(0.6, 1.2, 2.4), (0.4, 0.8, 1.6), (0.8, 1.6, 3.2)][rng.integers(3)],
              s_d_grid=[(0.8, 1.6, 3.2), (0.5, 1.0, 2.0), (1.0, 2.0, 4.0)][rng.integers(3)])
    if kw['rho'] + v3['rho_f'] > 0.97:
        kw['rho'] = 0.8
    df = evaluate(v4(**kw), n_per_fold=N, workers=W)
    d = df.ALC.to_numpy() - ref.ALC.to_numpy()
    res.append((d.mean(), d.std() / np.sqrt(len(d)), kw))
    print(f'{t:2d} dALC={d.mean():+.5f}±{d.std()/np.sqrt(len(d)):.5f}', fmt(df), kw, flush=True)
res.sort(key=lambda r: r[0])
print('TOP'); [print(r) for r in res[:6]]

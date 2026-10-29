import sys, time, json; sys.path.insert(0, '.')
import numpy as np
from scripts.cv import evaluate, fmt, load, Factory
from paiec import models
load()
N = 60
ref = evaluate(Factory(models.GPBMA2, s_m=2.0), n_per_fold=N, workers=24)
print('ref v1', fmt(ref), flush=True)
rng = np.random.default_rng(42)
res = []
for t in range(40):
    kw = dict(s_m=float(rng.choice([1.0, 1.5, 2.0, 2.5, 3.0])),
              rho=float(rng.uniform(0.3, 0.9)), rho_f=float(rng.uniform(0, 0.4)),
              ell=float(rng.choice([0.03, 0.05, 0.08, 0.12, 0.2])),
              var_scale=float(rng.choice([0.3, 0.5, 0.7, 1.0])),
              s_d_grid=[(0.5, 1.0, 2.0), (0.8, 1.6, 3.2), (0.4, 0.8, 1.6, 3.2)][rng.integers(3)])
    if kw['rho'] + kw['rho_f'] > 0.97:
        kw['rho_f'] = 0.97 - kw['rho']
    df = evaluate(Factory(models.GPBMA2, **kw), n_per_fold=N, workers=24)
    d = df.ALC.to_numpy() - ref.ALC.to_numpy()
    res.append((d.mean(), d.std() / np.sqrt(len(d)), kw, fmt(df)))
    print(f'{t:2d} dALC={d.mean():+.5f}±{d.std()/np.sqrt(len(d)):.5f}', fmt(df), kw, flush=True)
res.sort(key=lambda r: r[0])
print('TOP'); [print(r) for r in res[:6]]

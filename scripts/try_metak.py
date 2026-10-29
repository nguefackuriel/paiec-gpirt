import sys, time; sys.path.insert(0, '.')
import numpy as np
from scripts.cv import evaluate, fmt, load, Factory
from paiec import models
load()
N = 60
base = dict(s_m=1.0, rho=0.73, rho_f=0.17, ell=0.12, var_scale=0.5, s_d_grid=(0.8, 1.6, 3.2))
ref = evaluate(Factory(models.GPBMA2, **base), n_per_fold=N, workers=24)
print(f'{"gpbma2 best":18s}', fmt(ref), flush=True)
for name, kw in [('metak rho.73', dict()), ('metak rho.5', dict(rho=0.5, rho_f=0.17)), ('metak rho.85 rf.1', dict(rho=0.85, rho_f=0.1)),
                 ('metak mix.5', dict(mix=0.5))]:
    t = time.time()
    df = evaluate(Factory(models.GPBMA4, pass_fold=True, **{**base, **kw}), n_per_fold=N, workers=24)
    d = df.ALC.to_numpy() - ref.ALC.to_numpy()
    print(f'{name:18s}', fmt(df), f'dALC={d.mean():+.5f}±{d.std()/np.sqrt(len(d)):.5f}', f'{time.time()-t:.0f}s', flush=True)

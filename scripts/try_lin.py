import sys, time; sys.path.insert(0, '.')
import numpy as np
from scripts.cv import evaluate, fmt, load, Factory
from paiec import models
load()
N = 60
base = dict(s_m=1.0, rho=0.73, rho_f=0.17, ell=0.12, var_scale=0.5, s_d_grid=(0.8, 1.6, 3.2))
ref = evaluate(Factory(models.GPBMA2, **base), n_per_fold=N, workers=24)
print(f'{"v2":10s}', fmt(ref), flush=True)
for rl in [0.05, 0.15, 0.3]:
    df = evaluate(Factory(models.GPBMA5, rho_l=rl, **base), n_per_fold=N, workers=24)
    d = df.ALC.to_numpy() - ref.ALC.to_numpy()
    print(f'lin {rl:<6}', fmt(df), f'dALC={d.mean():+.5f}±{d.std()/np.sqrt(len(d)):.5f}', flush=True)

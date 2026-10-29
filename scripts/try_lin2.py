import sys, time; sys.path.insert(0, '.')
import numpy as np
from scripts.cv import evaluate, fmt, load, Factory
from paiec import models
load()
N = 60
base = dict(s_m=1.0, rho=0.73, rho_f=0.17, ell=0.12, var_scale=0.5, s_d_grid=(0.8, 1.6, 3.2))
ref = evaluate(Factory(models.GPBMA5, rho_l=0.15, **base), n_per_fold=N, workers=24)
print(f'{"lin1 .15":14s}', fmt(ref), flush=True)
for name, kw in [('stats2 .15', dict(rho_l=0.15)), ('stats2 .25', dict(rho_l=0.25)),
                 ('stats2 .15 le.1', dict(rho_l=0.15, rho_le=0.1)), ('stats2 .15 le.25', dict(rho_l=0.15, rho_le=0.25)),
                 ('le.2 only', dict(rho_l=0.0, rho_le=0.2))]:
    df = evaluate(Factory(models.GPBMA6, **kw, **base), n_per_fold=N, workers=24)
    d = df.ALC.to_numpy() - ref.ALC.to_numpy()
    print(f'{name:14s}', fmt(df), f'dALC={d.mean():+.5f}±{d.std()/np.sqrt(len(d)):.5f}', flush=True)

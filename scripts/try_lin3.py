import sys, time; sys.path.insert(0, '.')
import numpy as np
from scripts.cv import evaluate, fmt, load, Factory
from paiec import models
load()
N = 60
base = dict(s_m=1.0, rho=0.73, rho_f=0.17, ell=0.12, var_scale=0.5, s_d_grid=(0.8, 1.6, 3.2), stats="data/proc/item_stats.npy")
ref = evaluate(Factory(models.GPBMA6, rho_l=0.0, rho_le=0.2, **base), n_per_fold=N, workers=24)
print(f'{"le.2":16s}', fmt(ref), flush=True)
for name, kw in [('le.35', dict(rho_l=0.0, rho_le=0.35)), ('le.5', dict(rho_l=0.0, rho_le=0.5)), ('le.7', dict(rho_l=0.0, rho_le=0.7)),
                 ('l.1 le.35', dict(rho_l=0.1, rho_le=0.35)), ('l.1 le.5', dict(rho_l=0.1, rho_le=0.5)),
                 ('l.1 le.5 vs.7', dict(rho_l=0.1, rho_le=0.5, var_scale=0.7)), ('l.1 le.5 sd4', dict(rho_l=0.1, rho_le=0.5, s_d_grid=(0.6, 1.2, 2.4, 4.8)))]:
    df = evaluate(Factory(models.GPBMA6, **{**base, **kw}), n_per_fold=N, workers=24)
    d = df.ALC.to_numpy() - ref.ALC.to_numpy()
    print(f'{name:16s}', fmt(df), f'dALC={d.mean():+.5f}±{d.std()/np.sqrt(len(d)):.5f}', flush=True)

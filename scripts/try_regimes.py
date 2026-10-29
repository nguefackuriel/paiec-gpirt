import sys, os, time; sys.path.insert(0, '.')
import numpy as np
from scripts.cv import evaluate, fmt, load, Factory
from paiec import models
load()
N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
v2 = dict(s_m=1.0, rho=0.73, rho_f=0.17, ell=0.12, var_scale=0.5, s_d_grid=(0.8, 1.6, 3.2))
ref = evaluate(Factory(models.GPBMA2, **v2), n_per_fold=N, workers=24)
print(os.environ.get('PAIEC_DRAW'), f'{"v2":10s}', fmt(ref), flush=True)
for name, kw in [('v3 le.5', dict(rho_l=0.0, rho_le=0.5)), ('v3 le.3', dict(rho_l=0.0, rho_le=0.3)), ('v3 le.5 vs.35', dict(rho_l=0.0, rho_le=0.5, var_scale=0.35))]:
    df = evaluate(Factory(models.GPBMA6, **v2, **kw), n_per_fold=N, workers=24)
    d = df.ALC.to_numpy() - ref.ALC.to_numpy()
    print(os.environ.get('PAIEC_DRAW'), f'{name:10s}', fmt(df), f'dALC={d.mean():+.5f}±{d.std()/np.sqrt(len(d)):.5f}', flush=True)

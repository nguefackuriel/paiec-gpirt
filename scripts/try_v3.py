import sys, time; sys.path.insert(0, '.')
from scripts.cv import evaluate, fmt, load, Factory
from paiec import models
load()
B = dict(rho=0.7, rho_f=0.2)
cfgs = {
    'vs0.5': dict(var_scale=0.5),
    'vs0.35': dict(var_scale=0.35),
    'vs0.5 s_m3': dict(var_scale=0.5, s_m=3.0),
    'vs0.5 s_m4': dict(var_scale=0.5, s_m=4.0),
    'vs0.5 sa_grid wide': dict(var_scale=0.5, s_a_grid=(0.4, 0.8, 1.6, 3.2)),
    'vs0.5 sd_grid wide': dict(var_scale=0.5, s_d_grid=(0.6, 1.2, 2.4, 4.8)),
}
for name, kw in cfgs.items():
    t = time.time()
    print(f'{name:22s}', fmt(evaluate(Factory(models.GPBMA2, **B, **kw), n_per_fold=6, workers=24)), f'{time.time()-t:.0f}s', flush=True)

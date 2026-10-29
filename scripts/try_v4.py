import sys, time; sys.path.insert(0, '.')
from scripts.cv import evaluate, fmt, load, Factory
from paiec import models
load()
B = dict(rho=0.7, rho_f=0.2)
cfgs = {
    'vs0.2': dict(var_scale=0.2),
    'vs0.0': dict(var_scale=0.0),
    'vs0.2 s_b0.5': dict(var_scale=0.2, s_b=0.5),
    'vs0.2 s_b1.0': dict(var_scale=0.2, s_b=1.0),
}
for name, kw in cfgs.items():
    t = time.time()
    print(f'{name:22s}', fmt(evaluate(Factory(models.GPBMA3, **B, **kw), n_per_fold=6, workers=24)), f'{time.time()-t:.0f}s', flush=True)

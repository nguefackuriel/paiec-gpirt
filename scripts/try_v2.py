import sys, time; sys.path.insert(0, '.')
from scripts.cv import evaluate, fmt, load, Factory
from paiec import models
load()
cfgs = {
    'v2 base(=bma3x3)': dict(),
    'hp0.5': dict(hp=0.5),
    'hp2': dict(hp=2.0),
    'rho_f0.2 (rho_e .7)': dict(rho=0.7, rho_f=0.2),
    'rho_f0.4 (rho_e .5)': dict(rho=0.5, rho_f=0.4),
    'ell0.03': dict(ell=0.03),
    'var_scale1.5': dict(var_scale=1.5),
    'var_scale0.7': dict(var_scale=0.7),
}
for name, kw in cfgs.items():
    t = time.time()
    print(f'{name:22s}', fmt(evaluate(Factory(models.GPBMA2, **kw), n_per_fold=6, workers=24)), f'{time.time()-t:.0f}s', flush=True)

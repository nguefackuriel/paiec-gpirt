import sys, time; sys.path.insert(0, '.')
from scripts.cv import evaluate, fmt, load, Factory
from paiec import models
load()
cfgs = {
    'gp_single(2.5,1.2,2.0)': dict(s_m=2.5, s_a_grid=(1.2,), s_d_grid=(2.0,)),
    'bma3x3': dict(),
    'bma3x3_t0.5': dict(temp=0.5),
    'uniform3x3': dict(bma=False),
    'bma4x4': dict(s_a_grid=(0.4, 0.8, 1.6, 3.2), s_d_grid=(0.5, 1.0, 2.0, 4.0)),
}
for name, kw in cfgs.items():
    t = time.time()
    print(f'{name:24s}', fmt(evaluate(Factory(models.GPBMA, **kw), n_per_fold=6, workers=24)), f'{time.time()-t:.0f}s', flush=True)

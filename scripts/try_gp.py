import sys, time, itertools; sys.path.insert(0, '.')
from scripts.cv import evaluate, fmt, load, Factory
from paiec import models
load()
print('irt  ', fmt(evaluate(Factory(models.IRT), n_per_fold=4, workers=12)), flush=True)
print('gp0  ', fmt(evaluate(Factory(models.GPIRT, rho=0.0), n_per_fold=4, workers=12)), flush=True)
for rho, ell in itertools.product([0.3, 0.6, 0.85], [0.05, 0.15]):
    t = time.time()
    print(f'gp rho={rho} ell={ell}', fmt(evaluate(Factory(models.GPIRT, rho=rho, ell=ell), n_per_fold=4, workers=12)), f'{time.time()-t:.0f}s', flush=True)

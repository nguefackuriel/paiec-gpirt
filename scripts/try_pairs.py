import sys, time; sys.path.insert(0, '.')
from scripts.cv import evaluate, fmt, load, Factory
from paiec import models
load()
N = 40
cfgs = [
    ('groupmean', Factory(models.GroupMean)),
    ('groupmean s4', Factory(models.GroupMean, strength=4.0)),
    ('irt(2.5,1.2,2.0)', Factory(models.IRT, s_m=2.5, s_a=1.2, s_d=2.0)),
    ('gpbma2 v1cfg', Factory(models.GPBMA2, s_m=2.0)),
    ('gpbma2 best-old', Factory(models.GPBMA2, rho=0.7, rho_f=0.2, var_scale=0.35)),
    ('gpbma2 s_m1', Factory(models.GPBMA2, rho=0.7, rho_f=0.2, s_m=1.0)),
    ('gpbma2 s_m1 vs.5', Factory(models.GPBMA2, rho=0.7, rho_f=0.2, s_m=1.0, var_scale=0.5)),
]
for name, mk in cfgs:
    t = time.time()
    print(f'{name:20s}', fmt(evaluate(mk, n_per_fold=N, workers=24)), f'{time.time()-t:.0f}s', flush=True)

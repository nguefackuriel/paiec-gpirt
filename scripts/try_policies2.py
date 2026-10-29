import sys, time; sys.path.insert(0, '.')
import numpy as np
from scripts.cv import evaluate_stream, fmt, load, Factory
from paiec import models, policies
load()
base = dict(s_m=1.0, rho=0.73, rho_f=0.17, ell=0.12, var_scale=0.5, s_d_grid=(0.8, 1.6, 3.2), rho_l=0.0, rho_le=0.5)
M = Factory(models.GPBMA6, **base)
N, W = 60, 24
ref = evaluate_stream(M, Factory(policies.RandomPolicy), n_per_fold=N, workers=W)
print(f'{"random":14s}', fmt(ref), flush=True)
for name, P in [('var tau.5', Factory(policies.VariancePolicy, tau=0.5)), ('var tau.65', Factory(policies.VariancePolicy, tau=0.65)),
                ('var tau.8', Factory(policies.VariancePolicy, tau=0.8)), ('var tau.65 sl3', Factory(policies.VariancePolicy, tau=0.65, slack=3.0))]:
    df = evaluate_stream(M, P, n_per_fold=N, workers=W)
    d = df.ALC.to_numpy() - ref.ALC.to_numpy()
    print(f'{name:14s}', fmt(df), f'dALC={d.mean():+.5f}±{d.std()/np.sqrt(len(d)):.5f}', flush=True)

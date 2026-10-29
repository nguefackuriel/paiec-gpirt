import sys, time; sys.path.insert(0, '.')
import numpy as np
from scripts.cv import evaluate_stream, fmt, load, Factory
from paiec import models, policies
load()
M = Factory(models.GPBMA2, s_m=2.0)
N = 40
ref = evaluate_stream(M, Factory(policies.RandomPolicy), n_per_fold=N, workers=20)
print(f'{"random":16s}', fmt(ref), flush=True)
for name, P in [('div tau.3', Factory(policies.DiversityPolicy, tau=0.3)),
                ('div tau.5', Factory(policies.DiversityPolicy, tau=0.5)),
                ('div tau.7', Factory(policies.DiversityPolicy, tau=0.7)),
                ('div tau.5 sl3', Factory(policies.DiversityPolicy, tau=0.5, slack=3.0))]:
    t = time.time()
    df = evaluate_stream(M, P, n_per_fold=N, workers=20)
    d = df.ALC.to_numpy() - ref.ALC.to_numpy()
    print(f'{name:16s}', fmt(df), f'dALC={d.mean():+.5f}±{d.std()/np.sqrt(len(d)):.5f}', f'{time.time()-t:.0f}s', flush=True)

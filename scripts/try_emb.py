import sys, time; sys.path.insert(0, '.')
import numpy as np
from scripts.cv import evaluate, fmt, load, Factory
from paiec import models
load()
N = int(sys.argv[2]) if len(sys.argv) > 2 else 60
W = int(sys.argv[3]) if len(sys.argv) > 3 else 20
embs = sys.argv[1].split(',')
base = dict(s_m=1.0, rho=0.73, rho_f=0.17, ell=0.12, var_scale=0.5, s_d_grid=(0.8, 1.6, 3.2), rho_l=0.0, rho_le=0.5)
ref = evaluate(Factory(models.GPBMA6, **base), n_per_fold=N, workers=W)
print(f'{"e5small":30s}', fmt(ref), flush=True)
for e in embs:
    for tag, kw in [('', dict()), ('noRBF', dict(rho=0.0)), ('noRBF le.7', dict(rho=0.0, rho_le=0.7))]:
        t = time.time()
        df = evaluate(Factory(models.GPBMA6, **{**base, 'emb': e, **kw}), n_per_fold=N, workers=W)
        d = df.ALC.to_numpy() - ref.ALC.to_numpy()
        print(f'{e.split("/")[-1][:20]:20s} {tag:10s}', fmt(df), f'dALC={d.mean():+.5f}±{d.std()/np.sqrt(len(d)):.5f} {time.time()-t:.0f}s', flush=True)

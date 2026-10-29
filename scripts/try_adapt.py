"""Centre of the shift prior learned online from the other benchmarks of the episode (hierarchical prior)."""
import os, sys; sys.path.insert(0, '.')
import numpy as np
from scripts.cv import evaluate, fmt, load
from paiec import models
load()
N = int(sys.argv[1]); SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
W = int(os.environ.get('W', 26))
Z = 'data/proc/emb_qwen7b_L21_z_b0z.npy'
Q = 'data/proc/emb_qwen7b_L21_z.npy'
v8 = dict(s_m=1.5, rho=0.73, rho_f=0.17, ell=0.12, var_scale=0.35, s_a_grid=(0.4, 0.8, 1.6), s_d_grid=(0.8, 1.6, 3.2),
          rho_l=0.0, rho_le=0.4, zpath=Z, T=0.3, vs_lo=0.35, vs_hi=0.2,
          m_mix=[(-0.6, 0.8, 1), (-0.6, 1.6, 1), (-0.6, 3.2, 1)])
ens = lambda **kw: models.EnsFactory11([dict(v8, **kw), dict(v8, emb=Q, **kw)])
ref = evaluate(ens(), n_per_fold=N, seed=SEED, workers=W)
print(f'{"v8":24s}', fmt(ref), flush=True)
for name, kw in [('adapt tau .3', dict(adapt_tau=0.3)), ('adapt tau .6', dict(adapt_tau=0.6)),
                 ('adapt tau 1', dict(adapt_tau=1.0))]:
    df = evaluate(ens(**kw), n_per_fold=N, seed=SEED, workers=W)
    d = df.ALC.to_numpy() - ref.ALC.to_numpy()
    by = ' '.join(f'f{f}:{d[df.fold.to_numpy() == f].mean():+.4f}' for f in range(5))
    print(f'{name:24s}', fmt(df), f'dALC={d.mean():+.5f}±{d.std()/np.sqrt(len(d)):.5f}', by, flush=True)

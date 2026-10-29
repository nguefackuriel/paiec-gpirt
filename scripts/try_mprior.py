"""Empirical prior on the benchmark shift m (mean and mixture learned from training groups) vs v7."""
import os, sys; sys.path.insert(0, '.')
import numpy as np
from scripts.cv import evaluate, fmt, load
from paiec import models
load()
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
W = int(os.environ.get('W', 26))
Z = 'data/proc/emb_qwen7b_L21_z_b0z.npy'
Q = 'data/proc/emb_qwen7b_L21_z.npy'
v7 = dict(s_m=1.5, rho=0.73, rho_f=0.17, ell=0.12, var_scale=0.35, s_a_grid=(0.4, 0.8, 1.6), s_d_grid=(0.8, 1.6, 3.2),
          rho_l=0.0, rho_le=0.4, zpath=Z, T=0.3, s_m_grid=(1.0, 2.0, 4.0), vs_lo=0.35, vs_hi=0.2)


def ens11(mix):
    return models.EnsFactory11([dict(v7, m_mix=mix), dict(v7, m_mix=mix, emb=Q)])


SC = lambda mu: [(mu, 1.0, 1), (mu, 2.0, 1), (mu, 4.0, 1)]   # v7 scale mixture, shifted centre
ref = evaluate(models.EnsFactory8([v7, dict(v7, emb=Q)]), n_per_fold=N, seed=SEED, workers=W)
print(f'{"v7":34s}', fmt(ref), flush=True)
for name, mix in [('v7 as mixture (check)', SC(0.0)),
                  ('centre -0.4', SC(-0.4)),
                  ('centre -0.8', SC(-0.8)),
                  ('centre -1.2', SC(-1.2)),
                  ('3 comps (-2.5,-.6,1.5) s=1', [(-2.5, 1.0, 1), (-0.6, 1.0, 1), (1.5, 1.0, 1)]),
                  ('5 comps (-3..2) s=.8', [(-3.0, .8, 1), (-1.75, .8, 1), (-.5, .8, 1), (.75, .8, 1), (2.0, .8, 1)]),
                  ]:
    df = evaluate(ens11(mix), n_per_fold=N, seed=SEED, workers=W)
    d = df.ALC.to_numpy() - ref.ALC.to_numpy()
    print(f'{name:34s}', fmt(df), f'dALC={d.mean():+.5f}±{d.std()/np.sqrt(len(d)):.5f}', flush=True)

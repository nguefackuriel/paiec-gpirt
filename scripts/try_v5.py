"""v5 candidates: subject-name ability prior; BMA over kernel configurations."""
import sys; sys.path.insert(0, '.')
import numpy as np
from scripts.cv import evaluate, fmt, load
from paiec import models
load()
N, W = 100, 26
Z = 'data/proc/emb_qwen7b_L21_z_b0z.npy'
Q = 'data/proc/emb_qwen7b_L21_z.npy'
SA = 'data/proc/subj_ability_oof.npy'
v3 = dict(s_m=1.0, rho=0.73, rho_f=0.17, ell=0.12, var_scale=0.5, s_d_grid=(0.8, 1.6, 3.2), rho_l=0.0, rho_le=0.5)


def v4(**kw):
    a = {**v3, 'zpath': Z, 'T': 0.3, **kw}
    return models.EnsFactory7([a, {**a, 'emb': Q}])


ref = evaluate(v4(), n_per_fold=N, workers=W)
print(f'{"v4":30s}', fmt(ref), flush=True)


def show(name, mk):
    df = evaluate(mk, n_per_fold=N, workers=W)
    d = df.ALC.to_numpy() - ref.ALC.to_numpy()
    print(f'{name:30s}', fmt(df), f'dALC={d.mean():+.5f}±{d.std()/np.sqrt(len(d)):.5f}', flush=True)


for Ts in [0.3, 0.6, 1.0]:
    show(f'subject prior Ts={Ts}', v4(spath=SA, Ts=Ts))
# kernel-configuration BMA (each source), averaged over the two sources
for name, grid in [('BMA rho_le{.25,.5,.75}', [dict(rho_le=0.25), dict(rho_le=0.5), dict(rho_le=0.75)]),
                   ('BMA rho_le x rho', [dict(rho_le=0.3, rho=0.73), dict(rho_le=0.6, rho=0.73), dict(rho_le=0.3, rho=0.4), dict(rho_le=0.6, rho=0.4)])]:
    class F:
        def __init__(self, grid): self.grid = grid
        def __call__(self, f):
            mems = []
            for emb in (None, Q):
                cfgs = [{**v3, 'zpath': Z, 'T': 0.3, **({'emb': emb} if emb else {}), **g} for g in self.grid]
                mems.append(models.MultiCfgBMA(cfgs))
            return models.Ensemble(mems)
    show(name, F(grid))

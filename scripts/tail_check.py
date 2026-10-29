"""Per-group budget-curve statistics in simulation (v4 vs v5): how often do early labels hurt?"""
import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from scripts.cv import load, episodes
from paiec import sim, models

Z = 'data/proc/emb_qwen7b_L21_z_b0z.npy'
Q = 'data/proc/emb_qwen7b_L21_z.npy'
v3 = dict(s_m=1.0, rho=0.73, rho_f=0.17, ell=0.12, var_scale=0.5, s_d_grid=(0.8, 1.6, 3.2), rho_l=0.0, rho_le=0.5)
CFG = {'v4': {}, 'v5': dict(var_scale=0.2, rho_le=0.4, s_m=1.5, s_a_grid=(0.4, 0.8, 1.6)),
       'vs.35': dict(var_scale=0.35, rho_le=0.4, s_m=1.5, s_a_grid=(0.4, 0.8, 1.6))}


def run(args):
    name, f, s, ch = args
    c = load()
    a = {**v3, 'zpath': Z, 'T': 0.3, **CFG[name]}
    m = models.Ensemble([models.GPBMA7(**a), models.GPBMA7(**{**a, 'emb': Q})])
    res = sim.run_episode(m, c['o'], ch, np.random.default_rng(s))
    res['model'] = name
    return res


if __name__ == '__main__':
    eps = episodes(100)
    with ProcessPoolExecutor(26) as ex:
        out = pd.concat(list(ex.map(run, [(n, f, s, ch) for n in CFG for f, s, ch in eps])))
    for n, d in out.groupby('model'):
        print(n, f"groups={len(d)} ALC={d.ALC.mean():.4f}  P(B1>B0)={(d.B1 > d.B0).mean():.3f} P(B3>B0)={(d.B3 > d.B0).mean():.3f} "
              f"P(B3>0.3)={(d.B3 > 0.3).mean():.3f} P(ALC>0.28)={(d.ALC > 0.28).mean():.3f} q90 ALC={d.ALC.quantile(.9):.4f}", flush=True)

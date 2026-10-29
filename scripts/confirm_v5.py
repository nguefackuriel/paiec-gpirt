"""Confirm v5 candidates on fresh episodes (seed 1) under the current regime (env PAIEC_DRAW / PAIEC_MAX_GROUP)."""
import os, sys; sys.path.insert(0, '.')
import numpy as np
from scripts.cv import evaluate, fmt, load
from paiec import models
load()
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 1
W = 26
Z = 'data/proc/emb_qwen7b_L21_z_b0z.npy'
Q = 'data/proc/emb_qwen7b_L21_z.npy'
v3 = dict(s_m=1.0, rho=0.73, rho_f=0.17, ell=0.12, var_scale=0.5, s_d_grid=(0.8, 1.6, 3.2), rho_l=0.0, rho_le=0.5)


def v4(**kw):
    a = {**v3, 'zpath': Z, 'T': 0.3, **kw}
    return models.EnsFactory7([a, {**a, 'emb': Q}])


CANDS = {
    'A vs.1 le.3 sm1.5 T.45 rho.8 sa(.4,.8,1.6)': dict(var_scale=0.1, rho_le=0.3, s_m=1.5, T=0.45, rho=0.8, s_a_grid=(0.4, 0.8, 1.6)),
    'B vs.15 le.4 sm1.5 sa(.4,.8,1.6)': dict(var_scale=0.15, rho_le=0.4, s_m=1.5, s_a_grid=(0.4, 0.8, 1.6)),
    'C vs.2 le.4 sm1.5 sa(.4,.8,1.6)': dict(var_scale=0.2, rho_le=0.4, s_m=1.5, s_a_grid=(0.4, 0.8, 1.6)),
    'D vs.25 le.3': dict(var_scale=0.25, rho_le=0.3),
    'E vs.35 le.4 sm1.5 sa(.4,.8,1.6) [v6]': dict(var_scale=0.35, rho_le=0.4, s_m=1.5, s_a_grid=(0.4, 0.8, 1.6)),
}
if len(sys.argv) > 3:
    CANDS = {k: v for k, v in CANDS.items() if k.startswith(tuple(sys.argv[3].split(',')))}
tag = f"draw={os.environ.get('PAIEC_DRAW', 'pairs')} max={os.environ.get('PAIEC_MAX_GROUP', '1000')} seed={SEED}"
ref = evaluate(v4(), n_per_fold=N, seed=SEED, workers=W)
print(tag, f'{"v4":44s}', fmt(ref), flush=True)
for name, kw in CANDS.items():
    df = evaluate(v4(**kw), n_per_fold=N, seed=SEED, workers=W)
    d = df.ALC.to_numpy() - ref.ALC.to_numpy()
    print(tag, f'{name:44s}', fmt(df), f'dALC={d.mean():+.5f}±{d.std()/np.sqrt(len(d)):.5f}', flush=True)

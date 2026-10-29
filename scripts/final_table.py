"""Ablation table for the report: all models on identical simulated episodes.

Usage: PAIEC_DRAW=pairs|distinct python scripts/final_table.py <n_per_fold> <out.csv>
Set ONLY=<substring> to run only the matching rows and merge them into an existing <out.csv>.
"""
import os, sys, time; sys.path.insert(0, '.')
import numpy as np, pandas as pd
from scripts.cv import evaluate, load, Factory, COLS
from paiec import models
load()
N = int(sys.argv[1]); OUT = sys.argv[2]
W = int(os.environ.get('W', 26))
Z = 'data/proc/emb_qwen7b_L21_z_b0z.npy'
Q = 'data/proc/emb_qwen7b_L21_z.npy'
v2 = dict(s_m=1.0, rho=0.73, rho_f=0.17, ell=0.12, var_scale=0.5, s_d_grid=(0.8, 1.6, 3.2))
v3 = dict(v2, rho_l=0.0, rho_le=0.5)
v6 = dict(v3, var_scale=0.35, rho_le=0.4, s_m=1.5, s_a_grid=(0.4, 0.8, 1.6))
v7 = dict(v6, zpath=Z, T=0.3, s_m_grid=(1.0, 2.0, 4.0), vs_lo=0.35, vs_hi=0.2)
v8 = dict(v7, m_mix=[(-0.6, 0.8, 1), (-0.6, 1.6, 1), (-0.6, 3.2, 1)])
rows = [
    ('Constant 0.5', Factory(models.Constant, p=0.5)),
    ('Group mean, Beta(1,1)', Factory(models.GroupMean, prior=0.5, strength=2.0)),
    ('Bayes, own labels, empirical prior (reference)', Factory(models.BayesOwn, pass_fold=True)),
    ('IRT (no item effects)', Factory(models.IRT, s_m=2.5, s_a=1.2, s_d=2.0, use_items=False)),
    ('IRT (independent item effects)', Factory(models.IRT, s_m=2.5, s_a=1.2, s_d=2.0)),
    ('GP-IRT, e5 RBF kernel, BMA (v1)', Factory(models.GPBMA2, s_m=2.0)),
    ('+ metadata-token kernel, tuned (v2)', Factory(models.GPBMA2, **v2)),
    ('+ linear kernel on centered embeddings (v3)', Factory(models.GPBMA6, **v3)),
    ('+ Qwen cross-benchmark prior', Factory(models.GPBMA7, zpath=Z, T=0.3, **v3)),
    ('+ Qwen-kernel ensemble member (v4)', models.EnsFactory7([{**v3, 'zpath': Z, 'T': 0.3}, {**v3, 'zpath': Z, 'T': 0.3, 'emb': Q}])),
    ('+ retuned for small groups (v6)', models.EnsFactory7([{**v6, 'zpath': Z, 'T': 0.3}, {**v6, 'zpath': Z, 'T': 0.3, 'emb': Q}])),
    ('+ s_m mixture, label-count variance (v7)', models.EnsFactory8([v7, {**v7, 'emb': Q}])),
    ('+ empirical prior on benchmark shift (v8, final)', models.EnsFactory11([v8, {**v8, 'emb': Q}])),
]
ONLY = os.environ.get('ONLY')
if ONLY:
    rows = [r for r in rows if ONLY in r[0]]
res = []
ref = None
for name, mk in rows:
    t = time.time()
    df = evaluate(mk, n_per_fold=N, workers=W)
    if ref is None:
        ref = df
    r = {c: df[c].mean() for c in COLS}
    r['model'] = name
    r['se_ALC'] = df.ALC.std() / np.sqrt(len(df))
    res.append(r)
    print(name, ' '.join(f'{c}={r[c]:.4f}' for c in COLS), f'({time.time()-t:.0f}s)', flush=True)
out = pd.DataFrame(res)[['model'] + COLS + ['se_ALC']]
if ONLY and os.path.exists(OUT):
    old = pd.read_csv(OUT)
    old['model'] = old.model.replace({'+ retuned for small groups (v6, final)': '+ retuned for small groups (v6)',
                                      '+ s_m mixture, label-count variance (v7, final)': '+ s_m mixture, label-count variance (v7)'})
    out = pd.concat([old[~old.model.isin(out.model)], out])
out.to_csv(OUT, index=False)

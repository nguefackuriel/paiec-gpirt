"""Cross-fold nearest-neighbour item difficulty: coverage and correlation."""
import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd, torch
from scripts.cv import load
c = load(); o = c['o']
o = o.assign(y=o.k / o.n)
o['pg'] = o.groupby('g').y.transform('mean')
o['res'] = o.y - o.pg
item = o.groupby('i').agg(res=('res', 'mean'), b=('b', 'first'), ng=('g', 'nunique'))
fold = c['b2fold']
item['fold'] = item.b.map(fold).to_numpy()
E = torch.tensor(np.load('data/proc/emb_e5small.npy').astype(np.float32), device='cuda')
ids = item.index.to_numpy()
best_sim = np.zeros(len(ids)); best_j = np.zeros(len(ids), int)
for f in range(5):
    q = np.flatnonzero(item.fold.to_numpy() == f); k = np.flatnonzero(item.fold.to_numpy() != f)
    Ek = E[ids[k]]
    for s in range(0, len(q), 4096):
        qq = q[s:s + 4096]
        sim = E[ids[qq]] @ Ek.T
        v, j = sim.max(1)
        best_sim[qq] = v.cpu().numpy(); best_j[qq] = k[j.cpu().numpy()]
item['nn_sim'] = best_sim; item['nn_res'] = item.res.to_numpy()[best_j]
item.to_parquet('data/proc/item_nn.parquet')
for th in [0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.99]:
    m = item.nn_sim >= th
    cov_b = item[m].b.nunique()
    cr = np.corrcoef(item.res[m], item.nn_res[m])[0, 1] if m.sum() > 10 else np.nan
    print(f'sim>={th}: items {m.mean():.3f} ({m.sum()}) benches {cov_b} corr {cr:.3f}')

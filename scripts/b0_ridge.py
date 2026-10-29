"""Can item representations predict absolute accuracy on unseen benchmarks? (budget-0 prior)"""
import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd
from scripts.cv import load
c = load(); o = c['o']
o = o.assign(y=o.k / o.n)
it = o.groupby('i').agg(y=('y', 'mean'), b=('b', 'first'))
it['fold'] = it.b.map(c['b2fold'])
# group-level brier evaluation rows
for emb in sys.argv[1].split(','):
    E = np.load(emb).astype(np.float32)
    X = E[it.index.to_numpy()]
    yl = np.log((it.y.clip(0.02, 0.98)) / (1 - it.y.clip(0.02, 0.98))).to_numpy()
    wts = 1.0 / it.groupby('b').y.transform('size').to_numpy()     # benches equal
    pred = np.zeros(len(it))
    for f in range(5):
        tr = (it.fold != f).to_numpy(); te = ~tr
        Xt, yt, wt = X[tr], yl[tr], wts[tr]
        mu = np.average(Xt, axis=0, weights=wt); Xc = Xt - mu
        lam = 1e3
        A = (Xc * wt[:, None]).T @ Xc + lam * np.eye(X.shape[1]) * wt.sum() / len(wt)
        beta = np.linalg.solve(A, (Xc * wt[:, None]).T @ (yt - np.average(yt, weights=wt)))
        pred[te] = np.average(yt, weights=wt) + (X[te] - mu) @ beta
    it['z'] = pred
    zfull = np.zeros(E.shape[0], np.float32); zfull[it.index.to_numpy()] = pred
    np.save(emb.replace('.npy', '_b0z.npy'), zfull)
    oo = o.merge(it[['z']], left_on='i', right_index=True)
    for T in [0.0, 0.2, 0.4, 0.6, 1.0]:
        p = 1 / (1 + np.exp(-T * oo.z))
        se = oo.k * (1 - p) ** 2 + (oo.n - oo.k) * p ** 2
        g = pd.DataFrame({'g': oo.g, 'b': oo.b, 'se': se, 'n': oo.n}).groupby('g').agg(se=('se', 'sum'), n=('n', 'sum'), b=('b', 'first'))
        br = (g.se / g.n).groupby(g.b).mean().mean()
        print(emb.split('/')[-1], 'T', T, 'B0 brier (bench-avg)', round(br, 4), flush=True)

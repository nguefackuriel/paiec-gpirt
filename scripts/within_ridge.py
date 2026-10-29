"""Within-benchmark item difficulty from representations: OOF ridge on benchmark-centered data.

Target: item mean logit minus its benchmark mean; features: representation minus benchmark mean.
Reports the mean within-benchmark Spearman correlation (OOF by benchmark fold) and saves
w.x scores (data/proc/<emb>_wz.npy) for use as an item-level prior.
"""
import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from scripts.cv import load

c = load(); o = c['o']
o = o.assign(y=o.k / o.n)
it = o.groupby('i').agg(y=('y', 'mean'), b=('b', 'first'), ng=('g', 'nunique'))
it['fold'] = it.b.map(c['b2fold'])
it['yl'] = np.log(it.y.clip(0.02, 0.98) / (1 - it.y.clip(0.02, 0.98)))
it['yc'] = it.yl - it.groupby('b').yl.transform('mean')
for emb in sys.argv[1].split(','):
    E = np.load(emb).astype(np.float32)
    X = E[it.index.to_numpy()]
    Xc = X - pd.DataFrame(X).groupby(it.b.to_numpy()).transform('mean').to_numpy()
    w = 1.0 / it.groupby('b').y.transform('size').to_numpy()
    for lam in [1e2, 1e3, 1e4]:
        score = np.zeros(len(it))
        for f in range(5):
            tr = (it.fold != f).to_numpy()
            A = (Xc[tr] * w[tr, None]).T @ Xc[tr] + lam * np.eye(X.shape[1]) * w[tr].sum() / tr.sum()
            beta = np.linalg.solve(A, (Xc[tr] * w[tr, None]).T @ it.yc.to_numpy()[tr])
            score[~tr] = X[~tr] @ beta
        it['s'] = score
        cors = pd.Series({b: spearmanr(d.s, d.yc).correlation for b, d in it.groupby('b') if d.s.std() > 0 and d.yc.std() > 0})
        print(emb.split('/')[-1], 'lam', lam, 'within-bench spearman mean', round(cors.mean(), 4), 'median', round(cors.median(), 4),
              'frac>0', round((cors > 0).mean(), 3), flush=True)
        if lam == 1e3:
            full = np.zeros(E.shape[0], np.float32); full[it.index.to_numpy()] = score
            np.save(emb.replace('.npy', '_wz.npy'), full)

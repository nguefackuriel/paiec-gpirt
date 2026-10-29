"""Can subject metadata predict a subject's accuracy on a benchmark it was never seen on? (out-of-fold ridge)"""
import sys, pickle; sys.path.insert(0, '.')
import numpy as np, pandas as pd
from scipy import sparse
from scripts.cv import load
c = load(); o = c['o']; G = c['groups'].set_index('g')
tok = pickle.load(open('data/proc/subject_tokens.pkl', 'rb'))
g = o.groupby('g').agg(n=('n', 'sum'), k=('k', 'sum'), b=('b', 'first')).loc[G.index]
g['fold'] = G.fold; g['ni'] = G.n_items
g['lg'] = np.log((g.k + .5) / (g.n - g.k + .5)); g['q'] = g.k / g.n
# tokens seen on at least 2 benchmarks
tb = {}
for s, b in zip(g.index, g.b):
    for t in tok[s]:
        tb.setdefault(t, set()).add(b)
voc = {t: j for j, t in enumerate(sorted(t for t, bs in tb.items() if len(bs) >= 2))}
rows, cols = [], []
for r, s in enumerate(g.index):
    for t in tok[s]:
        if t in voc:
            rows.append(r); cols.append(voc[t])
X = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(g), len(voc)))
print('vocab', len(voc), 'groups with any token', (X.sum(1) > 0).mean().round(3))
w = 1.0 / g.groupby('b').b.transform('size').to_numpy()
for lam in (1.0, 10.0, 100.0):
    pred = np.zeros(len(g))
    for f in range(5):
        tr = (g.fold != f).to_numpy(); te = ~tr
        Xt, yt, wt = X[tr], g.lg.to_numpy()[tr], w[tr]
        mu = np.average(yt, weights=wt)
        A = (Xt.T @ sparse.diags(wt) @ Xt).toarray() + lam * np.eye(X.shape[1]) * wt.mean()
        beta = np.linalg.solve(A, Xt.T @ (wt * (yt - mu)))
        pred[te] = mu + X[te] @ beta
    for mx in (200, 1000):
        m = (g.ni <= mx).to_numpy()
        q = g.q.to_numpy()[m]; wm = w[m]
        p = 1 / (1 + np.exp(-pred[m]))
        const = np.average(q, weights=wm)
        b_c = np.average(q - 2 * const * q + const ** 2, weights=wm)
        b_p = np.average(q - 2 * p * q + p ** 2, weights=wm)
        r = np.corrcoef(pred[m], g.lg.to_numpy()[m])[0, 1]
        print(f'lam {lam:6.1f} groups<={mx}: corr {r:+.3f}  constant-level Brier {b_c:.4f}  metadata-level Brier {b_p:.4f}')

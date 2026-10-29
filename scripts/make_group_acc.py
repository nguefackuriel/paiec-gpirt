"""Accuracy of every training group (one subject on one benchmark): data/group_acc.csv."""
import sys; sys.path.insert(0, '.')
from scripts.cv import load
c = load(); o = c['o']; G = c['groups'].set_index('g')
g = o.groupby('g').agg(n=('n', 'sum'), k=('k', 'sum'), b=('b', 'first')).loc[G.index]
g['n_items'] = G.n_items; g['fold'] = G.fold; g['q'] = g.k / g.n
g[['b', 'fold', 'n_items', 'q']].to_csv('data/group_acc.csv')
print(len(g), 'groups')

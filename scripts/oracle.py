"""Headroom: Brier of oracles in the current draw regime (group-mean oracle, item-rate oracle)."""
import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd
from scripts.cv import load, episodes
from paiec import sim

c = load()
rows = []
for f, s, ch in episodes(int(sys.argv[1]) if len(sys.argv) > 1 else 100):
    ev, aq = sim.build_episode(c['o'], ch, np.random.default_rng(s))
    n, k = ev.n.to_numpy(float), ev.k.to_numpy(float)
    g = ev.g.to_numpy()
    pbar = (np.bincount(g, k) / np.maximum(np.bincount(g, n), 1))[g]
    pitem = k / n
    for name, p in [('group-mean oracle', pbar), ('item-rate oracle', pitem)]:
        se = k * (1 - p) ** 2 + (n - k) * p ** 2
        br = np.bincount(g, se)[np.unique(g)] / np.bincount(g, n)[np.unique(g)]
        rows.append({'oracle': name, 'brier': br.mean()})   # mean over groups, then over episodes, as in sim
print(pd.DataFrame(rows).groupby('oracle').brier.mean().round(4))

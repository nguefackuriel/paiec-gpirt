"""Simulate episodes and store per-row stacking features + outcomes for each budget."""
import os, sys, time; sys.path.insert(0, '.')
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from scripts.cv import load, episodes
from paiec import sim, models, meta

BASE = dict(s_m=1.0, rho=0.73, rho_f=0.17, ell=0.12, var_scale=0.5, s_d_grid=(0.8, 1.6, 3.2), rho_l=0.0, rho_le=0.5)


def run(args):
    ep, f, s, ch = args
    c = load()
    ev, aq = sim.build_episode(c['o'], ch, np.random.default_rng(s))
    m = models.GPBMA6(**BASE)
    rows = []
    for b in sim.BUDGETS:
        d = meta.detail(m, ev, aq[aq['rank'] < b], b)
        d['g'] = ev.g.to_numpy(); d['n'] = ev.n.to_numpy(); d['k'] = ev.k.to_numpy()
        d['ep'] = ep; d['fold'] = f; d['budget'] = b
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


if __name__ == '__main__':
    N = int(sys.argv[1]); seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    eps = episodes(N, seed)
    t = time.time()
    with ProcessPoolExecutor(int(os.environ.get('W', 24))) as ex:
        res = list(ex.map(run, [(1000 * seed + j, f, s, ch) for j, (f, s, ch) in enumerate(eps)]))
    df = pd.concat(res, ignore_index=True)
    df.to_parquet(f'data/proc/meta_rows_s{seed}.parquet')
    print(df.shape, time.time() - t)

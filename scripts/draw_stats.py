import sys, time; sys.path.insert(0, '.')
import numpy as np, pandas as pd
from paiec import sim
obs = pd.read_parquet('data/proc/obs.parquet')
groups = sim.eligible_groups(obs)
obs = obs.merge(groups[['bench','subject_id']], on=['bench','subject_id'])
for c in ['bench','subject_id','item_id','interactors']: obs[c] = obs[c].astype(str)
obs.to_parquet('data/proc/obs_elig.parquet')
benches = np.array(sorted(groups.bench.unique()))
rng = np.random.default_rng(0)
folds = rng.permutation(len(benches)) % 5
pd.DataFrame({'bench': benches, 'fold': folds}).to_csv('data/proc/folds.csv', index=False)
gi = {k: set(v) for k, v in obs.groupby(['bench','subject_id']).item_id}
for f in range(5):
    gf = groups[groups.bench.isin(benches[folds == f])]
    for r in range(3):
        t = time.time()
        ch = sim.draw(obs, gf, np.random.default_rng(100 * f + r), group_items=gi)
        bc = pd.Series([b for b, _ in ch]).value_counts()
        items = len({(b, i) for b, s in ch for i in gi[(b, s)]})
        print(f'fold{f} draw{r}: groups={len(ch)} benches={len(bc)} items={items} top={bc.head(4).to_dict()} {time.time()-t:.1f}s')

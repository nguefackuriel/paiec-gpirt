"""Cross-benchmark relative ability per model name (out-of-fold for CV, full for runtime).

For each training group: logit of its accuracy minus the mean group logit of its benchmark;
averaged per normalized model name with shrinkage (sum / (count + k)). Writes
  data/proc/subj_ability_oof.npy   (indexed by subject code s; held-out benchmarks use other folds)
  submission/subject_ability.json  (name -> ability, all training benchmarks)
"""
import sys, json; sys.path.insert(0, '.')
import numpy as np, pandas as pd
from scripts.cv import load

K = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
c = load(); o = c['o']
cs = pd.read_parquet('data/proc/code_subjects.parquet')
S = pd.read_parquet('data/proc/subjects.parquet')
nm = cs.merge(S[['bench', 'subject_id', 'normalized_name']], on=['bench', 'subject_id'], how='left').set_index('s')
name = nm.normalized_name.fillna('').str.strip().str.lower()
g = o.groupby('g').agg(b=('b', 'first'), n=('n', 'sum'), k=('k', 'sum'))
g['lg'] = np.log((g.k + 0.5) / (g.n - g.k + 0.5))
g['rel'] = g.lg - g.groupby('b').lg.transform('mean')
g['name'] = name.reindex(g.index).fillna('').to_numpy()
g['fold'] = g.b.map(c['b2fold'])
named = g[g.name != '']


def table(d):
    # one value per (benchmark, name) so benchmarks with many groups of one model do not dominate
    bn = d.groupby(['b', 'name']).rel.mean().reset_index()
    a = bn.groupby('name').rel.agg(['sum', 'size'])
    return (a['sum'] / (a['size'] + K)).to_dict()


n_codes = int(cs.s.max()) + 1
oof = np.zeros(n_codes, np.float32)
for f in range(5):
    tab = table(named[named.fold != f])
    held = g[g.fold == f]
    oof[held.index.to_numpy()] = [tab.get(nm_, 0.0) for nm_ in held.name]
np.save('data/proc/subj_ability_oof.npy', oof)
json.dump({k: round(float(v), 5) for k, v in table(named).items()}, open('submission/subject_ability.json', 'w'))
cov = (oof[g.index.to_numpy()] != 0).mean()
r = np.corrcoef(oof[g.index.to_numpy()], g.rel)[0, 1]
print(f'coverage {cov:.3f}  corr(oof ability, held-out relative logit) {r:.3f}')

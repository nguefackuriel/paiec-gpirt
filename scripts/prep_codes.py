import sys; sys.path.insert(0, '.')
import pandas as pd
from paiec import sim
obs = pd.read_parquet('data/proc/obs_elig.parquet')
o = sim.encode(obs)
o.to_parquet('data/proc/codes.parquet')
m = pd.DataFrame({'s': o.s, 'b': o.b, 'i': o.i, 'r': o.r, 'bench': obs.bench.astype(str), 'subject_id': obs.subject_id.astype(str),
                  'item_id': obs.item_id.astype(str), 'interactors': obs.interactors.astype(str)})
m.drop_duplicates('s')[['s', 'b', 'bench', 'subject_id']].to_parquet('data/proc/code_subjects.parquet')
m.drop_duplicates('r')[['r', 'i', 'b', 'bench', 'item_id', 'interactors']].to_parquet('data/proc/code_rows.parquet')
print(o.shape, o.b.max() + 1, o.s.max() + 1, o.i.max() + 1, o.r.max() + 1)

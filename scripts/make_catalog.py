"""Write data/benchmarks_catalog.csv from the downloaded benchmarks.parquet files."""
import glob, json
import pandas as pd

m = json.load(open('data/mdb/download_manifest.json'))
rows = []
for d in m['dirs']:
    p = glob.glob(f'data/mdb/{d}/**/benchmarks.parquet', recursive=True)
    if not p:
        rows.append({'dir': d, 'missing': True}); continue
    for _, r in pd.read_parquet(p[0]).iterrows():
        rows.append({'dir': d, **{k: r[k] for k in ['benchmark_id', 'name', 'response_type', 'granularity', 'release',
                                                     'n_subjects', 'n_items', 'n_responses', 'max_trial'] if k in r}})
pd.DataFrame(rows).to_csv('data/benchmarks_catalog.csv', index=False)
print(len(rows), 'benchmarks')

"""Submit dist/<name>.zip to the PAIEC Codabench phase and print the submission id.

Usage: python scripts/submit.py dist/<name>.zip
Then:  python scripts/fetch_feedback.py <id>   (once the run has finished)
"""
import sys, requests
from pathlib import Path

PHASE = 29785
path = Path(sys.argv[1])
tok = Path.home().joinpath('.config/paiec/codabench-token').read_text().strip()
s = requests.Session(); s.headers['Authorization'] = f'Token {tok}'
API = 'https://www.codabench.org/api'

d = s.post(f'{API}/datasets/', json={'type': 'submission', 'name': path.name,
                                      'request_sassy_file_name': path.name,
                                      'file_size': path.stat().st_size}, timeout=60)
d.raise_for_status(); d = d.json()
up = requests.put(d['sassy_url'], data=path.read_bytes(), timeout=300,
                  headers={'Content-Type': 'application/zip', 'x-ms-blob-type': 'BlockBlob'})
up.raise_for_status()
s.put(f'{API}/datasets/completed/{d["key"]}/', timeout=60).raise_for_status()
r = s.post(f'{API}/submissions/', json={'phase': PHASE, 'data': d['key']}, timeout=60)
r.raise_for_status()
print('submitted', path.name, 'id', r.json()['id'])

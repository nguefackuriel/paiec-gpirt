"""Download a submission's scores and formative scoring log into data/feedback/<id>/."""
import sys, json, requests
from pathlib import Path
sid = sys.argv[1]
tok = Path.home().joinpath('.config/paiec/codabench-token').read_text().strip()
s = requests.Session(); s.headers['Authorization'] = f'Token {tok}'
d = s.get(f'https://www.codabench.org/api/submissions/{sid}/get_details/', timeout=60).json()
st = s.get(f'https://www.codabench.org/api/submissions/{sid}/', timeout=60).json()
out = Path(f'data/feedback/{sid}'); out.mkdir(parents=True, exist_ok=True)
for lg in d['logs']:
    if lg['data_file']:
        (out / (lg['name'] + '.txt')).write_bytes(requests.get(lg['data_file'], timeout=120).content)
print('status', st['status'], 'scores', [(x['column_key'], x['score']) for x in st.get('scores', [])])
txt = (out / 'scoring_stdout.txt').read_text() if (out / 'scoring_stdout.txt').exists() else ''
print(txt.split('Label budget: 0')[0])

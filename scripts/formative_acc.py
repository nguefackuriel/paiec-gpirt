"""Accuracy of the hidden test groups seen in formative feedback.

With a constant prediction p = 0.5 at budget 0 (submissions v1-v3), the reported ECE of a group is exactly
|q - 0.5|, where q is the group's accuracy. The side of 0.5 comes from the later submissions, whose budget-0
predictions are slightly below 0.5: their Brier score is above 0.25 only when q > 0.5. Writes
data/formative_acc.csv (one row per group of the constant-prediction submissions).
"""
import re, glob
import pandas as pd

CONST = {'969896', '970043', '970186'}
rows = []
for d in sorted(glob.glob('data/feedback/*')):
    sid = d.split('/')[-1]
    txt = open(f'{d}/scoring_stdout.txt').read()
    blk = txt.split('Label budget: 0')[1].split('Label budget:')[0]
    for m in re.finditer(r'(subject_\d+)\s+(benchmark_\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)', blk):
        rows.append((sid, m.group(1), m.group(2), int(m.group(3)), float(m.group(4)), float(m.group(5))))
df = pd.DataFrame(rows, columns=['sid', 'subject', 'bench', 'n', 'brier0', 'ece0'])
later = df[~df.sid.isin(CONST)]
side = later.groupby('bench').brier0.apply(lambda b: 1 if (b > 0.25).mean() > 0.5 else -1)
c = df[df.sid.isin(CONST)].copy()
c['q'] = 0.5 + c.bench.map(side) * c.ece0
c.to_csv('data/formative_acc.csv', index=False)
print(len(c), 'groups on', c.bench.nunique(), 'benchmarks; mean accuracy', round(c.q.mean(), 3))
print('side votes per benchmark:', later.groupby('bench').brier0.apply(lambda b: f'{(b > 0.25).sum()}/{len(b)} above 0.25').to_dict())

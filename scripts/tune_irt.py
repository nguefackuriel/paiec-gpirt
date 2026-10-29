import sys, time, itertools; sys.path.insert(0, '.')
from scripts.cv import evaluate, fmt, load, Factory
from paiec import models
load()
res = []
for s_m, s_a, s_d, m0 in itertools.product([1.0, 1.5, 2.5], [0.5, 0.8, 1.2], [0.7, 1.0, 1.5, 2.0], [0.0, 0.3]):
    df = evaluate(Factory(models.IRT, s_m=s_m, s_a=s_a, s_d=s_d, m0=m0), n_per_fold=6, workers=24)
    res.append((df.ALC.mean(), s_m, s_a, s_d, m0, fmt(df)))
    print(f"s_m={s_m} s_a={s_a} s_d={s_d} m0={m0}: {fmt(df)}", flush=True)
res.sort()
print('BEST'); [print(r) for r in res[:5]]

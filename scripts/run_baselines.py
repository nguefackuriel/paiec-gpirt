import sys, time; sys.path.insert(0, '.')
from scripts.cv import evaluate, fmt, load, Factory
from paiec import models
load()
for name, mk in [('const0.5', Factory(models.Constant, p=0.5)),
                 ('groupmean', Factory(models.GroupMean)),
                 ('irt_noitem', Factory(models.IRT, use_items=False)),
                 ('irt', Factory(models.IRT))]:
    t = time.time()
    df = evaluate(mk, n_per_fold=4)
    print(f"{name:12s}", fmt(df), f"({time.time()-t:.0f}s)", flush=True)

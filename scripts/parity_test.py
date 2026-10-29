import os, sys, time; sys.path.insert(0, '.')
os.environ["OMP_NUM_THREADS"] = "8"
import numpy as np, pandas as pd, importlib.util
from scripts.cv import load
from scripts.runtime_episode import to_input
from paiec import sim, models

c = load()
gf = c['groups'][c['groups'].fold == 4]
ch = sim.draw(gf, c['gi'], np.random.default_rng(7), limit=400)
ev, aq = sim.build_episode(c['o'], ch, np.random.default_rng(1))
print('groups', len(ch), 'eval rows', len(ev))
spec = importlib.util.spec_from_file_location("subm", "submission/model.py")
subm = importlib.util.module_from_spec(spec); spec.loader.exec_module(subm)
import json; cfg = json.load(open("submission/config.json")); ref_model = models.GPBMA6(rho_l=0.0, **cfg)
for b in (0, 3, 31):
    lab = aq[aq['rank'] < b]
    ref = ref_model.predict_batch(ev, lab)
    labeled = [[to_input(s, r), int(y)] for s, r, y in zip(lab.s, lab.r, lab.y)]
    sel = np.random.default_rng(0).choice(len(ev), min(300, len(ev)), replace=False)
    t = time.time()
    got = np.array([subm.predict(to_input(ev.s.iat[j], ev.r.iat[j]), labeled) for j in sel])
    dt = (time.time() - t) / len(sel)
    print(f'budget {b}: max|diff|={np.abs(got - ref[sel]).max():.2e}  mean pred={got.mean():.3f}  {dt*1000:.1f} ms/call')

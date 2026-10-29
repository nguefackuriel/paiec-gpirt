"""Parity of the v7 runtime (submission/model.py with Qwen) against the simulator ensemble."""
import os, sys, time, json; sys.path.insert(0, '.')
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np, importlib.util
from scripts.cv import load
from scripts.runtime_episode import to_input
from paiec import sim, models

# final-prior logits for every training item, as the runtime computes them
P = np.load('submission/qwen_prior.npz')
X = np.load('data/proc/emb_qwen7b_L21_z.npy').astype(np.float32)
np.save('data/proc/qwen_final_z.npy', (float(P['c']) + (X - P['mu']) @ P['beta']).astype(np.float32))

cfg = json.load(open('submission/config.json'))
base = {k: cfg[k] for k in ('s_m', 's_a_grid', 's_d_grid', 'rho', 'rho_f', 'ell', 'var_scale', 'rho_le')}
base.update(s_m_grid=cfg['s_m_grid'], vs_lo=cfg['var_scale_lo'], vs_hi=cfg['var_scale_hi'], n_lo=cfg['n_lo'])
ref_model = models.Ensemble([
    models.GPBMA8(zpath='data/proc/qwen_final_z.npy', T=cfg['T'], rho_l=0.0, **base),
    models.GPBMA8(zpath='data/proc/qwen_final_z.npy', T=cfg['T'], rho_l=0.0, emb='data/proc/emb_qwen7b_L21_z.npy', **base)])

c = load()
gf = c['groups'][c['groups'].fold == 4]
ch = sim.draw_pairs(gf, np.random.default_rng(3), subj_key=c['skey'])
ev, aq = sim.build_episode(c['o'], ch, np.random.default_rng(1))
print('groups', len(ch), 'eval rows', len(ev), flush=True)
spec = importlib.util.spec_from_file_location("subm", "submission/model.py")
subm = importlib.util.module_from_spec(spec); spec.loader.exec_module(subm)
print('qwen loaded:', subm.QWEN is not None, flush=True)
for b in (0, 3, 31):
    lab = aq[aq['rank'] < b]
    ref = ref_model.predict_batch(ev, lab)
    labeled = [[to_input(s, r), int(y)] for s, r, y in zip(lab.s, lab.r, lab.y)]
    sel = np.random.default_rng(0).choice(len(ev), min(150, len(ev)), replace=False)
    t = time.time()
    got = np.array([subm.predict(to_input(ev.s.iat[j], ev.r.iat[j]), labeled) for j in sel])
    dt = (time.time() - t) / len(sel)
    d = np.abs(got - ref[sel])
    print(f'budget {b}: max|diff|={d.max():.2e} mean|diff|={d.mean():.2e} mean pred={got.mean():.3f} {dt*1000:.1f} ms/call', flush=True)

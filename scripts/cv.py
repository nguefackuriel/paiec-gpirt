"""Cross-validated simulation of the competition protocol (leave-benchmarks-out)."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from paiec import sim

C = {}


def load():
    if not C:
        o = pd.read_parquet('data/proc/codes.parquet')
        cs = pd.read_parquet('data/proc/code_subjects.parquet')
        folds = pd.read_csv('data/proc/folds.csv')
        bfold = dict(zip(folds.bench, folds.fold))
        b2fold = cs.drop_duplicates('b').set_index('b').bench.map(bfold)
        groups = sim.group_table(o)
        groups['fold'] = groups.b.map(b2fold)
        gi = {g: set(v) for g, v in o.groupby('g').i}
        S = pd.read_parquet('data/proc/subjects.parquet')
        nm = cs.merge(S[['bench', 'subject_id', 'normalized_name']], on=['bench', 'subject_id'], how='left')
        skey = {int(s): (n if isinstance(n, str) and n else f'#{s}') for s, n in zip(nm.s, nm.normalized_name)}
        C.update(o=o, groups=groups, gi=gi, b2fold=b2fold, skey=skey)
    return C


DRAW_MODE = os.environ.get('PAIEC_DRAW', 'pairs')
MAX_GROUP_ITEMS = int(os.environ.get('PAIEC_MAX_GROUP', '1000'))   # e.g. 200 to mimic formative group sizes


def episodes(n_per_fold=4, seed=0):
    c = load()
    out = []
    for f in range(5):
        gf = c['groups'][(c['groups'].fold == f) & (c['groups'].n_items <= MAX_GROUP_ITEMS)]
        for r in range(n_per_fold):
            rng = np.random.default_rng(seed * 1000 + 10 * f + r)
            if DRAW_MODE == 'pairs':
                ch = sim.draw_pairs(gf, rng, subj_key=c['skey'])
            else:
                ch = sim.draw(gf, c['gi'], rng)
            out.append((f, int(rng.integers(1 << 31)), ch))
    return out


def _run(args):
    make_model, f, s, ch = args
    c = load()
    res = sim.run_episode(make_model(f), c['o'], ch, np.random.default_rng(s))
    r = res.mean(); r['fold'] = f; r['groups'] = len(res)
    return r


def evaluate(make_model, n_per_fold=4, seed=0, workers=16):
    eps = episodes(n_per_fold, seed)
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(_run, [(make_model, f, s, ch) for f, s, ch in eps]))
    return pd.DataFrame(rows)


COLS = ['B0', 'B1', 'B3', 'B7', 'B15', 'B31', 'ALC']


def fmt(df):
    return ' '.join(f"{c}={df[c].mean():.4f}" for c in COLS)


class Factory:
    """Picklable model factory: Factory(cls, **kw)(fold) -> cls(**kw) (fold passed if wanted)."""
    def __init__(self, cls, pass_fold=False, **kw):
        self.cls, self.kw, self.pass_fold = cls, kw, pass_fold

    def __call__(self, f):
        return self.cls(fold=f, **self.kw) if self.pass_fold else self.cls(**self.kw)


def _run_stream(args):
    make_model, make_policy, f, s, ch = args
    c = load()
    rng = np.random.default_rng(s)
    res = sim.run_stream_episode(make_model(f), c['o'], ch, rng, make_policy(f))
    r = res.mean(); r['fold'] = f; r['groups'] = len(res)
    return r


def evaluate_stream(make_model, make_policy, n_per_fold=4, seed=0, workers=16):
    eps = episodes(n_per_fold, seed)
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(_run_stream, [(make_model, make_policy, f, s, ch) for f, s, ch in eps]))
    return pd.DataFrame(rows)

"""Leave-fold-out stacking: learn p = f(features) on simulated rows, evaluate Brier ALC."""
import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd, lightgbm as lgb
from paiec import meta
from paiec.sim import BUDGETS, WEIGHTS

files = sys.argv[1].split(',')
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
df = df[df.n > 0]
WB = dict(zip(BUDGETS, WEIGHTS))
df['y'] = df.k / df.n
# metric weights: each (episode, group) equal; within group proportional to n; budget weights
gsum = df.groupby(['ep', 'g', 'budget']).n.transform('sum')
df['w'] = df.n / gsum * df.budget.map(WB)
FEATS = [f for f in meta.FEATURES if f not in ('len', 'mean_cos', 'max_cos', 'rate_b', 'k_s')]
df['logit_gp'] = np.log(df.p_gp.clip(1e-4, 1 - 1e-4) / (1 - df.p_gp.clip(1e-4, 1 - 1e-4)))
FEATS = FEATS + ['logit_gp']


def alc(d, p):
    se = d.k * (1 - p) ** 2 + (d.n - d.k) * p ** 2
    t = pd.DataFrame({'ep': d.ep, 'g': d.g, 'b': d.budget, 'se': se, 'n': d.n}).groupby(['ep', 'g', 'b'])[['se', 'n']].sum()
    br = (t.se / t.n).unstack('b')
    a = (br[list(BUDGETS)].to_numpy() @ WEIGHTS)
    per_ep = pd.Series(a, index=br.index).groupby(level=0).mean()
    return per_ep


params = dict(objective='regression', learning_rate=0.03, num_leaves=7, min_data_in_leaf=5000, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=10, verbose=-1, num_threads=24)
res = []
for f in sorted(df.fold.unique()):
    tr, te = df[df.fold != f], df[df.fold == f]
    m = lgb.train(params, lgb.Dataset(tr[FEATS], tr.y, weight=tr.w), num_boost_round=int(sys.argv[2]) if len(sys.argv) > 2 else 300)
    p = np.clip(m.predict(te[FEATS]), 0.001, 0.999)
    # budget-0 rows: keep 0.5 (no information)
    p = np.where(te.budget == 0, 0.5, p)
    # simple per-budget Platt recalibration of logit_gp
    from sklearn.linear_model import LogisticRegression
    pc = te.p_gp.to_numpy().copy()
    for b in BUDGETS[1:]:
        mt, me = (tr.budget == b).to_numpy(), (te.budget == b).to_numpy()
        X = np.stack([tr.logit_gp[mt], np.zeros(mt.sum())], 1)[:, :1]
        yy = np.concatenate([np.ones(mt.sum()), np.zeros(mt.sum())])
        ww = np.concatenate([tr.w[mt] * tr.y[mt], tr.w[mt] * (1 - tr.y[mt])])
        lr = LogisticRegression(C=1e4).fit(np.concatenate([X, X]), yy, sample_weight=ww)
        pc[me] = lr.predict_proba(te.logit_gp[me].to_numpy()[:, None])[:, 1]
        if f == 0: print('  budget', b, 'platt a,b', lr.coef_[0][0].round(3), lr.intercept_[0].round(3))
    a2 = alc(te, pc)
    print(f'   platt d={((a2 - alc(te, te.p_gp)).mean()):+.5f}', flush=True)
    a0, a1 = alc(te, te.p_gp), alc(te, p)
    res.append(pd.DataFrame({'base': a0, 'meta': a1}))
    print(f'fold {f}: base {a0.mean():.5f} meta {a1.mean():.5f} d={((a1 - a0).mean()):+.5f}', flush=True)
r = pd.concat(res); d = r.meta - r.base
print(f'ALL base {r.base.mean():.5f} meta {r.meta.mean():.5f} d={d.mean():+.5f}±{d.std()/np.sqrt(len(d)):.5f}')
imp = pd.Series(m.feature_importance('gain'), index=FEATS).sort_values(ascending=False)
print(imp.head(12).round(0).to_dict())

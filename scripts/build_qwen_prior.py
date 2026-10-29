"""Final cross-benchmark accuracy prior on Qwen hidden states (all training benchmarks).

Pipeline reproduced at runtime:
  h   = mean-pooled hidden_states[LAYER] of Qwen2.5-7B-Instruct over "Task:\n<content>\n<features>" (<=512 tokens)
  x   = (h - raw_mean) / raw_std, then L2-normalized
  z   = c + (x - mu) @ beta          (ridge, benchmarks weighted equally, lambda = 1e3)
  base logit offset = T * z
Writes data/proc/qwen_prior.npz.
"""
import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd
from scripts.cv import load

LAYER = 21
raw = np.load(f'data/proc/emb_qwen7b_L{LAYER}.npy').astype(np.float32)
raw_mean, raw_std = raw.mean(0), raw.std(0) + 1e-6
X = (raw - raw_mean) / raw_std
X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
c = load(); o = c['o']
o = o.assign(y=o.k / o.n)
it = o.groupby('i').agg(y=('y', 'mean'), b=('b', 'first'))
y = np.log(it.y.clip(0.02, 0.98) / (1 - it.y.clip(0.02, 0.98))).to_numpy()
w = 1.0 / it.groupby('b').y.transform('size').to_numpy()
Xt = X[it.index.to_numpy()]
mu = np.average(Xt, axis=0, weights=w); Xc = Xt - mu
ybar = np.average(y, weights=w)
A = (Xc * w[:, None]).T @ Xc + 1e3 * np.eye(X.shape[1]) * w.sum() / len(w)
beta = np.linalg.solve(A, (Xc * w[:, None]).T @ (y - ybar))
np.savez('data/proc/qwen_prior.npz', raw_mean=raw_mean, raw_std=raw_std, mu=mu.astype(np.float32),
         beta=beta.astype(np.float32), c=np.float32(ybar), layer=LAYER)
z = ybar + (Xt - mu) @ beta
print('saved; in-sample z std', z.std(), 'mean', z.mean())

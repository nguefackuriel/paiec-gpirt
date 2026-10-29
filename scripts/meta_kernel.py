"""Meta-learn an item kernel on training benchmarks.

For each training group (one subject on one benchmark) the standardized
outcome residuals r_i = (y_i - pbar) / sqrt(pbar (1 - pbar)) are modelled as a
GP over item embeddings:  r ~ N(0, a * k_W + s2 * I),  k_W(i, j) = exp(-|W e_i - W e_j|^2 / 2).
W, a, s2 are fit by maximizing the exact log marginal likelihood over random
item subsets of random groups (benchmarks sampled uniformly). One projection is
trained per CV fold (excluding that fold's benchmarks) plus one on all data.
"""
import sys, time; sys.path.insert(0, '.')
import numpy as np, pandas as pd, torch
from scripts.cv import load

DIM = int(sys.argv[1]) if len(sys.argv) > 1 else 64
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
EMB = sys.argv[3] if len(sys.argv) > 3 else 'data/proc/emb_e5small.npy'
TAG = sys.argv[4] if len(sys.argv) > 4 else 'e5s'
import os
dev = os.environ.get('DEV', 'cuda')
c = load(); o = c['o']
o = o.assign(y=o.k / o.n)
# one row per (group, item): mean over interactor variants
gi = o.groupby(['g', 'i'], as_index=False).agg(y=('y', 'mean'), b=('b', 'first'))
gi['pbar'] = gi.groupby('g').y.transform('mean')
gi = gi[(gi.pbar > 0.02) & (gi.pbar < 0.98)]
gi['r'] = (gi.y - gi.pbar) / np.sqrt(gi.pbar * (1 - gi.pbar))
E = torch.tensor(np.load(EMB).astype(np.float32), device=dev)
groups = {g: (d.i.to_numpy(), d.r.to_numpy(np.float32), int(d.b.iat[0])) for g, d in gi.groupby('g')}
bench_groups = {}
for g, (_, _, b) in groups.items():
    bench_groups.setdefault(b, []).append(g)
b2fold = c['b2fold']


def train(exclude_fold, seed=0, batch=16, n_items=96):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    benches = [b for b in bench_groups if b2fold[b] != exclude_fold]
    D = E.shape[1]
    W = torch.nn.Parameter(torch.randn(D, DIM, device=dev) * (1.0 / np.sqrt(D)) * 3)
    la = torch.nn.Parameter(torch.tensor(-1.0, device=dev)); ls = torch.nn.Parameter(torch.tensor(0.0, device=dev))
    opt = torch.optim.Adam([W, la, ls], lr=3e-3)
    hist = []
    for step in range(STEPS):
        loss = 0.0
        for _ in range(batch):
            b = benches[rng.integers(len(benches))]
            g = bench_groups[b][rng.integers(len(bench_groups[b]))]
            items, r, _ = groups[g]
            if len(items) > n_items:
                sel = rng.choice(len(items), n_items, replace=False); items, r = items[sel], r[sel]
            z = E[items] @ W
            K = la.exp() * torch.exp(-0.5 * torch.cdist(z, z) ** 2) + (ls.exp() + 1e-4) * torch.eye(len(items), device=dev)
            y = torch.tensor(r, device=dev)
            L = torch.linalg.cholesky(K)
            alpha = torch.cholesky_solve(y[:, None], L)
            nll = 0.5 * (y[:, None] * alpha).sum() + torch.log(torch.diagonal(L)).sum()
            loss = loss + nll / len(items)
        loss = loss / batch
        opt.zero_grad(); loss.backward(); opt.step()
        hist.append(loss.item())
        if step % 500 == 0 or step == STEPS - 1:
            print(f'fold {exclude_fold} step {step} nll/item {np.mean(hist[-200:]):.4f} a={la.exp().item():.3f} s2={ls.exp().item():.3f}', flush=True)
    return W.detach().cpu().numpy(), la.exp().item(), ls.exp().item()


out = {}
for f in list(range(5)) + [-1]:
    t = time.time()
    W, a, s2 = train(f)
    out[f] = dict(W=W, a=a, s2=s2)
    print('trained fold', f, f'{time.time()-t:.0f}s', flush=True)
np.save(f'data/proc/metak_{TAG}_d{DIM}.npy', out, allow_pickle=True)

"""Meta-learn a low-rank linear item metric for the within-benchmark difficulty regression.

Within each training group, standardized outcome residuals r_i are modelled as
  r ~ N(0, a * Zc M M^T Zc^T / tr + b * Zc Zc^T / tr + s2 * I),
where Zc are group-centred item embeddings. M (D x R), a, b, s2 are fit by maximizing
the exact Gaussian marginal likelihood over random item subsets of random groups
(benchmarks sampled uniformly). One M per CV fold (excluding the fold) plus one on all data.
Usage: meta_linear.py <emb.npy> <tag> [rank] [steps]
"""
import os, sys, time; sys.path.insert(0, '.')
import numpy as np, pandas as pd, torch
from scripts.cv import load

EMB, TAG = sys.argv[1], sys.argv[2]
R = int(sys.argv[3]) if len(sys.argv) > 3 else 32
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 3000
dev = os.environ.get('DEV', 'cuda')
c = load(); o = c['o']
o = o.assign(y=o.k / o.n)
gi = o.groupby(['g', 'i'], as_index=False).agg(y=('y', 'mean'), b=('b', 'first'))
gi['pbar'] = gi.groupby('g').y.transform('mean')
gi = gi[(gi.pbar > 0.02) & (gi.pbar < 0.98)]
gi['r'] = (gi.y - gi.pbar) / np.sqrt(gi.pbar * (1 - gi.pbar))
E = torch.tensor(np.load(EMB).astype(np.float32), device=dev)
groups = {g: (d.i.to_numpy(), d.r.to_numpy(np.float32), int(d.b.iat[0])) for g, d in gi.groupby('g') if len(d) >= 10}
bench_groups = {}
for g, (_, _, b) in groups.items():
    bench_groups.setdefault(b, []).append(g)
b2fold = c['b2fold']


def train(exclude_fold, seed=0, batch=16, n_items=48):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    benches = [b for b in bench_groups if b2fold[b] != exclude_fold]
    D = E.shape[1]
    M = torch.nn.Parameter(torch.randn(D, R, device=dev) * 0.01)
    la, lb, ls = (torch.nn.Parameter(torch.tensor(v, device=dev)) for v in (-1.0, -1.0, 0.0))
    opt = torch.optim.Adam([M, la, lb, ls], lr=3e-3)
    hist = []
    for step in range(STEPS):
        loss = 0.0
        for _ in range(batch):
            b = benches[rng.integers(len(benches))]
            g = bench_groups[b][rng.integers(len(bench_groups[b]))]
            items, r, _ = groups[g]
            if len(items) > n_items:
                sel = rng.choice(len(items), n_items, replace=False); items, r = items[sel], r[sel]
            Z = E[items]; Z = Z - Z.mean(0, keepdim=True)
            ZM = Z @ M
            Kl = ZM @ ZM.T; Ki = Z @ Z.T
            Kl = Kl / (torch.diagonal(Kl).mean() + 1e-6); Ki = Ki / (torch.diagonal(Ki).mean() + 1e-6)
            K = la.exp() * Kl + lb.exp() * Ki + (ls.exp() + 1e-3) * torch.eye(len(items), device=dev)
            y = torch.tensor(r, device=dev)
            L = torch.linalg.cholesky(K)
            alpha = torch.cholesky_solve(y[:, None], L)
            loss = loss + (0.5 * (y[:, None] * alpha).sum() + torch.log(torch.diagonal(L)).sum()) / len(items)
        loss = loss / batch
        opt.zero_grad(); loss.backward(); opt.step()
        hist.append(loss.item())
        if step % 500 == 0 or step == STEPS - 1:
            print(f'fold {exclude_fold} step {step} nll/item {np.mean(hist[-200:]):.4f} a={la.exp().item():.3f} '
                  f'b={lb.exp().item():.3f} s2={ls.exp().item():.3f}', flush=True)
    return dict(M=M.detach().cpu().numpy(), a=la.exp().item(), b=lb.exp().item(), s2=ls.exp().item())


out = {}
for f in list(range(5)) + [-1]:
    t = time.time(); out[f] = train(f); print('trained fold', f, f'{time.time()-t:.0f}s', flush=True)
np.save(f'data/proc/metalin_{TAG}_r{R}.npy', out, allow_pickle=True)

"""Figure: accuracy of training groups (realistic sizes and all sizes) and of the formative test groups."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

g = pd.read_csv("data/group_acc.csv")
f = pd.read_csv("data/formative_acc.csv")
bins = np.linspace(0, 1, 11)


def bench_weighted(d):
    # each benchmark counts once, as in the training of the prior
    w = 1.0 / d.groupby("b").q.transform("size")
    h, _ = np.histogram(d.q, bins, weights=w)
    return h / h.sum()


small = g[g.n_items <= 200]
series = [(bench_weighted(small), f"training, groups of 80-200 items ({small.b.nunique()} benchmarks)", "#2a78d6"),
          (bench_weighted(g), f"training, all groups ({g.b.nunique()} benchmarks)", "#8a8a85"),
          (np.histogram(f.q, bins)[0] / len(f), f"formative test groups ({len(f)} groups)", "#eb6834")]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(4.6, 2.8), dpi=200)
c = 0.5 * (bins[1:] + bins[:-1]); wd = 0.026
for k, (h, lab, col) in enumerate(series):
    ax.bar(c + (k - 1) * (wd + 0.004), h, width=wd, color=col, label=lab, zorder=3)
ax.set_xticks(bins[::2])
ax.set_xlabel("accuracy of the subject on the benchmark")
ax.set_ylabel("share of groups")
ax.grid(axis="y", color="#e6e5df", lw=0.6)
ax.tick_params(colors="#5f5e5a")
for s in ("left", "bottom"):
    ax.spines[s].set_color("#b5b4ad")
ax.set_ylim(0, 0.42)
ax.legend(frameon=False, loc="upper right", fontsize=7.5)
fig.tight_layout()
fig.savefig("report/group_accuracy.pdf"); fig.savefig("report/group_accuracy.png")
print("means:", round(small.groupby("b").q.mean().mean(), 3), round(g.groupby("b").q.mean().mean(), 3), round(f.q.mean(), 3))

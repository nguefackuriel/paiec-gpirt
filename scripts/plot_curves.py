"""Learning curves (Brier vs label budget) for the report, with two reference levels."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t = pd.read_csv("report/table_small.csv").set_index("model")
B = [0, 1, 3, 7, 15, 31]
x = np.log2(1 + np.array(B))
# group-mean oracle on the same episodes (scripts/oracle.py 200)
oracle = 0.1503
series = [("Group mean, Beta(1,1)", "Group mean, Beta(1,1)", "#2a78d6"),
          ("+ s_m mixture, label-count variance (v7)", "Before the centred shift prior", "#eb6834"),
          ("+ empirical prior on benchmark shift (v8, final)", "Final model", "#1baf7a")]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(4.8, 3.1), dpi=200)
ax.axhline(oracle, color="#8a8a85", lw=1, ls=(0, (3, 3)))
ax.text(x[-1], oracle - 0.0012, "group accuracy known", ha="right", va="top", color="#5f5e5a", fontsize=7.5)
y = t.loc["Bayes, own labels, empirical prior (reference)", [f"B{b}" for b in B]].to_numpy(float)
ax.plot(x, y, color="#8a8a85", lw=1.2, ls=(0, (1, 2)), label="Bayes, own labels only")
for key, lab, col in series:
    y = t.loc[key, [f"B{b}" for b in B]].to_numpy(float)
    ax.plot(x, y, color=col, lw=2, marker="o", ms=4.5, mec="white", mew=1, label=lab, zorder=3)
ax.set_xticks(x, [str(b) for b in B])
ax.set_xlabel("label budget per subject and benchmark")
ax.set_ylabel("Brier score (lower is better)")
ax.grid(axis="y", color="#e6e5df", lw=0.6)
ax.tick_params(colors="#5f5e5a")
for s in ("left", "bottom"):
    ax.spines[s].set_color("#b5b4ad")
ax.legend(frameon=False, loc="upper right", fontsize=7.5)
ax.set_ylim(0.145, 0.252)
fig.tight_layout()
fig.savefig("report/learning_curves.pdf"); fig.savefig("report/learning_curves.png")
print("saved")

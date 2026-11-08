import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
import drcc
from baselines import conf_q
C = {"Standard": "#2a78d6", "DRCC-affine": "#eb6834", "DRCC": "#1baf7a", "DRCC-safe": "#eda100", "Oracle": "#8a8a85"}
MK = {"Standard": "o", "DRCC-affine": "s", "DRCC": "^", "DRCC-safe": "D", "Oracle": None}
plt.rcParams.update({"font.size": 15, "axes.labelsize": 16, "legend.fontsize": 14, "xtick.labelsize": 14, "ytick.labelsize": 14,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8a85",
                     "axes.labelcolor": "#2a2a28", "xtick.color": "#4a4a46", "ytick.color": "#4a4a46"})

def style(ax):
    ax.grid(axis="y", color="#ecebe6", lw=0.9)

# ---------------- Fig 1: decomposition illustration
rng = np.random.default_rng(3)
R = rng.standard_normal(400); clean = 0.8 + 1.3 * rng.standard_normal(170)
pois = 0.8 + 1.3 * 0.5 + 0.12 * rng.standard_normal(30)          # dense band in the body
W = np.r_[clean, pois]
th, p = drcc.theta_drcc(R, W, 0.1, transport="affine", return_parts=True)
a, b = p["a"], p["b"]
edges, cw, exc, E = drcc._excess_scan(W, a + b * R)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.3))
bins = np.linspace(-4, 6, 41)
ax[0].hist(R, bins=bins, density=True, color="#b9b8b0", alpha=0.9, label="stale reference $R$")
ax[0].hist(W, bins=bins, density=True, histtype="step", lw=2.2, color="#2a78d6", label="window $W$")
ax[0].set_xlabel("score"); ax[0].set_ylabel("density"); ax[0].legend(frameon=False, loc="upper left"); style(ax[0])
inner = edges[1:-1]; cnt_r = np.histogram(a + b * R, bins=edges)[0] / len(R) * (len(W) - E)
mid = np.arange(len(cw))
ax[1].bar(mid, cw, color="#2a78d6", width=0.8, label="window count")
ax[1].bar(mid, exc, bottom=cw - exc, color="#eb6834", width=0.8, label="estimated excess")
ax[1].plot(mid, cnt_r, "k_", ms=14, mew=2.4, label="transported reference")
ax[1].set_xlabel("bin of the transported reference"); ax[1].set_ylabel("count"); ax[1].legend(frameon=False, loc="upper left")
ax[1].set_xticks([]); style(ax[1])
fig.tight_layout(); fig.savefig("fig_decomposition.pdf"); fig.savefig("fig_decomposition.png", dpi=150)

# ---------------- Fig 2: breakdown at eps = alpha
eps = np.round(np.arange(0, 0.305, 0.02), 2)
top = {"Standard": [907,855,784,674,499,138,78,50,37,31,23,16,16,15,13,11], "DRCC-affine": [907,856,835,844,848,813,833,843,865,844,857,838,841,847,847,859],
       "DRCC": [903,851,807,789,708,562,626,510,340,376,373,327,319,411,437,463], "DRCC-safe": [903,850,780,671,495,137,77,50,34,30,22,16,16,15,13,11],
       "Oracle": [907,896,905,896,901,903,894,902,893,900,901,893,897,892,893,905]}
bot = {"Standard": [100,101,103,106,109,110,112,116,119,123,125,126,131,133,139,143], "DRCC-affine": [101,102,101,96,95,96,93,93,96,95,95,92,98,101,104,109],
       "DRCC": [99,100,100,91,92,94,91,92,94,93,93,89,96,97,102,105], "DRCC-safe": [99,100,99,91,92,94,91,92,94,93,93,89,96,97,102,105],
       "Oracle": [100,96,98,96,98,99,95,97,96,98,100,94,99,93,98,99]}
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
for k in top:
    kw = dict(color=C[k], lw=2.4 if k != "Oracle" else 1.8, ls="--" if k == "Oracle" else "-", marker=MK[k], ms=6, label=k)
    ax[0].plot(eps, np.array(bot[k]) / 1000, **kw); ax[1].plot(eps, np.array(top[k]) / 1000, **kw)
ax[0].plot(eps, 0.1 / (1 - eps), color="#4a4a46", lw=1.6, ls=":", label="upper bound")
for x in ax:
    x.axvline(0.1, color="#c9c8c2", lw=1.2); x.set_xlabel(r"poison fraction $\varepsilon$"); style(x)
ax[0].set_ylabel("type-I error"); ax[1].set_ylabel("power"); ax[0].set_ylim(0.08, 0.16)
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.03), handlelength=1.6, columnspacing=1.0)
fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig("fig_breakdown.pdf"); fig.savefig("fig_breakdown.png", dpi=150)

# ---------------- Fig 3: white-box attacks
e2 = np.array([0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30])
lo = {"Standard": [99,102,109,108,116,121,138], "DRCC-affine": [105,124,141,138,170,231,403], "DRCC": [100,109,111,114,124,128,151], "DRCC-safe": [98,102,107,108,115,124,137]}
hi = {"Standard": [-51,-179,-382,-724,-784,-798,-787], "DRCC-affine": [-50,-168,-272,-289,-326,-310,-300], "DRCC": [-54,-185,-387,-485,-666,-654,-589], "DRCC-safe": [-53,-188,-392,-725,-785,-790,-787]}
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
for k in lo:
    kw = dict(color=C[k], lw=2.4, marker=MK[k], ms=6, label=k)
    ax[0].plot(e2, np.array(lo[k]) / 1000, **kw); ax[1].plot(e2, np.array(hi[k]) / 1000, **kw)
for x in ax:
    x.set_xlabel(r"poison fraction $\varepsilon$"); style(x)
ax[0].set_ylabel("type-I error"); ax[1].set_ylabel("power minus oracle power")
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.03), handlelength=1.6, columnspacing=1.0)
fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig("fig_whitebox.pdf"); fig.savefig("fig_whitebox.png", dpi=150)

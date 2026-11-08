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
top = {"Standard": [887,861,829,778,672,223,125,83,58,46,33,23,22,20,15,13],
       "DRCC-affine": [887,862,856,876,869,864,857,843,843,824,818,812,777,752,789,774],
       "DRCC": [886,861,843,859,825,670,517,394,96,129,149,169,98,81,89,150],
       "DRCC-safe": [886,861,829,778,671,223,125,80,53,44,33,23,21,19,14,12],
       "Oracle": [887,881,887,882,884,884,878,883,878,879,883,878,880,877,879,887]}
bot = {"Standard": [101,102,104,106,108,109,112,115,118,122,121,127,128,136,141,141],
       "DRCC-affine": [102,102,101,96,97,98,99,95,95,98,96,96,95,97,97,95],
       "DRCC": [100,102,100,95,96,98,99,95,95,97,94,95,94,96,96,94],
       "DRCC-safe": [100,102,100,95,96,98,99,95,95,97,94,95,94,96,96,94],
       "Oracle": [101,97,99,96,98,98,95,97,94,98,97,95,96,94,99,98]}
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
lo = {"Standard": [100,103,109,110,118,123,139], "DRCC-affine": [117,126,150,158,182,230,409], "DRCC": [103,108,116,118,127,136,156], "DRCC-safe": [100,103,107,110,118,123,140]}
hi = {"Standard": [-62,-206,-395,-746,-747,-756,-741], "DRCC-affine": [-12,-173,-356,-549,-557,-635,-707], "DRCC": [-60,-202,-406,-709,-737,-744,-748], "DRCC-safe": [-61,-210,-402,-744,-757,-754,-748]}
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

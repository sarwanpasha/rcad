import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
raw = """arxiv|0:111,103,101,101,100/738,958,882,737,973 1:115,103,103,103,105/703,935,858,703,949 2:110,101,100,100,99/707,913,825,707,950 3:113,105,103,103,102/695,895,791,695,922 4:112,101,98,98,101/702,907,821,700,937 5:118,112,103,103,106/623,883,791,622,917
news|0:104,99,97,97,93/19,404,236,19,462 1:112,107,103,103,99/27,399,217,27,461 2:110,102,98,98,99/24,436,259,24,460 3:112,105,99,99,99/24,350,234,24,385 4:107,99,97,97,97/7,263,170,7,313
creditcard|0:109,106,100,100,96/409,929,522,409,940 1:109,102,99,99,99/338,789,497,338,855 2:108,101,100,100,97/270,779,404,268,822 3:106,100,98,98,97/238,735,369,238,829 4:111,98,96,96,100/420,812,556,417,940 5:110,99,96,96,99/106,620,274,102,897 6:110,98,97,97,101/162,727,298,156,893 7:110,97,97,97,99/169,496,252,167,805
cifar|0:105,95,94,94,96/158,767,480,158,832 1:112,111,104,104,102/125,772,496,125,832 2:113,108,105,105,101/51,705,338,51,793 3:113,108,105,105,102/46,584,329,46,677 4:113,103,101,100,102/16,281,160,16,462 5:107,94,93,93,96/7,171,89,7,288
cicids|1:108,102,101,101,98/4,156,40,4,219 2:109,103,101,101,98/90,572,232,90,633 3:120,109,106,106,109/6,229,141,6,300 4:110,161,139,100,101/133,689,336,125,705 5:111,103,100,100,100/3,184,62,3,268 6:112,104,104,104,101/11,426,100,11,457 7:115,241,116,106,104/0,384,0,0,28
gas|0:104,104,101,101,95/146,768,332,145,930 2:111,189,101,89,99/70,522,58,58,709 3:113,148,95,92,104/162,391,171,151,599 5:108,143,120,94,97/17,238,22,16,235"""
names = dict(arxiv="arXiv", news="News", creditcard="Credit card", cifar="Degraded CIFAR-10", cicids="CIC-IDS2017", gas="Gas")
R = ["Standard", "DRCC-affine", "DRCC", "DRCC-safe", "Oracle"]
col = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8a85"]; mk = ["o", "s", "^", "D", None]; ls = ["-", "-", "-", "-", "--"]
plt.rcParams.update({"font.size": 15, "axes.labelsize": 16, "legend.fontsize": 15, "xtick.labelsize": 13, "ytick.labelsize": 13, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#9a9a95",
                     "axes.labelcolor": "#3a3a37", "xtick.color": "#5a5a55", "ytick.color": "#5a5a55"})
fig, ax = plt.subplots(2, 6, figsize=(20, 7.2), sharey="row")
for j, line in enumerate(raw.split("\n")):
    ds, rest = line.split("|"); P = []; I = []; Pw = []
    for tok in rest.split():
        p, v = tok.split(":"); a, b = v.split("/")
        P.append(int(p)); I.append([int(x) / 1000 for x in a.split(",")]); Pw.append([int(x) / 1000 for x in b.split(",")])
    I, Pw = np.array(I), np.array(Pw)
    for k in range(5):
        kw = dict(color=col[k], lw=2.6 if k < 4 else 1.8, ls=ls[k], marker=mk[k], ms=6.5, label=R[k], zorder=3 if k == 2 else 2)
        ax[0, j].plot(P, I[:, k], **kw); ax[1, j].plot(P, Pw[:, k], **kw)
    ax[0, j].axhline(0.10, color="#c9c8c2", lw=1, zorder=1)
    ax[0, j].text(0.02, 0.97, "(" + "abcdef"[j] + ")", transform=ax[0, j].transAxes, va="top", fontsize=15); ax[1, j].set_xlabel("period")
    for a in ax[:, j]:
        a.set_xticks(P); a.grid(axis="y", color="#ecebe6", lw=0.8)
ax[0, 0].set_ylabel("type-I error"); ax[1, 0].set_ylabel("power")
ax[0, 0].set_ylim(0.07, 0.30); ax[1, 0].set_ylim(-0.02, 1.0)
h, l = ax[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.02))
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig("timeline_streams.png", dpi=200); fig.savefig("timeline_streams.pdf")

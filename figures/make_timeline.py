import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
raw = """arxiv|0:113,107,97,97,102/740,959,884,734,973 1:114,103,100,100,104/712,927,894,711,948 2:110,96,94,94,99/708,937,856,707,950 3:114,99,96,96,102/694,906,825,692,922 4:112,99,96,96,101/700,908,850,700,937 5:116,103,96,96,103/622,902,786,622,918
news|0:102,89,88,88,90/21,418,280,21,462 1:111,107,97,97,100/28,446,242,28,464 2:110,103,98,98,99/24,448,294,24,460 3:112,98,95,95,99/24,376,278,24,387 4:107,95,94,93,97/7,294,210,7,312
creditcard|0:109,102,98,95,98/412,927,754,412,941 1:107,99,94,94,98/338,840,717,338,852 2:108,97,96,94,96/269,793,677,269,820 3:108,94,93,93,98/243,803,608,243,832 4:111,95,93,93,100/418,871,734,418,940 5:110,98,92,92,100/105,844,609,105,898 6:110,98,95,95,100/162,801,604,162,889 7:110,97,92,92,99/169,753,602,167,800
cifar|0:107,95,91,91,95/162,787,575,162,834 1:113,112,99,99,102/120,761,452,120,833 2:113,107,101,101,100/54,713,431,54,793 3:114,108,100,100,102/44,584,357,42,687 4:111,103,97,96,99/15,300,210,15,466 5:110,96,91,91,97/7,174,112,6,288
cicids|1:106,100,99,97,97/4,199,141,4,211 2:110,104,99,99,98/92,637,392,92,636 3:116,103,101,100,105/6,241,153,6,284 4:112,165,146,99,103/128,784,537,125,709 5:112,99,95,95,100/2,250,173,2,263 6:114,106,100,100,100/12,452,401,12,456 7:116,270,137,104,105/0,459,3,0,24
gas|0:106,96,92,92,93/149,852,690,147,925 2:109,209,104,85,98/71,710,239,67,695 3:112,159,121,96,101/165,459,274,158,598 5:106,156,131,92,95/16,336,45,16,255"""
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

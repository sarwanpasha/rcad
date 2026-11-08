"""Per-period type-I/power on every real stream (stale protocol, score mode, alpha 0.10) for the
timeline figure: clean windows and a body band attack (q=0.5) at eps 0.1 and 0.2."""
import glob, numpy as np, pandas as pd
cols = ["dataset", "protocol", "mode", "alpha", "window", "attack", "q", "eps", "rule", "typeI", "power"]
out = []
for f in sorted(glob.glob("results_drift/H_*.csv")):
    d = pd.read_csv(f, usecols=lambda c: c in cols, low_memory=False)
    d = d[(d["mode"] == "score") & (d.alpha == 0.10) & d.rule.isin(["standard", "drcc", "drcc_affine", "drcc_safe", "oracle", "aci"])]
    d = d[(d.attack == "none") | ((d.attack == "band") & (d.q == 0.5)) | (d.attack == "top")]
    out.append(d)
d = pd.concat(out)
t = d.groupby(["dataset", "protocol", "window", "attack", "eps", "rule"])[["typeI", "power"]].mean().reset_index()
import os; os.makedirs("results_drift/tables", exist_ok=True)
t.to_csv("results_drift/tables/timeline2_H.csv", index=False)
print(len(t))

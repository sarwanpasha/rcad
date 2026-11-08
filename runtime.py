"""Wall-clock cost of each calibration rule per window (seconds), by window size m = n_ref."""
import time, numpy as np, pandas as pd, rcad, drcc
from baselines import conf_q, theta_qtc, theta_weighted
rows = []
rng = np.random.default_rng(0)
for m in (100, 200, 500, 1000, 2000, 5000):
    reps = 30 if m <= 1000 else 10
    for rep in range(reps):
        R = rng.standard_normal(m); W = 0.5 + 1.2 * rng.standard_normal(m)
        k = int(0.1 * m); W[:k] = W.max() + rng.uniform(0.5, 3, k)
        RX = rng.standard_normal((m, 16)); WX = rng.standard_normal((m, 16)) + 0.3
        rules = {"standard": lambda: conf_q(W, .1), "naive_trim": lambda: rcad.theta_naive_trim(W, .1, .1),
                 "rcad_u": lambda: rcad.theta_rcad_u(W, R, .1, estimator="tail"), "qtc": lambda: theta_qtc(R, W, .1),
                 "weighted": lambda: theta_weighted(R, RX, WX, .1),
                 "drcc_affine": lambda: drcc.theta_drcc(R, W, .1, transport="affine"),
                 "drcc": lambda: drcc.theta_drcc(R, W, .1), "drcc_safe": lambda: max(drcc.theta_drcc(R, W, .1), conf_q(W, .1))}
        for r, f in rules.items():
            t0 = time.perf_counter(); f(); rows.append((m, r, time.perf_counter() - t0))
    print("m", m, flush=True)
d = pd.DataFrame(rows, columns=["m", "rule", "sec"])
import os; os.makedirs("results_drift", exist_ok=True)
d.to_csv("results_drift/runtime.csv", index=False)
print((d.groupby(["rule", "m"]).sec.median().unstack("m") * 1000).round(2).to_string())

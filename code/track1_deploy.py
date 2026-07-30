import os, sys, json, numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
import close_open as C
import rcad_v2 as R

ALPHA = 0.10
DELTAS = [0.0, 0.05, 0.10, 0.20]
SEEDS = list(range(10))
# per-dataset window config from design
CFG = {
    "mammography": dict(W=8, m_cal=200, n_ref=400, fit=700),
    "satellite":   dict(W=6, m_cal=150, n_ref=200, fit=380),
}

def build_windows(Xn, Xa, W, seed):
    """Permute normals iid into W equal windows; anomalies split across windows too."""
    rng = np.random.RandomState(seed)
    permn = rng.permutation(len(Xn)); Xn = Xn[permn]
    perma = rng.permutation(len(Xa)); Xa = Xa[perma]
    n_per = len(Xn) // W
    a_per = len(Xa) // W
    wins = []
    for w in range(W):
        nw = Xn[w*n_per:(w+1)*n_per]
        aw = Xa[w*a_per:(w+1)*a_per]
        wins.append((nw, aw))
    return wins

def apply_drift(win_norm, win_anom, v, sd, delta, w):
    shift = delta * w * sd * v
    return win_norm + shift, win_anom + shift

def score_split(Xfit, Xcal, Xref, Xte_n, Xte_a, detector, seed):
    det = C.make_scorer(detector, seed).fit(Xfit)
    return (det, det.score(Xcal), det.score(Xref),
            det.score(Xte_n), det.score(Xte_a))

def run_dataset(name, X, y, detector="isolation_forest"):
    cfg = CFG[name]; W = cfg["W"]
    Xn_all = X[y==0]; Xa_all = X[y==1]
    rows_gate = []      # stage-1a
    rows_grid = []      # stage-1b five-arm
    # --- STAGE 1a: standard arm, eps=0, no attack, per window; gate per delta ---
    gate_pass = {}
    for delta in DELTAS:
        cov_mat = np.zeros((W, len(SEEDS)))   # [window, seed]
        for si, seed in enumerate(SEEDS):
            wins = build_windows(Xn_all, Xa_all, W, seed)
            Xn_w0 = wins[0][0]
            v = PCA(1).fit(Xn_w0).components_[0]
            sd = float((Xn_w0 @ v).std())
            for w in range(W):
                nw, aw = wins[w]
                nw, aw = apply_drift(nw, aw, v, sd, delta, w)
                need = cfg["fit"] + cfg["m_cal"] + cfg["n_ref"]
                if len(nw) < need or len(aw) < 5:
                    cov_mat[w, si] = np.nan; continue
                rng = np.random.RandomState(seed*131 + w)
                pn = rng.permutation(len(nw))
                Xfit = nw[pn[:cfg["fit"]]]
                Xcal = nw[pn[cfg["fit"]:cfg["fit"]+cfg["m_cal"]]]
                Xref = nw[pn[cfg["fit"]+cfg["m_cal"]:cfg["fit"]+cfg["m_cal"]+cfg["n_ref"]]]
                det, s_cal, s_ref, s_te_n, s_te_a = score_split(Xfit, Xcal, Xref, nw, aw, detector, seed)
                thr = C.thr_standard(s_cal, ALPHA)
                cov = float(np.mean(s_te_n <= thr))
                cov_mat[w, si] = cov
        # aggregate cov over seeds -> per-window mean coverage
        win_cov = np.nanmean(cov_mat, axis=1)   # length W
        # gate on the [W, seeds]-collapsed structure: use per-(window) means as the cov[w][s] surrogate
        g = gate_1a(cov_mat, ALPHA)
        g.update(dataset=name, delta=delta)
        gate_pass[delta] = g["PASS"]
        rows_gate.append(g)
        print("[1a] %s delta=%.2f PASS=%s spread=%.3f thr=%.3f rho=%.2f p=%.3f bias=%.3f" %
              (name, delta, g["PASS"], g["spread"], g["thresh"], g["rho"], g["p"], g["mean_bias"]), flush=True)
    return rows_gate, gate_pass

def gate_1a(cov, alpha=0.10):
    # cov: array [W, S]; may contain nan
    W, S = cov.shape
    win = np.nanmean(cov, 1)                       # per-window mean coverage
    sd_w = np.nanstd(cov, 1, ddof=1)
    se_within = float(np.nanmean(sd_w) / np.sqrt(S))
    spread = float(np.nanmax(win) - np.nanmin(win))
    thresh = max(0.03, 4*se_within)
    rho, p = spearmanr(np.arange(W), win)
    trend = (abs(rho) >= 0.7) and (p < 0.05)
    bias = float(np.nanmean(win) - (1-alpha))
    return dict(spread=spread, thresh=float(thresh), se_within=se_within,
                rho=float(rho), p=float(p), mean_bias=bias,
                PASS=bool(spread <= thresh and not trend))

if __name__ == "__main__":
    DS = R.load_datasets(use_openml=False)
    all_gate = []
    passmap = {}
    for name in ("mammography","satellite"):
        X, y = DS[name]
        X = np.asarray(X, float); y = np.asarray(y).astype(int)
        rg, gp = run_dataset(name, X, y)
        all_gate += rg; passmap[name] = gp
    import csv
    os.makedirs("track1", exist_ok=True)
    with open("track1/stage1a_gate.csv","w",newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(all_gate[0].keys()))
        wcsv.writeheader()
        for r in all_gate: wcsv.writerow(r)
    json.dump(passmap, open("track1/passmap.json","w"), indent=2)
    print("STAGE1A_DONE", json.dumps(passmap))

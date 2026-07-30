import os, sys, json, csv, numpy as np
from sklearn.decomposition import PCA
import close_open as C
import rcad_v2 as R

ALPHA = 0.10
SEEDS = list(range(10))
EPS_TRUE = 0.10          # attack contamination injected into calibration
W_START = 3              # attack ONSET window; windows < W_START are clean (prior ref clean at onset)
CFG = {
    "mammography": dict(W=8, m_cal=200, n_ref=400, fit=700),
    "satellite":   dict(W=6, m_cal=150, n_ref=200, fit=380),
}

def windows(Xn, Xa, W, seed):
    rng = np.random.RandomState(seed)
    Xn = Xn[rng.permutation(len(Xn))]; Xa = Xa[rng.permutation(len(Xa))]
    npw = len(Xn)//W; apw = len(Xa)//W
    return [(Xn[w*npw:(w+1)*npw], Xa[w*apw:(w+1)*apw]) for w in range(W)]

def drift(Xn, Xa, v, sd, delta, w):
    sh = delta*w*sd*v
    return Xn+sh, Xa+sh

def cp_lo(k, n):
    lo, _ = R.clopper_pearson(int(k), int(n), alpha=0.10)
    return float(lo)

def run(name, X, y, deltas, detector="isolation_forest"):
    cfg = CFG[name]; W = cfg["W"]
    Xn_all = X[y==0]; Xa_all = X[y==1]
    rows = []
    for delta in deltas:
        for seed in SEEDS:
            wins = windows(Xn_all, Xa_all, W, seed)
            Xn_w0 = wins[0][0]
            v = PCA(1).fit(Xn_w0).components_[0]; sd = float((Xn_w0@v).std())
            # curated reference: window 0, clean (no attack), fixed
            nw0, aw0 = drift(wins[0][0], wins[0][1], v, sd, delta, 0)
            need = cfg["fit"]+cfg["m_cal"]+cfg["n_ref"]
            # evaluate windows w=1..W-1 (need a prior window)
            prev_ref = None; prev_det = None
            # first build detector + curated ref from window 0
            rng0 = np.random.RandomState(seed*777)
            p0 = rng0.permutation(len(nw0))
            if len(nw0) < need: 
                continue
            Xfit0 = nw0[p0[:cfg["fit"]]]
            det0 = C.make_scorer(detector, seed).fit(Xfit0)
            Xcur_ref = nw0[p0[cfg["fit"]:cfg["fit"]+cfg["n_ref"]]]
            s_curated_ref = det0.score(Xcur_ref)   # curated: fixed window-0 clean reference
            for w in range(1, W):
                nw, aw = drift(wins[w][0], wins[w][1], v, sd, delta, w)
                if len(nw) < need or len(aw) < 5: 
                    continue
                # detector refit per window on clean normals of this window (deployment: model updated)
                rng = np.random.RandomState(seed*131+w)
                pn = rng.permutation(len(nw))
                Xfit = nw[pn[:cfg["fit"]]]
                Xcal_clean = nw[pn[cfg["fit"]:cfg["fit"]+cfg["m_cal"]]]
                Xref_same  = nw[pn[cfg["fit"]+cfg["m_cal"]:cfg["fit"]+cfg["m_cal"]+cfg["n_ref"]]]
                det = C.make_scorer(detector, seed).fit(Xfit)
                # prior-window reference: clean normals of window w-1 (drifted at w-1)
                pnw, paw = drift(wins[w-1][0], wins[w-1][1], v, sd, delta, w-1)
                rngp = np.random.RandomState(seed*131+(w-1))
                pp = rngp.permutation(len(pnw))
                Xref_prior = pnw[pp[cfg["fit"]+cfg["m_cal"]:cfg["fit"]+cfg["m_cal"]+cfg["n_ref"]]] if len(pnw)>=need else Xref_same
                # inject attack contamination ONLY at/after onset window W_START
                attacked = (w >= W_START)
                k = int(np.floor(EPS_TRUE*cfg["m_cal"])) if attacked else 0
                pool = C.attack_pool(Xfit, aw, np.random.RandomState(seed*991+w))
                sp = det.score(pool); order = np.argsort(sp)
                Xinj = pool[order[-k:]] if k>0 else pool[:0]
                keep = np.random.RandomState(seed*53+w).permutation(cfg["m_cal"])[:cfg["m_cal"]-k]
                Xcal = np.vstack([Xcal_clean[keep], Xinj]) if k>0 else Xcal_clean
                s_cal = det.score(Xcal)
                s_te_n = det.score(nw); s_te_a = det.score(aw)
                # SAME-window reference shares the window-w contamination state (realistic: recent live data)
                def poison_ref(Xref_clean, aw_src, attacked_flag, rs):
                    if not attacked_flag: return Xref_clean
                    kk=int(np.floor(EPS_TRUE*len(Xref_clean)))
                    pl=C.attack_pool(Xfit, aw_src, np.random.RandomState(rs)); ss=det.score(pl)
                    inj=pl[np.argsort(ss)[-kk:]]
                    keepr=np.random.RandomState(rs+7).permutation(len(Xref_clean))[:len(Xref_clean)-kk]
                    return np.vstack([Xref_clean[keepr], inj])
                Xref_same_c  = poison_ref(Xref_same, aw, attacked, seed*313+w)
                prior_attacked = ((w-1) >= W_START)
                Xref_prior_c = poison_ref(Xref_prior, wins[w-1][1], prior_attacked, seed*317+w)
                s_ref_same  = det.score(Xref_same_c)
                s_ref_prior = det.score(Xref_prior_c)
                # eps estimates
                e_same,_,_  = C.eps_hat_ks(s_cal, s_ref_same)
                e_prior,_,_ = C.eps_hat_ks(s_cal, s_ref_prior)
                e_cur,_,_   = C.eps_hat_ks(s_cal, s_curated_ref)   # curated ref scored by det0 (approx)
                arms = {
                    "standard":     C.thr_standard(s_cal, ALPHA),
                    "robust_known": C.thr_robust(s_cal, ALPHA, EPS_TRUE),
                    "rcadu_same":   C.thr_robust(s_cal, ALPHA, max(e_same,1e-9)),
                    "rcadu_prior":  C.thr_robust(s_cal, ALPHA, max(e_prior,1e-9)),
                    "rcadu_curated":C.thr_robust(s_cal, ALPHA, max(e_cur,1e-9)),
                }
                ehat = {"standard":0.0,"robust_known":EPS_TRUE,"rcadu_same":e_same,
                        "rcadu_prior":e_prior,"rcadu_curated":e_cur}
                for arm, thr in arms.items():
                    cov = float(np.mean(s_te_n <= thr))
                    pw  = float(np.mean(s_te_a > thr))
                    rows.append(dict(dataset=name, delta=delta, seed=seed, window=w, attacked=int(attacked), arm=arm,
                        eps_hat=float(ehat[arm]), coverage=cov, power=pw,
                        cov_lcb90=cp_lo(int(np.sum(s_te_n<=thr)), len(s_te_n)),
                        pow_lcb90=cp_lo(int(np.sum(s_te_a>thr)), len(s_te_a)),
                        n_norm=len(s_te_n), n_anom=len(s_te_a)))
        print("[1b] %s delta=%.2f done rows=%d" % (name, delta, len(rows)), flush=True)
    return rows

if __name__ == "__main__":
    pm = json.load(open("track1/passmap.json"))
    DS = R.load_datasets(use_openml=False)
    allrows = []
    for name in ("mammography","satellite"):
        gp = pm.get(name, {})
        deltas = [float(d) for d,ok in gp.items() if ok]
        if not deltas:
            print("[1b] %s: no passing deltas, skipping grid" % name, flush=True); continue
        X, y = DS[name]; X=np.asarray(X,float); y=np.asarray(y).astype(int)
        allrows += run(name, X, y, sorted(deltas))
    os.makedirs("track1", exist_ok=True)
    if allrows:
        with open("track1/stage1b_grid_v2.csv","w",newline="") as f:
            wc = csv.DictWriter(f, fieldnames=list(allrows[0].keys())); wc.writeheader()
            for r in allrows: wc.writerow(r)
    print("STAGE1B_DONE rows=%d" % len(allrows))

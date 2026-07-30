import os
#!/usr/bin/env python
"""Closing the open items for the RCAD rebuttal.
T1  rank-displacement bound (sharp, model free)  -> closes 'bound is loose' + 'intermediate model'
E5  bound-saturating attacks (low / high / split) -> attainability of both ends
E7  RCAD-Auto, data driven epsilon                -> closes 'eps must be specified'
E8  learned + modern detectors (AE, ECOD)         -> closes 'detectors are classical'
E9  m sweep + smoothed conformal at eps=0         -> closes '+0.012 offset'
P1  Clopper-Pearson lower bounds on power         -> closes 'level cap not lower bound'
"""
import os, sys, time, json, math
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rcad_v2 as R
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest

OUT = os.path.join(HERE, "rcad_v3_results")
os.makedirs(OUT, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]

def _q(scores, alpha, eps, robust=True):
    cfg = R.RCADConfig(alpha=alpha, epsilon=eps, robust=robust)
    return R.RCAD(cfg)._trimmed_quantile(np.asarray(scores, float), alpha, eps)

def rank_bound(m, alpha, eps):
    """Sharp interval for RCAD coverage under arbitrary corruption of k=floor(eps*m)
    of the m calibration points. Returns (lo, hi, k, j_pos)."""
    k = int(np.floor(eps * m))
    nc = m - k
    if eps <= 0:
        j = np.ceil((m + 1) * (1 - alpha))
        return float(j / (m + 1)), float(j / (m + 1)), 0, j
    mt = m - 2 * k
    if 2 * k >= m - 2:
        return 0.0, 1.0, k, np.nan
    aadj = min(alpha * (1 + 2 * eps), 0.99)
    pos = (1.0 - aadj) * (mt - 1)
    lo = (pos + 1.0) / (nc + 1.0)
    hi = (pos + 1.0 + k) / (nc + 1.0)
    return float(min(lo, 1.0)), float(min(hi, 1.0)), k, pos

# ---------------- T1: numerical verification of the rank bound ----------------
def t1(n_rep=20000):
    rows = []
    rng = np.random.RandomState(0)
    grid = [(200, 0.10, e) for e in (0.0, 0.05, 0.10, 0.20, 0.30)]
    grid += [(m, 0.10, 0.10) for m in (100, 200, 500, 1000, 2000, 5000)]
    grid += [(200, 0.05, e) for e in (0.05, 0.10, 0.20)]
    for (m, alpha, eps) in grid:
        k = int(np.floor(eps * m))
        nc = m - k
        lo, hi, _, pos = rank_bound(m, alpha, eps)
        cov_lo, cov_hi, cov_mid = [], [], []
        for _ in range(n_rep):
            clean = rng.rand(nc)
            if k > 0:
                s_lo = np.concatenate([clean, -1.0 - rng.rand(k)])
                s_hi = np.concatenate([clean, 2.0 + rng.rand(k)])
                half = k // 2
                s_md = np.concatenate([clean, -1.0 - rng.rand(half),
                                       2.0 + rng.rand(k - half)])
            else:
                s_lo = s_hi = s_md = clean
            cov_lo.append(min(max(_q(s_lo, alpha, eps), 0.0), 1.0))
            cov_hi.append(min(max(_q(s_hi, alpha, eps), 0.0), 1.0))
            cov_mid.append(min(max(_q(s_md, alpha, eps), 0.0), 1.0))
        rows.append(dict(m=m, alpha=alpha, eps=eps, k=k,
                         bound_lo=lo, bound_hi=hi,
                         sim_all_low=float(np.mean(cov_lo)),
                         sim_all_high=float(np.mean(cov_hi)),
                         sim_split=float(np.mean(cov_mid)),
                         nominal=1 - alpha))
        print("m=%5d a=%.2f e=%.2f k=%3d  bound [%.4f, %.4f]  low %.4f  high %.4f  split %.4f"
              % (m, alpha, eps, k, lo, hi, rows[-1]['sim_all_low'],
                 rows[-1]['sim_all_high'], rows[-1]['sim_split']), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "t1_rank_bound.csv"), index=False)
    return df

# ---------------- scorers: classical, learned (AE), and modern (ECOD) --------
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

class SkScorer(object):
    def __init__(self, kind, seed):
        self.kind, self.seed = kind, seed
    def fit(self, X):
        if self.kind == "isolation_forest":
            self.m_ = IsolationForest(n_estimators=100, contamination=0.1,
                                      random_state=self.seed).fit(X)
        elif self.kind == "lof":
            self.m_ = LocalOutlierFactor(n_neighbors=20, contamination=0.1,
                                         novelty=True).fit(X)
        elif self.kind == "ocsvm":
            self.m_ = OneClassSVM(nu=0.1, kernel="rbf", gamma="scale").fit(X)
        else:
            raise ValueError(self.kind)
        return self
    def score(self, X):
        if self.kind == "lof":
            return -self.m_.score_samples(X)
        return -self.m_.decision_function(X)

class ECODScorer(object):
    """Unsupervised tail-probability detector (Li et al. 2022), numpy implementation."""
    def __init__(self, seed=0):
        self.seed = seed
    def fit(self, X):
        self.X_ = np.asarray(X, float)
        self.n_ = len(self.X_)
        self.sk_ = stats.skew(self.X_, axis=0)
        return self
    def _tail(self, X):
        X = np.asarray(X, float)
        n = self.n_
        left = np.empty_like(X); right = np.empty_like(X)
        for j in range(X.shape[1]):
            col = np.sort(self.X_[:, j])
            lo = np.searchsorted(col, X[:, j], side="right")
            left[:, j] = (lo + 1.0) / (n + 1.0)
            right[:, j] = (n - np.searchsorted(col, X[:, j], side="left") + 1.0) / (n + 1.0)
        return left, right
    def score(self, X):
        left, right = self._tail(X)
        eps = 1e-12
        Ol = -np.log(left + eps).sum(1)
        Or = -np.log(right + eps).sum(1)
        auto = np.where(self.sk_ < 0, -np.log(left + eps), -np.log(right + eps)).sum(1)
        return np.maximum(np.maximum(Ol, Or), auto)

class AEScorer(object):
    """Nonlinear MLP autoencoder trained with Adam, pure numpy. Score = recon error."""
    def __init__(self, seed=0, hidden=None, epochs=400, lr=3e-3, code=None):
        self.seed, self.hidden, self.epochs, self.lr, self.code = seed, hidden, epochs, lr, code
    def _init(self, d):
        rng = np.random.RandomState(self.seed)
        h = self.hidden or max(8, min(64, 2 * d))
        c = self.code or max(2, d // 4)
        def W(a, b):
            return rng.randn(a, b) * np.sqrt(2.0 / a)
        self.p = [W(d, h), np.zeros(h), W(h, c), np.zeros(c),
                  W(c, h), np.zeros(h), W(h, d), np.zeros(d)]
        self.m = [np.zeros_like(q) for q in self.p]
        self.v = [np.zeros_like(q) for q in self.p]
    def _fwd(self, X):
        W1, b1, W2, b2, W3, b3, W4, b4 = self.p
        z1 = X @ W1 + b1; a1 = np.tanh(z1)
        z2 = a1 @ W2 + b2; a2 = np.tanh(z2)
        z3 = a2 @ W3 + b3; a3 = np.tanh(z3)
        out = a3 @ W4 + b4
        return z1, a1, z2, a2, z3, a3, out
    def fit(self, X):
        X = np.asarray(X, float)
        n, d = X.shape
        self._init(d)
        t = 0
        rng = np.random.RandomState(self.seed + 1)
        bs = min(128, n)
        for ep in range(self.epochs):
            idx = rng.permutation(n)
            for s in range(0, n, bs):
                xb = X[idx[s:s + bs]]
                nb = len(xb)
                z1, a1, z2, a2, z3, a3, out = self._fwd(xb)
                g = 2.0 * (out - xb) / nb
                W1, b1, W2, b2, W3, b3, W4, b4 = self.p
                gW4 = a3.T @ g; gb4 = g.sum(0)
                g3 = (g @ W4.T) * (1 - a3 ** 2)
                gW3 = a2.T @ g3; gb3 = g3.sum(0)
                g2 = (g3 @ W3.T) * (1 - a2 ** 2)
                gW2 = a1.T @ g2; gb2 = g2.sum(0)
                g1 = (g2 @ W2.T) * (1 - a1 ** 2)
                gW1 = xb.T @ g1; gb1 = g1.sum(0)
                grads = [gW1, gb1, gW2, gb2, gW3, gb3, gW4, gb4]
                t += 1
                for i in range(8):
                    self.m[i] = 0.9 * self.m[i] + 0.1 * grads[i]
                    self.v[i] = 0.999 * self.v[i] + 0.001 * grads[i] ** 2
                    mh = self.m[i] / (1 - 0.9 ** t)
                    vh = self.v[i] / (1 - 0.999 ** t)
                    self.p[i] -= self.lr * mh / (np.sqrt(vh) + 1e-8)
        return self
    def score(self, X):
        X = np.asarray(X, float)
        out = self._fwd(X)[-1]
        return ((out - X) ** 2).mean(1)

def make_scorer(kind, seed):
    if kind == "ae":
        return AEScorer(seed=seed)
    if kind == "ecod":
        return ECODScorer(seed=seed)
    return SkScorer(kind, seed)

# ---------------- attack pool and the three saturating placements ------------
def attack_pool(X_norm, X_anom, rng, n_pool=1200):
    d = X_norm.shape[1]
    mu = X_norm.mean(0); sd = X_norm.std(0) + 1e-6
    parts = []
    for sc in (2.0, 4.0, 8.0, 16.0, 50.0):
        dirn = rng.randn(n_pool // 10, d)
        dirn /= (np.linalg.norm(dirn, axis=1, keepdims=True) + 1e-9)
        parts.append(mu + sc * sd * dirn + 0.5 * sd * rng.randn(n_pool // 10, d))
    parts.append(mu + 0.01 * sd * rng.randn(n_pool // 4, d))
    parts.append(mu + 0.25 * sd * rng.randn(n_pool // 4, d))
    if len(X_anom) > 0:
        idx = rng.choice(len(X_anom), size=min(n_pool // 4, 5 * len(X_anom)), replace=True)
        parts.append(X_anom[idx])
    idx2 = rng.choice(len(X_norm), size=n_pool // 4, replace=True)
    lam = rng.rand(n_pool // 4, 1) * 0.5
    parts.append(mu + lam * (X_norm[idx2] - mu))
    return np.vstack(parts)

def inject(mode, X_norm, X_anom, k, rng, defender):
    if k <= 0:
        return np.empty((0, X_norm.shape[1]))
    pool = attack_pool(X_norm, X_anom, rng)
    s = defender.score(pool)
    order = np.argsort(s)
    if mode == "sat_high":
        pick = order[-k:]
    elif mode == "sat_low":
        pick = order[:k]
    elif mode == "sat_split":
        h = k // 2
        pick = np.concatenate([order[:k - h], order[-h:]]) if h > 0 else order[:k]
    elif mode == "whitebox":
        return R.gen_adversarial_whitebox(X_norm, X_anom, k, rng, defender.m_)
    else:
        raise ValueError(mode)
    return pool[pick]

# ---------------- data driven epsilon (validity safe by construction) --------
def eps_hat_ks(cal_scores, ref_scores, conf=0.95):
    m, n = len(cal_scores), len(ref_scores)
    if m < 10 or n < 10:
        return 0.0, 0.0, 0.0
    D = float(stats.ks_2samp(cal_scores, ref_scores).statistic)
    c = float(np.sqrt(-0.5 * np.log((1.0 - conf) / 2.0)))
    crit = c * np.sqrt((m + n) / (m * n))
    return max(0.0, D - crit), D, float(crit)

# ---------------- threshold rules --------------------------------------------
def thr_standard(s, alpha):
    m = len(s)
    q = np.clip(np.ceil((m + 1) * (1 - alpha)) / m, 0.0, 1.0)
    return float(np.quantile(s, q))

def thr_robust(s, alpha, eps):
    return _q(s, alpha, eps, robust=True)

def smoothed_coverage(cal, test, alpha, rng):
    m = len(cal)
    cov = 0
    cal = np.sort(np.asarray(cal, float))
    for t in test:
        gt = m - np.searchsorted(cal, t, side="right")
        eqv = np.searchsorted(cal, t, side="right") - np.searchsorted(cal, t, side="left")
        u = rng.rand()
        p = (gt + u * (eqv + 1.0)) / (m + 1.0)
        cov += (p > alpha)
    return cov / float(len(test))

# ---------------- placement-controlled contamination -------------------------
def inject_placed(X_norm, X_anom, k, rng, defender, frac_below):
    """frac_below = c/(eps*m): share of poisoned points placed BELOW the threshold."""
    if k <= 0:
        return np.empty((0, X_norm.shape[1]))
    pool = attack_pool(X_norm, X_anom, rng)
    s = defender.score(pool)
    order = np.argsort(s)
    c = int(round(frac_below * k))
    c = max(0, min(k, c))
    lo = order[:c] if c > 0 else np.array([], dtype=int)
    hi = order[len(order) - (k - c):] if (k - c) > 0 else np.array([], dtype=int)
    return pool[np.concatenate([lo, hi]).astype(int)]

# ---------------- one generic trial ------------------------------------------
def trial(X, y, seed, alpha, eps_true, dsname, detector="isolation_forest",
          mode="whitebox", frac_below=1.0, m_cal_target=200,
          eps_assumed=None, want_smooth=False, want_auto=True):
    rng = np.random.RandomState(seed)
    try:
        Xtr, Xte, ytr, yte = R.train_test_split(
            X, y, test_size=0.30, random_state=seed, stratify=y)
    except ValueError:
        return []
    Xn = Xtr[ytr == 0]; Xa = Xtr[ytr == 1]
    if len(Xn) < 150 or len(Xa) < 5 or yte.sum() < 5:
        return []
    sc = StandardScaler().fit(Xn)
    Xn = sc.transform(Xn); Xa = sc.transform(Xa); Xt = sc.transform(Xte)
    n = len(Xn)
    m = int(min(m_cal_target, n - 60))
    if m < 40:
        return []
    perm = rng.permutation(n)
    Xcal0 = Xn[perm[:m]]
    rest = Xn[perm[m:]]
    n_ref = int(min(max(40, len(rest) // 4), 400))
    if len(rest) - n_ref < 50:
        n_ref = max(30, len(rest) // 5)
    Xref = rest[:n_ref]; Xfit = rest[n_ref:]
    if len(Xfit) < 50:
        return []
    det = make_scorer(detector, seed).fit(Xfit)
    k = int(np.floor(eps_true * m))
    if k > 0:
        if mode == "placed":
            Xinj = inject_placed(Xn, Xa, k, rng, det, frac_below)
        elif mode == "whitebox":
            Xinj = R.gen_adversarial_whitebox(Xn, Xa, k, rng, det.m_) \
                if hasattr(det, "m_") else inject_placed(Xn, Xa, k, rng, det, 0.0)
        else:
            Xinj = inject(mode, Xn, Xa, k, rng, det)
        keep = rng.permutation(m)[:m - k]
        Xcal = np.vstack([Xcal0[keep], Xinj])
    else:
        Xinj = np.empty((0, Xn.shape[1])); Xcal = Xcal0.copy()
    s_cal = det.score(Xcal); s_ref = det.score(Xref)
    s_clean = det.score(Xcal0); s_test = det.score(Xt)
    tau_clean = float(np.quantile(s_clean, 1 - alpha))
    if len(Xinj):
        s_inj = det.score(Xinj)
        beta = float(np.mean(s_inj > tau_clean))
        c_below = int(np.sum(s_inj <= tau_clean))
    else:
        beta = np.nan; c_below = 0
    ehat, Dks, crit = eps_hat_ks(s_cal, s_ref)
    ea = eps_true if eps_assumed is None else eps_assumed
    rules = {"Standard": ("std", 0.0), "Robust": ("rob", ea), "Oracle-eps": ("rob", eps_true)}
    if want_auto:
        rules["Auto"] = ("rob", ehat)
    nm = (yte == 0); am = (yte == 1)
    out = []
    for name, (kind, e) in rules.items():
        thr = thr_standard(s_cal, alpha) if kind == "std" else thr_robust(s_cal, alpha, max(e, 1e-9))
        cov = float(np.mean(s_test[nm] <= thr))
        pw = float(np.mean(s_test[am] > thr))
        lo_p, _ = R.clopper_pearson(int(np.sum(s_test[am] > thr)), int(am.sum()), alpha=0.10)
        lo_c, _ = R.clopper_pearson(int(np.sum(s_test[nm] <= thr)), int(nm.sum()), alpha=0.10)
        row = dict(dataset=dsname, seed=seed, alpha=alpha, eps_true=eps_true,
                   eps_assumed=e, detector=detector, mode=mode,
                   frac_below=frac_below, method=name, m_cal=m, n_ref=n_ref,
                   coverage=cov, cov_lcb90=float(lo_c), type1=1 - cov,
                   power=pw, power_lcb90=float(lo_p), threshold=thr,
                   eps_hat=ehat, ks_D=Dks, ks_crit=crit,
                   beta_tail=beta, c_below=c_below, k_inject=k,
                   n_norm=int(nm.sum()), n_anom=int(am.sum()))
        if want_smooth:
            row["cov_smoothed"] = smoothed_coverage(
                s_cal, s_test[nm], alpha, np.random.RandomState(seed + 77))
        out.append(row)
    return out

# ---------------- cached preparation, then the real trial --------------------
_PREP = {}

def prep(X, y, seed, dsname, detector, m_cal_target):
    key = (dsname, seed, detector, m_cal_target)
    if key in _PREP:
        return _PREP[key]
    rng = np.random.RandomState(seed)
    try:
        Xtr, Xte, ytr, yte = R.train_test_split(X, y, test_size=0.30,
                                                random_state=seed, stratify=y)
    except ValueError:
        _PREP[key] = None; return None
    Xn = Xtr[ytr == 0]; Xa = Xtr[ytr == 1]
    if len(Xn) < 150 or len(Xa) < 5 or yte.sum() < 5:
        _PREP[key] = None; return None
    sc = StandardScaler().fit(Xn)
    Xn = sc.transform(Xn); Xa = sc.transform(Xa); Xt = sc.transform(Xte)
    n = len(Xn); m = int(min(m_cal_target, n - 60))
    if m < 40:
        _PREP[key] = None; return None
    perm = rng.permutation(n)
    Xcal0 = Xn[perm[:m]]; rest = Xn[perm[m:]]
    n_ref = int(min(max(40, len(rest) // 4), 400))
    if len(rest) - n_ref < 50:
        n_ref = max(30, len(rest) // 5)
    Xref = rest[:n_ref]; Xfit = rest[n_ref:]
    if len(Xfit) < 50:
        _PREP[key] = None; return None
    det = make_scorer(detector, seed).fit(Xfit)
    d = dict(Xn=Xn, Xa=Xa, Xt=Xt, yte=yte, Xcal0=Xcal0, Xref=Xref, m=m,
             n_ref=n_ref, det=det, s_clean=det.score(Xcal0),
             s_ref=det.score(Xref), s_test=det.score(Xt))
    _PREP[key] = d
    return d

def trial(X, y, seed, alpha, eps_true, dsname, detector="isolation_forest",
          mode="whitebox", frac_below=1.0, m_cal_target=200,
          eps_assumed=None, want_smooth=False, want_auto=True):
    P = prep(X, y, seed, dsname, detector, m_cal_target)
    if P is None:
        return []
    det = P["det"]; m = P["m"]; Xn = P["Xn"]; Xa = P["Xa"]
    rng = np.random.RandomState(int(seed * 7919 + eps_true * 10007 + frac_below * 101) % (2 ** 31 - 1))
    k = int(np.floor(eps_true * m))
    if k > 0:
        if mode == "placed":
            Xinj = inject_placed(Xn, Xa, k, rng, det, frac_below)
        elif mode == "whitebox" and hasattr(det, "m_"):
            Xinj = R.gen_adversarial_whitebox(Xn, Xa, k, rng, det.m_)
        else:
            Xinj = inject_placed(Xn, Xa, k, rng, det, 0.0)
        keep = rng.permutation(m)[:m - k]
        Xcal = np.vstack([P["Xcal0"][keep], Xinj])
    else:
        Xinj = np.empty((0, Xn.shape[1])); Xcal = P["Xcal0"].copy()
    s_cal = det.score(Xcal); s_test = P["s_test"]
    tau_clean = float(np.quantile(P["s_clean"], 1 - alpha))
    if len(Xinj):
        s_inj = det.score(Xinj)
        beta = float(np.mean(s_inj > tau_clean)); c_below = int(np.sum(s_inj <= tau_clean))
    else:
        beta = np.nan; c_below = 0
    ehat, Dks, crit = eps_hat_combo(s_cal, P["s_ref"], alpha)
    ea = eps_true if eps_assumed is None else eps_assumed
    rules = [("Standard", "std", 0.0), ("Robust", "rob", ea)]
    if want_auto:
        rules.append(("Auto", "rob", ehat))
    yte = P["yte"]; nm = (yte == 0); am = (yte == 1)
    out = []
    for name, kind, e in rules:
        thr = thr_standard(s_cal, alpha) if kind == "std" else thr_robust(s_cal, alpha, max(e, 1e-9))
        kc = int(np.sum(s_test[nm] <= thr)); kp = int(np.sum(s_test[am] > thr))
        cov = kc / float(nm.sum()); pw = kp / float(am.sum())
        lo_c, _ = R.clopper_pearson(kc, int(nm.sum()), alpha=0.10)
        lo_p, _ = R.clopper_pearson(kp, int(am.sum()), alpha=0.10)
        row = dict(dataset=dsname, seed=seed, alpha=alpha, eps_true=eps_true,
                   eps_assumed=e, detector=detector, mode=mode,
                   frac_below=frac_below, method=name, m_cal=m, n_ref=P["n_ref"],
                   coverage=cov, cov_lcb90=float(lo_c), type1=1 - cov,
                   power=pw, power_lcb90=float(lo_p), threshold=float(thr),
                   eps_hat=ehat, ks_D=Dks, ks_crit=crit, beta_tail=beta,
                   c_below=c_below, k_inject=k, n_norm=int(nm.sum()),
                   n_anom=int(am.sum()))
        if want_smooth:
            row["cov_smoothed"] = smoothed_coverage(s_cal, s_test[nm], alpha,
                                                    np.random.RandomState(seed + 77))
        out.append(row)
    return out

def _run(DS, jobs, tag):
    rows = []; t0 = time.time(); tot = len(jobs)
    for i, kw in enumerate(jobs):
        ds = kw.pop("ds")
        X, y = DS[ds]
        try:
            rows.extend(trial(X, y, dsname=ds, **kw))
        except Exception as e:
            print("  [skip] %s %s: %s: %s" % (ds, kw, type(e).__name__, e), flush=True)
        if (i + 1) % 50 == 0 or i + 1 == tot:
            el = time.time() - t0
            print("  [%s] %d/%d  %.1fs elapsed, ETA %.1fs" %
                  (tag, i + 1, tot, el, el / (i + 1) * (tot - i - 1)), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "%s_raw.csv" % tag), index=False)
    print("[%s] %d rows -> %s_raw.csv" % (tag.upper(), len(df), tag), flush=True)
    return df

def e5(DS):
    jobs = []
    for ds in DS:
        for s in SEEDS:
            for e in (0.05, 0.10, 0.20):
                for f in (0.0, 0.25, 0.50, 0.75, 1.0):
                    jobs.append(dict(ds=ds, seed=s, alpha=0.10, eps_true=e,
                                     detector="isolation_forest", mode="placed",
                                     frac_below=f, want_auto=False))
    return _run(DS, jobs, "e5")

def e7(DS):
    jobs = []
    for ds in DS:
        for s in SEEDS:
            for det in ("isolation_forest", "lof", "ocsvm"):
                for e in (0.0, 0.05, 0.10, 0.20, 0.30):
                    for md, f in (("whitebox", 1.0), ("placed", 1.0)):
                        jobs.append(dict(ds=ds, seed=s, alpha=0.10, eps_true=e,
                                         detector=det, mode=md, frac_below=f,
                                         want_auto=True))
    return _run(DS, jobs, "e7")

def e8(DS):
    jobs = []
    for ds in DS:
        for s in SEEDS:
            for det in ("ae", "ecod"):
                for e in (0.0, 0.05, 0.10, 0.20, 0.30):
                    jobs.append(dict(ds=ds, seed=s, alpha=0.10, eps_true=e,
                                     detector=det, mode="whitebox",
                                     frac_below=1.0, want_auto=True))
    return _run(DS, jobs, "e8")

def e9(DS):
    jobs = []
    for ds in DS:
        for s in SEEDS:
            for mc in (100, 200, 400, 800, 1600, 3200):
                jobs.append(dict(ds=ds, seed=s, alpha=0.10, eps_true=0.0,
                                 detector="isolation_forest", mode="whitebox",
                                 m_cal_target=mc, want_smooth=True,
                                 want_auto=False))
    return _run(DS, jobs, "e9")

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--exp", default="all")
    a = ap.parse_args()
    print("loading datasets", flush=True)
    DS = R.load_datasets(use_openml=False)
    stats_rows = []
    for k, (X, y) in DS.items():
        stats_rows.append(dict(dataset=k, n=len(X), d=X.shape[1],
                               anomaly_rate=float(np.mean(y)), n_anom=int(y.sum())))
        print("  %-14s n=%6d d=%3d rate=%.4f" % (k, len(X), X.shape[1], np.mean(y)), flush=True)
    pd.DataFrame(stats_rows).to_csv(os.path.join(OUT, "dataset_stats.csv"), index=False)
    todo = ["t1", "e5", "e9", "e7", "e8"] if a.exp == "all" else a.exp.split(",")
    for t in todo:
        print("\n===== %s =====" % t.upper(), flush=True)
        t0 = time.time()
        if t == "t1":
            t1()
        else:
            {"e5": e5, "e7": e7, "e8": e8, "e9": e9}[t](DS)
        print("  %s done in %.1f min" % (t, (time.time() - t0) / 60), flush=True)


def eps_hat_tail(cal, ref, alpha, z=1.2816):
    """Upper-tail excess-mass estimator. Under upper-tail contamination the share of
    calibration scores above the clean (1-alpha) reference quantile is (1-eps)*a0 + eps,
    so eps_hat = (p - a0)/(1 - a0). One-sided 90% floor subtracted, so it under-estimates."""
    cal = np.asarray(cal, float); ref = np.asarray(ref, float)
    m, nr = len(cal), len(ref)
    if m < 20 or nr < 20:
        return 0.0
    tau = float(np.quantile(ref, 1 - alpha))
    a0 = float(np.mean(ref > tau))
    p = float(np.mean(cal > tau))
    if a0 >= 0.999:
        return 0.0
    se = np.sqrt(max(p * (1 - p), 1e-9) / m + max(a0 * (1 - a0), 1e-9) / nr) / (1 - a0)
    return float(max(0.0, (p - a0) / (1 - a0) - z * se))


def eps_hat_combo(cal, ref, alpha):
    ek, D, crit = eps_hat_ks(cal, ref)
    et = eps_hat_tail(cal, ref, alpha)
    return max(ek, et), D, crit

if __name__ == "__main__":
    main()

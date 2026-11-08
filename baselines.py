"""Calibration rules compared in the drift study.  All return a threshold on the
score scale; a point is flagged anomalous when score > threshold."""
import math, os
import numpy as np
from scipy.stats import theilslopes
from sklearn.linear_model import LogisticRegression

import rcad
from drcc import theta_drcc


def conf_q(s, alpha):
    s = np.sort(np.asarray(s)); n = s.size
    return s[int(np.clip(math.ceil((1 - alpha) * (n + 1)), 1, n)) - 1]


def weighted_conf_q(scores, w, w_test, alpha):
    """Weighted split-conformal quantile (Tibshirani et al. 2019)."""
    o = np.argsort(scores); s, ww = scores[o], w[o]
    p = np.cumsum(ww) / (np.sum(ww) + w_test)
    j = np.searchsorted(p, 1 - alpha)
    return s[j] if j < s.size else np.inf


def theta_weighted(ref_scores, ref_X, win_X, alpha):
    """Covariate-shift conformal: reweight the stale clean reference by an
    estimated likelihood ratio new/old, learned from reference vs window features.
    The window is poisoned, so the ratio is estimated from poisoned data."""
    Z = np.vstack([ref_X, win_X]); lab = np.r_[np.zeros(len(ref_X)), np.ones(len(win_X))]
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(Z, lab)
    odds = lambda Z: (lambda p: p / (1 - p))(np.clip(clf.predict_proba(Z)[:, 1], 1e-4, 1 - 1e-4))
    c = len(ref_X) / len(win_X)
    w = odds(ref_X) * c
    w_test = float(np.median(odds(win_X) * c))       # a typical test point comes from the window law
    return weighted_conf_q(ref_scores, w, w_test, alpha)


def theta_qtc(ref, win, alpha, p_lo=0.10, p_hi=0.70, npts=25):
    """Transport only: affine quantile map from body levels, no excess removal."""
    ps = np.linspace(p_lo, p_hi, npts)
    b, a, _, _ = theilslopes(np.quantile(win, ps), np.quantile(ref, ps))
    return a + max(b, 1e-6) * conf_q(ref, alpha)


# ---------------------------------------------------------------- drift-aware conformal baselines
def theta_aci(R, W, alpha, gamma=0.01):
    """Adaptive conformal inference (Gibbs & Candes 2021) run over the window in arrival
    order.  The calibration set is a sliding buffer that starts as the stale reference;
    the miscoverage feedback is the only one available without labels: whether each new
    (possibly poisoned) point exceeds the current threshold."""
    buf = list(np.asarray(R, float)); n = len(buf); a = alpha
    for w in np.asarray(W, float):
        th = conf_q(buf[-n:], float(np.clip(a, 1e-3, 0.999)))
        a += gamma * (alpha - float(w > th))
        buf.append(w)
    return conf_q(buf[-n:], float(np.clip(a, 1e-3, 0.999)))


def theta_nexcp(R, W, alpha, rho=0.99):
    """Non-exchangeable conformal (Barber et al. 2023): weighted quantile of reference and
    window with weights rho^age (the newest window point has age 0)."""
    s = np.r_[np.asarray(W, float)[::-1], np.asarray(R, float)]
    w = rho ** np.arange(s.size)
    return weighted_conf_q(s, w, 1.0, alpha)


def _kl_level(alpha, rho):
    """Corrected level of f-divergence robust conformal (Cauchois et al. 2024), KL ball:
    the smallest beta with g(beta) >= 1 - alpha, g(beta) = inf{z: beta f(z/beta) +
    (1-beta) f((1-z)/(1-beta)) <= rho}, f(t) = t log t."""
    def div(b, z):
        out = 0.0
        for p, q in ((z, b), (1 - z, 1 - b)):
            if p > 0:
                out += p * math.log(p / q)
        return out
    def g(b):
        lo, hi = 0.0, b                                # div(b, z) decreases in z on [0, b]
        if div(b, 0.0) <= rho:
            return 0.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if div(b, mid) <= rho: hi = mid
            else: lo = mid
        return hi
    lo, hi = 1 - alpha, 1 - 1e-9
    if g(hi) < 1 - alpha:
        return 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if g(mid) >= 1 - alpha: hi = mid
        else: lo = mid
    return hi


def _kl_hist(R, W, bins=20):
    e = np.unique(np.quantile(np.r_[R, W], np.linspace(0, 1, bins + 1)))
    p = np.histogram(W, e)[0] + 0.5; q = np.histogram(R, e)[0] + 0.5
    p, q = p / p.sum(), q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def theta_robust_kl(R, W, alpha, rho=None):
    """Shift-robust conformal from the clean stale reference: reference quantile at the
    level that covers every test law within KL radius rho; rho estimated from (R, W)."""
    rho = _kl_hist(R, W) if rho is None else rho
    beta = _kl_level(alpha, max(rho, 1e-6))
    if beta >= 1.0:
        return np.inf
    return conf_q(R, 1 - beta)


def theta_mad_filter(W, alpha, k=3.0):
    """Clean-then-calibrate: drop window scores farther than k robust SDs from the median."""
    W = np.asarray(W, float); med = np.median(W); mad = 1.4826 * np.median(np.abs(W - med)) + 1e-12
    keep = W[np.abs(W - med) <= k * mad]
    return conf_q(keep if keep.size >= 20 else W, alpha)


# ---------------------------------------------------------------- contamination-robust baselines
def theta_cdc(R, W, alpha, lam=0.5, eta=0.05, slack=True):
    """Contamination-corrected calibration (CDC, Balachandran 2026): conformal p-values of
    the window against the stale reserve R, an upper Storey estimate of the contaminated
    fraction, and the window quantile at level 1 - alpha (1 - pi_up) + eps_n.  slack=False
    drops the two finite-sample terms (plug-in variant)."""
    R = np.sort(np.asarray(R, float)); W = np.asarray(W, float); n, m = R.size, W.size
    p = (1.0 + (n - np.searchsorted(R, W, side="left"))) / (n + 1.0)
    pi_l = 1.0 - np.sum(p > lam) / (m * (1.0 - lam))
    h = math.sqrt(math.log(2.0 / eta) / (2.0 * m))
    pi_up = float(np.clip(pi_l + (h / (1.0 - lam) if slack else 0.0), 0.0, 1.0))
    level = 1.0 - alpha * (1.0 - pi_up) + ((h + 1.0 / m) if slack else 0.0)
    if level > 1.0:
        return np.inf
    s = np.sort(W)
    return s[int(np.clip(math.ceil(m * level), 1, m)) - 1]


def theta_eps_corrected(W, alpha, eps):
    """Rank correction with a known poison budget: by interlacing, W_(r_c + k) is at least
    the conformal rank r_c of the m - k clean entries for every placement of k points."""
    s = np.sort(np.asarray(W, float)); m = s.size; k = int(math.floor(eps * m))
    r_c = math.ceil((1 - alpha) * (m - k + 1))
    j = r_c + k
    return s[j - 1] if j <= m else np.inf


def theta_rps(W, alpha, k_parts=None, seed=0):
    """Reliable prediction sets (Scholten and Gunnemann 2025) at the score level: split the
    window into k parts, take the conformal threshold of each part, and accept a score as
    normal when more than tau parts accept it, tau = max{x : BinCDF(x; k, 1 - alpha) <= alpha}.
    The threshold is the (k - tau)-th smallest part threshold."""
    from scipy.stats import binom
    W = np.asarray(W, float); m = W.size
    if k_parts is None:
        k_parts = int(max(1, min(20, math.floor(m * alpha))))
    perm = np.random.default_rng(seed).permutation(m)
    parts = np.array_split(W[perm], k_parts)
    t = np.sort([conf_q(p_, alpha) if p_.size >= math.ceil(1 / alpha) - 1 else np.max(p_) for p_ in parts])
    xs = np.arange(k_parts + 1)
    ok = xs[binom.cdf(xs, k_parts, 1 - alpha) <= alpha]
    tau = int(ok.max()) if ok.size else -1
    if tau < 0:
        return float(t[-1])
    return float(t[k_parts - tau - 1])


def theta_rps_mv(W, alpha, k_parts=None, seed=0):
    """Valid score-level partition rule: split the window into k parts, take the split
    conformal threshold of each part at level alpha/2, and accept a score when a strict
    majority of parts accept it.  By Markov's inequality the majority fails with
    probability at most 2 * (alpha/2) = alpha, whatever the dependence between parts;
    k poison points can affect at most k parts."""
    W = np.asarray(W, float); m = W.size; a2 = alpha / 2.0
    if k_parts is None:
        k_parts = int(max(1, min(20, math.floor(m / math.ceil(1.0 / a2)))))
    perm = np.random.default_rng(seed).permutation(m)
    parts = np.array_split(W[perm], k_parts)
    t = np.sort([conf_q(p_, a2) if p_.size >= math.ceil(1 / a2) - 1 else np.max(p_) for p_ in parts])
    need = k_parts // 2 + 1                  # accept iff #{t_i >= s} >= need
    return float(t[k_parts - need])


def affine_history_ok(R_old, R_prev, level=0.10, npts=19):
    """Diagnostic for DRCC-auto: does an affine transport explain the clean drift between
    the stale reference and the most recent vetted sample?  Theil-Sen fit on quantile
    pairs, two-sample KS distance, compared with the asymptotic KS critical value."""
    R_old = np.asarray(R_old, float); R_prev = np.asarray(R_prev, float)
    ps = np.linspace(0.05, 0.95, npts)
    b, a, _, _ = theilslopes(np.quantile(R_prev, ps), np.quantile(R_old, ps))
    x = np.sort(a + max(b, 1e-6) * R_old); y = np.sort(R_prev)
    grid = np.r_[x, y]
    D = float(np.max(np.abs(np.searchsorted(x, grid, "right") / x.size - np.searchsorted(y, grid, "right") / y.size)))
    c = {0.10: 1.224, 0.05: 1.358, 0.01: 1.628}[level]
    return D <= c * math.sqrt((x.size + y.size) / (x.size * y.size))


ABLATE = os.environ.get("ABLATE", "0") == "1"
ABLATIONS = {"drcc_single": dict(transport="affine", init="single"),
             "drcc_auto": dict(transport="auto"), "drcc_unioncorr": dict(transport="union"),
             "drcc_B10": dict(B=10), "drcc_B40": dict(B=40),
             "drcc_lam025": dict(lam=0.25), "drcc_lam1": dict(lam=1.0),
             "drcc_delta01": dict(delta=0.01), "drcc_delta10": dict(delta=0.10),
             "drcc_iters2": dict(iters=2), "drcc_L15": dict(npts=15),
             "drcc_normal": dict(scan="normal")}


def _drcc_with(R, W, alpha, kw):
    """DRCC with keyword overrides; 'scan' switches the scan test globally for the call."""
    import drcc as _d
    kw = dict(kw); scan = kw.pop("scan", None)
    if scan is None:
        return theta_drcc(R, W, alpha, **kw)
    old = _d.SCAN; _d.SCAN = scan
    try:
        return theta_drcc(R, W, alpha, **kw)
    finally:
        _d.SCAN = old


def all_thresholds(R, Wc, isp, alpha, eps_true, ref_X=None, win_X=None,
                   eps_misspec=(0.05, 0.20), drcc_kw=None, R_prev=None):
    """Thresholds of every rule, plus DRCC diagnostics (estimated excess, flagged bins,
    fitted map)."""
    kw0 = dict(drcc_kw or {})
    if "scan" in kw0:                              # scan-test ablation applies to the whole call
        import drcc as _d
        scan = kw0.pop("scan"); old = _d.SCAN; _d.SCAN = scan
        try:
            return all_thresholds(R, Wc, isp, alpha, eps_true, ref_X, win_X, eps_misspec, kw0, R_prev)
        finally:
            _d.SCAN = old
    th_d, parts = theta_drcc(R, Wc, alpha, return_parts=True, **kw0)
    if "weights" in parts:                         # realized undetected-poison and removed-clean mass (poison labels known in simulation)
        w_ = parts.pop("weights")
        parts["Ec"] = float(np.sum(1.0 - w_[~isp])); parts["Pund"] = float(np.sum(w_[isp]))
    th = {
        "standard":   conf_q(Wc, alpha),
        "ref_only":   conf_q(R, alpha),
        "pooled":     conf_q(np.r_[R, Wc], alpha),
        "naive_trim": rcad.theta_naive_trim(Wc, alpha, eps_true) if eps_true > 0 else conf_q(Wc, alpha),
        "rcad":       rcad.theta_rcad(Wc, alpha, eps_true) if eps_true > 0 else conf_q(Wc, alpha),
        "rcad_u":     rcad.theta_rcad_u(Wc, R, alpha, estimator="tail")[0],
        "qtc":        theta_qtc(R, Wc, alpha),
        "drcc":       th_d,
        # DRCC-safe: union with the no-excess explanation (valid also under mixture drift)
        "drcc_safe":  max(th_d, conf_q(Wc, alpha)),
        "drcc_affine": theta_drcc(R, Wc, alpha, transport="affine"),
        "oracle":     conf_q(Wc[~isp], alpha),
        "aci":        theta_aci(R, Wc, alpha),
        "nexcp":      theta_nexcp(R, Wc, alpha),
        "robust_kl":  theta_robust_kl(R, Wc, alpha),
        "mad_filter": theta_mad_filter(Wc, alpha),
        "cdc":        theta_cdc(R, Wc, alpha),
        "cdc_plugin": theta_cdc(R, Wc, alpha, slack=False),
        "eps_corr":   theta_eps_corrected(Wc, alpha, eps_true),
        "rps":        theta_rps(Wc, alpha),
        "rps_mv":     theta_rps_mv(Wc, alpha),
    }
    # DRCC-auto: DRCC when the clean drift between the stale reference and the most recent
    # vetted sample is explained by a transport (affine check), DRCC-safe otherwise (and
    # whenever no vetted history is available)
    use_tr = R_prev is not None and affine_history_ok(R, R_prev)
    th["drcc_auto_hist"] = th["drcc"] if use_tr else th["drcc_safe"]
    if ABLATE:
        for nm, kw in ABLATIONS.items():
            th[nm] = _drcc_with(R, Wc, alpha, kw)
    for e in eps_misspec:
        th[f"naive_trim_e{int(round(100*e)):02d}"] = rcad.theta_naive_trim(Wc, alpha, e)
    if ref_X is not None and win_X is not None:
        try:
            th["weighted"] = theta_weighted(R, ref_X, win_X, alpha)
        except Exception:
            th["weighted"] = np.nan
    return th, parts

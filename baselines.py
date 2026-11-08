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


ABLATE = os.environ.get("ABLATE", "0") == "1"
ABLATIONS = {"drcc_single": dict(transport="affine", init="single"),
             "drcc_auto": dict(transport="auto"), "drcc_unioncorr": dict(transport="union"),
             "drcc_B10": dict(B=10), "drcc_B40": dict(B=40)}


def all_thresholds(R, Wc, isp, alpha, eps_true, ref_X=None, win_X=None,
                   eps_misspec=(0.05, 0.20), drcc_kw=None):
    """Thresholds of every rule, plus DRCC diagnostics (estimated excess, flagged bins,
    fitted map)."""
    th_d, parts = theta_drcc(R, Wc, alpha, return_parts=True, **(drcc_kw or {}))
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
    }
    if ABLATE:
        for nm, kw in ABLATIONS.items():
            th[nm] = theta_drcc(R, Wc, alpha, **kw)
    for e in eps_misspec:
        th[f"naive_trim_e{int(round(100*e)):02d}"] = rcad.theta_naive_trim(Wc, alpha, e)
    if ref_X is not None and win_X is not None:
        try:
            th["weighted"] = theta_weighted(R, ref_X, win_X, alpha)
        except Exception:
            th["weighted"] = np.nan
    return th, parts

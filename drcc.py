"""Drift-robust conformal calibration (DRCC) via transport + sparse-excess decomposition.

Inputs: clean but STALE reference scores R (n), fresh window W (m) whose normal
part has drifted and an unknown fraction has been replaced by poison.

Model: clean window scores ~ F_0 o g^{-1} with g monotone affine (drift),
       contamination = sparse excess mass located anywhere.

Algorithm (alternating):
  1. transport: fit g on matched quantiles of R and W, using only levels whose
     bins are currently judged clean, with levels corrected for removed mass.
  2. excess scan: on B equal-probability bins of the transported reference g(R)
     (plus two open tail bins), test each bin of W for excess count; for bins
     that pass a Bonferroni one-sided test, take the point estimate of excess.
  3. iterate 1-2.
Threshold: the conformal rank r = ceil((1-alpha)(m - E + 1)) on the
mass-corrected empirical distribution of W, where E is total estimated excess.
"""
import math
import numpy as np
from statistics import NormalDist
from scipy.stats import theilslopes

_Z = NormalDist().inv_cdf
TOL_K = 0.0           # transport-uncertainty slack in the scan, in units of the median fit residual
Z_INFL = 0.0          # additive inflation of the Bonferroni critical value (map-estimation slack)


def _widen(e, frac=0.5):
    """Peaks or atoms of the score law put several equal-probability edges almost on
    top of each other, and then a tiny error in the fitted map moves the whole peak
    across an edge.  Runs of edges closer than frac * (median edge spacing) are
    replaced by one bin centred on the run, of width at least that spacing."""
    e = np.sort(e)
    d = np.diff(e)
    pos = d[d > 0]
    if pos.size == 0:
        return e
    w = frac * np.median(pos) if np.median(d) > 0 else frac * np.mean(pos)
    out, i = [], 0
    while i < e.size:
        j = i
        while j + 1 < e.size and e[j + 1] - e[j] < w:
            j += 1
        if j > i:                                   # a run of near-duplicate edges
            c_lo, c_hi = e[i] - w / 2, e[j] + w / 2
            out = [x for x in out if x < c_lo]
            out += [c_lo, c_hi]
        else:
            if not out or e[i] > out[-1]:
                out.append(e[i])
        i = j + 1
    return np.unique(np.array(out))


def _excess_scan(win, ref_t, B=20, delta=0.05, tol=0.0):
    m, n = win.size, ref_t.size
    inner = _widen(np.quantile(ref_t, np.linspace(0, 1, B + 1)[1:-1]))
    lo, hi = min(np.min(ref_t), inner[0]), max(np.max(ref_t), inner[-1])
    edges = np.concatenate([[-np.inf], [lo], inner, [hi], [np.inf]])
    edges = np.unique(edges)
    cw = np.histogram(win, bins=edges)[0].astype(float)
    cr = np.histogram(ref_t, bins=edges)[0].astype(float)
    if tol > 0:
        # transport uncertainty: a bin is compared with the reference mass of the bin
        # widened by tol on each side, so a slightly misplaced steep feature of the
        # score law (a cliff or an atom) is not read as excess mass
        rs = np.sort(ref_t)
        lo_e = np.searchsorted(rs, edges[:-1] - tol, side="left")
        hi_e = np.searchsorted(rs, edges[1:] + tol, side="left")
        cr = np.maximum(cr, (hi_e - lo_e).astype(float))
    pi = (cr + 0.5) / (n + 0.5 * cr.size)          # smoothed bin probability under the reference
    z = _Z(1 - delta / cw.size) + Z_INFL
    E = 0.0
    for _ in range(20):                            # fixed point: expected clean count = (m - E) pi
        lam = m - E
        exp_c = lam * pi
        se = np.sqrt(exp_c * (1 - pi) + (lam ** 2) * pi * (1 - pi) / n)
        stat = (cw - exp_c) / np.maximum(se, 1e-9)
        exc = np.where(stat > z, np.maximum(cw - exp_c, 0.0), 0.0)
        # the two open bins beyond the reference extremes: the normal approximation is
        # useless there; use the exact exchangeable law of the number of clean window
        # points beyond the extreme of n reference points, P(K >= k) = prod (l-i)/(n+l-i)
        for jb in (0, cw.size - 1):
            k = int(cw[jb]); lam_i = max(int(round(lam)), 1)
            if k == 0:
                exc[jb] = 0.0; continue
            i = np.arange(min(k, lam_i))
            pval = float(np.exp(np.sum(np.log((lam_i - i) / (n + lam_i - i))))) if k <= lam_i else 0.0
            exc[jb] = max(cw[jb] - lam / (n + 1.0), 0.0) if pval < delta / cw.size else 0.0
        E_new = float(np.sum(exc))
        if abs(E_new - E) < 1e-6:
            break
        E = min(E_new, 0.5 * m)
    return edges, cw, exc, E


def _robust_line(qr, qw):
    """Theil-Sen slope restricted to pairs of levels whose reference quantiles are
    separated (pairs inside an atom of the score law carry no scale information);
    intercept = median residual."""
    i, j = np.triu_indices(qr.size, 1)
    dx = qr[j] - qr[i]
    tau = 0.02 * (np.max(qr) - np.min(qr))
    ok = dx > max(tau, 1e-12)
    if ok.sum() < 5:
        b, a, _, _ = theilslopes(qw, qr)
        return a, b
    b = float(np.median((qw[j] - qw[i])[ok] / dx[ok]))
    a = float(np.median(qw - b * qr))
    return a, b


def decompose(ref, win, iters=4, B=20, p_lo=0.05, p_hi_init=0.60, p_hi=0.75, npts=30, E0=0.0, delta=0.05):
    """Stage 1 fits the map on the lower body only (top poison can at most stretch
    levels there by 1/(1-eps)); later stages extend to p_hi using mass-corrected
    quantiles, excluding levels that fall in bins flagged as excess."""
    a, b = 0.0, 1.0
    edges = exc = None
    E = 0.0
    for it in range(iters):
        ps = np.linspace(p_lo, p_hi_init if it == 0 else p_hi, npts)
        if it == 0 and E0 > 0:
            # multi-start: assume a top mass E0 (fraction of m), so clean level p
            # sits at level p (1 - E0) of the window
            qw = np.quantile(win, ps * (1 - E0))
        elif exc is not None and E > 0:
            # mass-corrected window quantile at clean level p
            qw = _corrected_quantile(win, edges, exc, ps)
            # drop levels that fall inside flagged bins of the transported reference
            ok = ~np.isin(np.searchsorted(edges, a + b * np.quantile(ref, ps)) - 1,
                          np.flatnonzero(exc > 0))
            if ok.sum() >= 6:
                ps, qw = ps[ok], qw[ok]
        else:
            qw = np.quantile(win, ps)
        qr = np.quantile(ref, ps)
        a, b = _robust_line(qr, qw)
        b = max(b, 1e-6)
        ref_t = a + b * ref
        tol = TOL_K * float(np.median(np.abs(qw - (a + b * qr))))
        edges, cw, exc, E = _excess_scan(win, ref_t, B=B, delta=delta, tol=tol)
    return a, b, edges, exc, E


def _corrected_quantile(win, edges, exc, ps):
    """Quantiles of W after removing `exc[b]` points uniformly from bin b."""
    s = np.sort(win)
    bin_of = np.searchsorted(edges, s, side="right") - 1
    keep_w = np.ones(s.size)
    for bb in np.flatnonzero(exc > 0):
        sel = np.flatnonzero(bin_of == bb)
        if sel.size:
            keep_w[sel] = max(0.0, 1.0 - exc[bb] / sel.size)
    cum = np.cumsum(keep_w)
    tot = cum[-1]
    out = []
    for p in np.atleast_1d(ps):
        j = int(np.searchsorted(cum, p * tot))
        out.append(s[min(j, s.size - 1)])
    return np.array(out)


def _ks_fit(ref, win, a, b, edges, exc):
    """KS distance between the decontaminated window and the transported reference."""
    s = np.sort(win)
    bin_of = np.searchsorted(edges, s, side="right") - 1
    w = np.ones(s.size)
    for bb in np.flatnonzero(exc > 0):
        sel = np.flatnonzero(bin_of == bb)
        if sel.size:
            w[sel] = max(0.0, 1.0 - exc[bb] / sel.size)
    Fw = np.cumsum(w) / max(np.sum(w), 1e-9)
    rt = np.sort(a + b * ref)
    Fr = np.searchsorted(rt, s, side="right") / rt.size
    return float(np.max(np.abs(Fw - Fr)))


def decompose_ms(ref, win, starts=(0.0, 0.05, 0.10, 0.20, 0.30), lam=0.5, **kw):
    """Penalized minimum-distance multi-start: KS misfit + lam * (removed mass).
    The penalty is the sparsity prior; without it the extra flexibility always
    removes mass, even from an uncontaminated window."""
    best = None
    for e0 in starts:
        a, b, edges, exc, E = decompose(ref, win, E0=e0, **kw)
        d = _ks_fit(ref, win, a, b, edges, exc) + lam * E / win.size
        if best is None or d < best[0]:
            best = (d, a, b, edges, exc, E)
    return best[1:]


WINDOWS = ((0.05, 0.60), (0.05, 0.30), (0.30, 0.60), (0.50, 0.80))


def decompose_mw(ref, win, windows=WINDOWS, lam=0.5, **kw):
    """Multi-window initialization.  A dense band of poison inside the initial
    fitting levels can take more than half of those levels, which no regression on
    quantile pairs survives; so the alternation is started from several level
    windows and the decomposition with the smallest penalized misfit
    KS + lam * E / m is kept (the penalty is the sparsity prior on poison)."""
    best = None
    for lo, hi in windows:
        kw2 = dict(kw); kw2["p_lo"] = lo; kw2["p_hi_init"] = hi
        a, b, edges, exc, E = decompose(ref, win, **kw2)
        d = _ks_fit(ref, win, a, b, edges, exc) + lam * E / win.size
        if best is None or d < best[0]:
            best = (d, a, b, edges, exc, E)
    return best[1:]


KNOTS = np.array([0.05, 0.25, 0.45, 0.65, 0.85])


def _pl_map(qr, qw):
    """Monotone piecewise-linear transport through matched quantiles, extended
    linearly beyond the end knots with the slopes of the end segments."""
    o = np.argsort(qr)
    qr, qw = qr[o], np.maximum.accumulate(qw[o])
    sl = np.maximum(np.diff(qw) / np.maximum(np.diff(qr), 1e-12), 1e-6)

    def T(x):
        y = np.interp(x, qr, qw)
        y = np.where(x < qr[0], qw[0] + sl[0] * (x - qr[0]), y)
        return np.where(x > qr[-1], qw[-1] + sl[-1] * (x - qr[-1]), y)
    return T


def _ks_t(ref_t, win, edges, exc):
    s = np.sort(win)
    bin_of = np.searchsorted(edges, s, side="right") - 1
    w = np.ones(s.size)
    for bb in np.flatnonzero(exc > 0):
        sel = np.flatnonzero(bin_of == bb)
        if sel.size:
            w[sel] = max(0.0, 1.0 - exc[bb] / sel.size)
    Fw = np.cumsum(w) / max(np.sum(w), 1e-9)
    rt = np.sort(ref_t)
    Fr = np.searchsorted(rt, s, side="right") / rt.size
    return float(np.max(np.abs(Fw - Fr)))


def decompose_auto(ref, win, lam=0.5, gam=0.10, eps_max=0.35, iters_pl=3, B=20, delta=0.05, **kw):
    """Transport-family selection.  The affine decomposition (multi-window start)
    competes with a piecewise-linear monotone transport through five body knots,
    started either from raw window quantiles or from the affine stage's
    mass-corrected quantiles.  The richer family is consulted only when the affine
    explanation needs more excess mass than the contamination budget eps_max allows;
    the winner then minimizes KS + lam * E / m, plus gam for the piecewise-linear
    family (a more flexible transport can absorb poison, so it pays a price)."""
    a, b, edges, exc, E = decompose_mw(ref, win, lam=lam, B=B, delta=delta, **kw)
    m = win.size
    if E <= eps_max * m:                 # the parsimonious explanation fits the budget
        return edges, exc, E, "affine", a, b
    ref_t = a + b * ref
    best = (_ks_t(ref_t, win, edges, exc) + lam * E / m, edges, exc, E, "affine", a, b)
    qr = np.quantile(ref, KNOTS)
    for start in ("raw", "corrected"):
        e2, x2, E2 = edges, exc, E
        for it in range(iters_pl):
            if (start == "raw" and it == 0) or E2 <= 0:
                qw = np.quantile(win, KNOTS)
            else:
                qw = _corrected_quantile(win, e2, x2, KNOTS)
            T = _pl_map(qr, qw)
            rt = T(ref)
            e2, _, x2, E2 = _excess_scan(win, rt, B=B, delta=delta)
        d = _ks_t(rt, win, e2, x2) + lam * E2 / m + gam
        if d < best[0]:
            best = (d, e2, x2, E2, "pl_" + start, np.nan, np.nan)
    return best[1:]


def _theta_from(win, edges, exc, alpha):
    """Conformal rank ceil((1-alpha)(m-E+1)) on the mass-corrected window."""
    m = win.size
    s = np.sort(win)
    bin_of = np.searchsorted(edges, s, side="right") - 1
    keep_w = np.ones(m)
    for bb in np.flatnonzero(exc > 0):
        sel = np.flatnonzero(bin_of == bb)
        if sel.size:
            keep_w[sel] = max(0.0, 1.0 - exc[bb] / sel.size)
    eff = float(np.sum(keep_w))
    r = math.ceil((1 - alpha) * (eff + 1))
    cum = np.cumsum(keep_w)
    j = int(np.searchsorted(cum, r - 1e-9))
    return s[min(j, m - 1)] if r <= eff else s[-1]


def decompose_pl(ref, win, start_edges, start_exc, start_E, start="corrected", iters_pl=3, B=20, delta=0.05,
                 knots=None):
    """Piecewise-linear monotone transport through the body knots, alternated with
    the excess scan, started from raw or from mass-corrected window quantiles."""
    KNOTS = np.asarray(knots) if knots is not None else globals()["KNOTS"]
    qr = np.quantile(ref, KNOTS)
    e2, x2, E2 = start_edges, start_exc, start_E
    for it in range(iters_pl):
        if (start == "raw" and it == 0) or E2 <= 0:
            qw = np.quantile(win, KNOTS)
        else:
            qw = _corrected_quantile(win, e2, x2, KNOTS)
        T = _pl_map(qr, qw)
        e2, _, x2, E2 = _excess_scan(win, T(ref), B=B, delta=delta)
    return e2, x2, E2


KNOTS_S = np.array([0.05, 0.25, 0.45, 0.65, 0.85, 0.95])


def _pl_map_capped(qr, qw, kappa=2.5, free=4):
    """Monotone piecewise-linear map whose tail segments (beyond knot index free-1)
    may change slope by at most a factor kappa from one segment to the next: a
    regular drift bends the tail gradually, while absorbing a detached cluster of
    poison would need an abrupt jump in slope."""
    o = np.argsort(qr); qr, qw = qr[o], np.maximum.accumulate(qw[o])
    sl = np.maximum(np.diff(qw) / np.maximum(np.diff(qr), 1e-12), 1e-6)
    for i in range(free - 1, sl.size):
        sl[i] = min(max(sl[i], sl[i - 1] / kappa), sl[i - 1] * kappa)
    qw2 = np.concatenate([[qw[0]], qw[0] + np.cumsum(sl * np.diff(qr))])

    def T(x):
        y = np.interp(x, qr, qw2)
        y = np.where(x < qr[0], qw2[0] + sl[0] * (x - qr[0]), y)
        return np.where(x > qr[-1], qw2[-1] + sl[-1] * (x - qr[-1]), y)
    return T


def decompose_smooth(ref, win, starts=(0.0, 0.05, 0.10, 0.20, 0.30), lam=0.5, iters=3, B=20, delta=0.05,
                     kappa=2.5, knots=KNOTS_S):
    """Slope-regular piecewise-linear transport with top-mass multi-start; the start
    with the smallest KS + lam * E / m is kept."""
    qr = np.quantile(ref, knots); m = win.size
    best = None
    for e0 in starts:
        qw = np.quantile(win, knots * (1 - e0))
        for it in range(iters):
            T = _pl_map_capped(qr, qw, kappa=kappa)
            rt = T(ref)
            e2, _, x2, E2 = _excess_scan(win, rt, B=B, delta=delta)
            qw = _corrected_quantile(win, e2, x2, knots) if E2 > 0 else np.quantile(win, knots)
        d = _ks_t(rt, win, e2, x2) + lam * E2 / m
        if best is None or d < best[0]:
            best = (d, e2, x2, E2)
    return best[1:]


def _deficit_z(win, edges, exc, ref_t, maxrun=6):
    """Most negative standardized count of W over runs of up to maxrun consecutive
    bins of the transported reference (expected clean count (m - E) pi).  Poison
    only adds mass, so under a correct transport no run of bins is significantly
    short; a map bent to absorb a detached cluster of poison stretches reference
    mass over a gap of W.  Returns (z_min, number of runs tested)."""
    m, n = win.size, ref_t.size
    cw = np.histogram(win, bins=edges)[0].astype(float) - exc
    cr = np.histogram(ref_t, bins=edges)[0].astype(float)
    lam = m - float(np.sum(exc))
    zmin, cnt = np.inf, 0
    K = cw.size
    for L in range(1, maxrun + 1):
        for i in range(0, K - L + 1):
            c = cw[i:i + L].sum(); p = (cr[i:i + L].sum() + 0.5) / (n + 1.0)
            se = np.sqrt(lam * p * (1 - p) + lam ** 2 * p * (1 - p) / n)
            zmin = min(zmin, (c - lam * p) / max(se, 1e-9)); cnt += 1
    return float(zmin), cnt


def _slope_ratio(qr, qw):
    """Largest segment slope of the knot map relative to its median body slope."""
    sl = np.diff(np.maximum.accumulate(qw)) / np.maximum(np.diff(qr), 1e-12)
    return float(np.max(sl) / max(np.median(sl[: max(2, sl.size - 2)]), 1e-12))


def theta_drcc_guard(ref, win, alpha, return_parts=False, lam=0.5, B=20, delta=0.05, knots=KNOTS_S,
                     starts=("raw", "corrected"), iters_pl=3, kappa=3.0, use_deficit=False, **kw):
    """Union over admissible transports: affine, plus piecewise-linear maps (knots up
    to 0.95) that leave no bin significantly short of the transported reference.
    The threshold is the largest admissible one."""
    a, b, edges, exc, E = decompose_mw(ref, win, lam=lam, B=B, delta=delta, **kw)
    best = (_theta_from(win, edges, exc, alpha), E, int(np.sum(exc > 0)), "affine")
    qr = np.quantile(ref, knots)
    for st in starts:
        e2, x2, E2 = edges, exc, E
        for it in range(iters_pl):
            qw = np.quantile(win, knots) if ((st == "raw" and it == 0) or E2 <= 0) else _corrected_quantile(win, e2, x2, knots)
            rt = _pl_map(qr, qw)(ref)
            e2, _, x2, E2 = _excess_scan(win, rt, B=B, delta=delta)
        if _slope_ratio(qr, qw) > kappa:
            continue
        if use_deficit:
            zmin, nb = _deficit_z(win, e2, x2, rt)
            if zmin < -_Z(1 - delta / nb):
                continue
        th2 = _theta_from(win, e2, x2, alpha)
        if th2 > best[0]:
            best = (th2, E2, int(np.sum(x2 > 0)), "pl_" + st)
    if return_parts:
        return best[0], dict(a=a, b=b, E=best[1], flagged=best[2], family=best[3])
    return best[0]


def theta_drcc_union(ref, win, alpha, return_parts=False, lam=0.5, starts=("corrected",), B=20, delta=0.05,
                     knots_raw=None, knots_corr=None, **kw):
    """Union-of-models DRCC: the threshold is the largest of the thresholds computed
    under each candidate transport family (affine, piecewise-linear).  Removing
    excess mass can only lower a threshold when the removed mass lies above it, so
    the maximum is valid whenever any one family contains the true drift."""
    a, b, edges, exc, E = decompose_mw(ref, win, lam=lam, B=B, delta=delta, **kw)
    best = (_theta_from(win, edges, exc, alpha), E, int(np.sum(exc > 0)), "affine")
    if "smooth" in starts:
        e2, x2, E2 = decompose_smooth(ref, win, lam=lam, B=B, delta=delta, kappa=kw.get("kappa", 2.5))
        th2 = _theta_from(win, e2, x2, alpha)
        if th2 > best[0]:
            best = (th2, E2, int(np.sum(x2 > 0)), "pl_smooth")
    for st in [x for x in starts if x != "smooth"]:
        e2, x2, E2 = decompose_pl(ref, win, edges, exc, E, start=st, B=B, delta=delta,
                                  knots=knots_raw if st == "raw" else knots_corr)
        th2 = _theta_from(win, e2, x2, alpha)
        if th2 > best[0]:
            best = (th2, E2, int(np.sum(x2 > 0)), "pl_" + st)
    if return_parts:
        return best[0], dict(a=a, b=b, E=best[1], flagged=best[2], family=best[3])
    return best[0]


def theta_drcc(ref, win, alpha, return_parts=False, init="multiwindow", lam=0.5,
               transport="union3", gam=0.10, eps_max=0.35, multistart=None, **kw):
    """transport: "union3" (default: largest threshold over affine and piecewise-linear transports), "auto" (gated selection) or "affine".
    init (affine stage): 'multiwindow' (default), 'single', or 'e0'."""
    if multistart is not None:                        # backward-compatible switch
        init = "e0" if multistart else "single"
    if "p_hi" in kw:                                  # keep the initial window below p_hi
        kw.setdefault("p_hi_init", min(0.60, kw["p_hi"] - 0.05))
    if transport in ("union", "union3"):
        return theta_drcc_union(ref, win, alpha, return_parts=return_parts, lam=lam,
                                starts=("corrected",) if transport == "union" else ("raw", "corrected"), **kw)
    if transport == "guard":
        return theta_drcc_guard(ref, win, alpha, return_parts=return_parts, lam=lam, **kw)
    if transport == "guard_k2":
        return theta_drcc_guard(ref, win, alpha, return_parts=return_parts, lam=lam, kappa=2.0, **kw)
    if transport == "guard85_k3":
        return theta_drcc_guard(ref, win, alpha, return_parts=return_parts, lam=lam, knots=KNOTS, **kw)
    if transport == "guard85":
        return theta_drcc_guard(ref, win, alpha, return_parts=return_parts, lam=lam, knots=KNOTS, kappa=1e9, **kw)
    if transport == "union_s":
        return theta_drcc_union(ref, win, alpha, return_parts=return_parts, lam=lam, starts=("smooth",), **kw)
    if transport == "union_sc":
        return theta_drcc_union(ref, win, alpha, return_parts=return_parts, lam=lam, starts=("smooth", "corrected"), **kw)
    if transport == "union65":
        return theta_drcc_union(ref, win, alpha, return_parts=return_parts, lam=lam, starts=("raw", "corrected"),
                                knots_raw=[0.05, 0.20, 0.35, 0.50, 0.65], **kw)
    if transport == "union55":
        return theta_drcc_union(ref, win, alpha, return_parts=return_parts, lam=lam, starts=("raw", "corrected"),
                                knots_raw=[0.05, 0.175, 0.30, 0.425, 0.55], **kw)
    family = "affine"
    if transport == "auto" and init == "multiwindow":
        kw.pop("p_hi_init", None)
        edges, exc, E, family, a, b = decompose_auto(ref, win, lam=lam, gam=gam, eps_max=eps_max, **kw)
    elif init == "multiwindow":
        kw.pop("p_hi_init", None)
        a, b, edges, exc, E = decompose_mw(ref, win, lam=lam, **kw)
    elif init == "e0":
        a, b, edges, exc, E = decompose_ms(ref, win, lam=lam, **kw)
    else:
        a, b, edges, exc, E = decompose(ref, win, **kw)
    m = win.size
    s = np.sort(win)
    bin_of = np.searchsorted(edges, s, side="right") - 1
    keep_w = np.ones(m)
    for bb in np.flatnonzero(exc > 0):
        sel = np.flatnonzero(bin_of == bb)
        if sel.size:
            keep_w[sel] = max(0.0, 1.0 - exc[bb] / sel.size)
    eff = float(np.sum(keep_w))                          # effective clean sample size m - E
    r = math.ceil((1 - alpha) * (eff + 1))
    cum = np.cumsum(keep_w)
    j = int(np.searchsorted(cum, r - 1e-9))
    th = s[min(j, m - 1)] if r <= eff else s[-1]
    if return_parts:
        return th, dict(a=a, b=b, E=E, flagged=int(np.sum(exc > 0)), family=family)
    return th


def dequantize(arrays, rng):
    """Randomized tie-breaking for scores with atoms.  Each value v of the pooled
    sample is spread uniformly over [v, v'), v' the next distinct pooled value, so an
    atom becomes a flat density over its gap while the order of distinct values is
    kept (continuous scores move by a negligible amount).  Applied identically to
    reference, window and test scores."""
    arrs = [None if x is None else np.asarray(x, float) for x in arrays]
    pool = np.unique(np.concatenate([x.ravel() for x in arrs if x is not None]))
    if pool.size < 3:
        return arrs
    gaps = np.diff(pool)
    gaps = np.append(gaps, np.median(gaps))
    out = []
    for x in arrs:
        if x is None:
            out.append(None); continue
        # test scores may fall between pooled values: use the gap of the value below
        i = np.clip(np.searchsorted(pool, x, side="right") - 1, 0, pool.size - 1)
        out.append(x + rng.uniform(0, 1, x.shape) * gaps[i])
    return out

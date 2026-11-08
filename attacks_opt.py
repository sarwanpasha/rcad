"""White-box targeted poisoning.  The attacker knows the calibration rule, the stale
reference R and the clean window W, and places k = floor(eps m) poison scores to move
that rule's threshold as far as possible: down ("lo", raises type-I error) or up ("hi",
removes detection power).  Search: a structured sweep of single and double bands, top
and bottom clusters, wide sub-threshold spreads and transport-mimicking placements (the
window body is made to look like a different affine image of the reference, which
targets the transport fit and the misfit selection), then randomized local search from
the best few starts.  `iters` local-search steps per start (OPT_ITERS, default 240)."""
import math, os
import numpy as np
from baselines import (conf_q, theta_aci, theta_nexcp, theta_robust_kl, theta_mad_filter,
                       theta_cdc, theta_eps_corrected, theta_rps, theta_rps_mv)
from drcc import theta_drcc

OPT_ITERS = int(os.environ.get("OPT_ITERS", "240"))

RULES = {
    "standard": lambda R, W, a, e=0.0: conf_q(W, a),
    "drcc": lambda R, W, a, e=0.0: theta_drcc(R, W, a),
    "drcc_safe": lambda R, W, a, e=0.0: max(theta_drcc(R, W, a), conf_q(W, a)),
    "drcc_affine": lambda R, W, a, e=0.0: theta_drcc(R, W, a, transport="affine"),
    "mad_filter": lambda R, W, a, e=0.0: theta_mad_filter(W, a),
    "aci": lambda R, W, a, e=0.0: theta_aci(R, W, a),
    "nexcp": lambda R, W, a, e=0.0: theta_nexcp(R, W, a),
    "robust_kl": lambda R, W, a, e=0.0: theta_robust_kl(R, W, a),
    "cdc_plugin": lambda R, W, a, e=0.0: theta_cdc(R, W, a, slack=False),
    "eps_corr": lambda R, W, a, e=0.0: theta_eps_corrected(W, a, e),
    "rps": lambda R, W, a, e=0.0: theta_rps(W, a),
    "rps_mv": lambda R, W, a, e=0.0: theta_rps_mv(W, a),
}


def opt_attack(rule, R, W, eps, direction, rng, alpha=0.10, iters=None, starts=3):
    iters = OPT_ITERS if iters is None else iters
    f = RULES[rule]; m = W.size; k = int(math.floor(eps * m))
    idx = rng.choice(m, k, replace=False); isp = np.zeros(m, bool); isp[idx] = True
    sd = np.std(W) + 1e-9; lo_b, hi_b = W.min() - 5 * sd, W.max() + 5 * sd
    sign = -1.0 if direction == "lo" else 1.0

    def place(vals):
        Wc = W.copy(); Wc[idx] = np.clip(vals, lo_b, hi_b); return Wc

    def obj(vals):
        th = f(R, place(vals), alpha, eps)
        return sign * (th if np.isfinite(th) else (hi_b * 10 if th > 0 else lo_b * 10))

    cands = []
    for q in np.linspace(0, 1, 21):
        for sp in (0.02, 0.1, 0.3):
            cands.append(np.quantile(W, q) + sp * sd * rng.uniform(-1, 1, k))
    cands.append(W.max() + sd * rng.uniform(0.5, 3, k)); cands.append(W.min() - sd * rng.uniform(0.5, 3, k))
    for _ in range(20):
        q1, q2 = rng.uniform(0, 1, 2); h = k // 2
        cands.append(np.r_[np.quantile(W, q1) + 0.05 * sd * rng.uniform(-1, 1, h),
                           np.quantile(W, q2) + 0.05 * sd * rng.uniform(-1, 1, k - h)])
    # wide sub-threshold spreads: poison spread evenly over a range of window levels, so
    # each bin of the scan receives only a small excess
    for q1 in (0.0, 0.2, 0.4, 0.6):
        for w_ in (0.2, 0.4, 0.6):
            cands.append(np.quantile(W, np.clip(np.linspace(q1, q1 + w_, k), 0, 1)))
    # transport mimicry: place poison on a different affine image a' + b' R of the
    # reference, restricted to a level range, so that the body of the window supports a
    # wrong transport and the scan removes clean mass
    medR, medW = np.median(R), np.median(W)
    for bb in (0.6, 0.8, 1.25, 1.6):
        for da in (-1.0, -0.5, 0.0, 0.5, 1.0):
            for (p1, p2) in ((0.0, 0.5), (0.3, 0.8), (0.5, 1.0)):
                lv = np.quantile(R, np.linspace(p1, p2, k))
                cands.append(medW + da * sd + bb * (lv - medR) * sd / (np.std(R) + 1e-9))
    scores = np.array([obj(v) for v in cands])
    top = np.argsort(scores)[::-1][:max(1, starts)]
    best, bs = cands[int(top[0])].copy(), float(scores[top[0]])
    for t0 in top:
        cur, cs = cands[int(t0)].copy(), float(scores[t0])
        for _ in range(iters // max(1, starts)):
            v = cur.copy(); j = rng.random(k) < 0.25
            if not j.any():
                j[rng.integers(k)] = True
            u = rng.random()
            if u < 0.4:
                v[j] = np.quantile(W, rng.uniform(0, 1, j.sum())) + 0.05 * sd * rng.standard_normal(j.sum())
            elif u < 0.8:
                v[j] = v[j] + 0.3 * sd * rng.standard_normal(j.sum())
            else:                                            # shift or rescale the whole block
                c0 = np.median(v); v = c0 + (v - c0) * np.exp(0.2 * rng.standard_normal()) + 0.2 * sd * rng.standard_normal()
            s_ = obj(v)
            if s_ > cs:
                cur, cs = v, s_
        if cs > bs:
            best, bs = cur, cs
    return place(best), isp

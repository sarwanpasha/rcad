"""White-box targeted poisoning.  The attacker knows the calibration rule, the stale
reference R and the clean window W, and places k = floor(eps m) poison scores to move
that rule's threshold as far as possible: down ("lo", raises type-I error) or up ("hi",
removes detection power).  Search: a structured sweep of single and double bands, top
and bottom clusters, then randomized local search (about 150 rule evaluations)."""
import math
import numpy as np
from baselines import conf_q, theta_aci, theta_nexcp, theta_robust_kl, theta_mad_filter
from drcc import theta_drcc

RULES = {
    "standard": lambda R, W, a: conf_q(W, a),
    "drcc": lambda R, W, a: theta_drcc(R, W, a),
    "drcc_safe": lambda R, W, a: max(theta_drcc(R, W, a), conf_q(W, a)),
    "drcc_affine": lambda R, W, a: theta_drcc(R, W, a, transport="affine"),
    "mad_filter": lambda R, W, a: theta_mad_filter(W, a),
    "aci": lambda R, W, a: theta_aci(R, W, a),
    "nexcp": lambda R, W, a: theta_nexcp(R, W, a),
    "robust_kl": lambda R, W, a: theta_robust_kl(R, W, a),
}


def opt_attack(rule, R, W, eps, direction, rng, alpha=0.10, iters=60):
    f = RULES[rule]; m = W.size; k = int(math.floor(eps * m))
    idx = rng.choice(m, k, replace=False); isp = np.zeros(m, bool); isp[idx] = True
    sd = np.std(W) + 1e-9; lo_b, hi_b = W.min() - 5 * sd, W.max() + 5 * sd
    sign = -1.0 if direction == "lo" else 1.0

    def place(vals):
        Wc = W.copy(); Wc[idx] = np.clip(vals, lo_b, hi_b); return Wc

    def obj(vals):
        th = f(R, place(vals), alpha)
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
    scores = [obj(v) for v in cands]
    best = cands[int(np.argmax(scores))].copy(); bs = max(scores)
    for _ in range(iters):
        v = best.copy(); j = rng.random(k) < 0.25
        if not j.any():
            j[rng.integers(k)] = True
        if rng.random() < 0.5:
            v[j] = np.quantile(W, rng.uniform(0, 1, j.sum())) + 0.05 * sd * rng.standard_normal(j.sum())
        else:
            v[j] = v[j] + 0.3 * sd * rng.standard_normal(j.sum())
        s = obj(v)
        if s > bs:
            best, bs = v, s
    return place(best), isp

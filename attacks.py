"""Poison generators.  Score-space versions for controlled studies, and a
feasible-input version that picks REAL candidate inputs whose scores land in the
attacker's target band (used on real data)."""
import numpy as np


def target_scores(kind, rng, k, W_clean, q=None, spread=0.05):
    """Desired poison scores, relative to the clean window distribution."""
    sd = np.std(W_clean) + 1e-9
    if kind == "top":
        return np.max(W_clean) + sd * rng.uniform(0.5, 3.0, k)
    if kind == "band":
        return np.quantile(W_clean, q) + spread * sd * rng.uniform(0, 1, k)
    if kind == "bottom":
        return np.min(W_clean) - sd * rng.uniform(0.5, 1.5, k)
    if kind == "mimic":
        # adaptive: shaped like a stretched copy of the clean window's upper half,
        # i.e. it looks like extra scale drift rather than a local bump
        base = rng.choice(W_clean[W_clean > np.median(W_clean)], size=k, replace=True)
        return np.median(W_clean) + 1.6 * (base - np.median(W_clean)) + 0.02 * sd * rng.standard_normal(k)
    raise ValueError(kind)


def feasible_pick(desired, cand_scores, rng):
    """For each desired score, choose the unused candidate input whose score is nearest.
    Returns indices into the candidate pool."""
    order = np.argsort(cand_scores); cs = cand_scores[order]
    used = np.zeros(cs.size, bool); out = []
    for d in np.sort(desired)[::-1]:
        j = int(np.clip(np.searchsorted(cs, d), 0, cs.size - 1))
        lo, hi = j, j
        while used[j]:
            lo, hi = max(lo - 1, 0), min(hi + 1, cs.size - 1)
            j = hi if not used[hi] else lo
            if lo == 0 and hi == cs.size - 1 and used[j]:
                break
        used[j] = True; out.append(order[j])
    return np.array(out, dtype=int)


def poison_window(kind, rng, eps, W_clean, q=None, cand_scores=None):
    """Fixed-budget replacement.  Returns (contaminated scores, poison mask,
    picked candidate indices or None, replaced window positions)."""
    m = W_clean.size; k = int(np.floor(eps * m))
    Wc = W_clean.copy(); isp = np.zeros(m, bool); picked = None
    if k == 0:
        return Wc, isp, picked, np.zeros(0, int)
    want = target_scores(kind, rng, k, W_clean, q=q)
    if cand_scores is not None:
        picked = feasible_pick(want, cand_scores, rng)
        vals = cand_scores[picked]
    else:
        vals = want
    idx = rng.choice(m, k, replace=False)
    Wc[idx] = vals; isp[idx] = True
    return Wc, isp, picked, idx

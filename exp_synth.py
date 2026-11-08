"""expF (score-level drift, exact metrics), expI-synthetic (dense adaptive attack grid)
and expJ-synthetic (ablations).

Scores: the stale reference is drawn from a base law F0, the current clean window
from g#F0 for a drift map g, and anomalies from g#F0 shifted by 2.5 sd.  Type-I and
power are computed against 400k-draw Monte Carlo laws, so they are exact up to 1e-3.

usage: python exp_synth.py {F,I,J} TASK NTASKS
"""
import sys, os, time, itertools
import zlib
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
from common import make_poison, rows_for, attack_list
from baselines import all_thresholds

BASES = ("gauss", "gamma")
DRIFTS = [("none", 0)] + [(f, d) for f in ("affine", "tailstretch", "tailheavy", "mixture") for d in (1, 2)]
EPS_F = (0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30)


def base_draw(base, rng, n):
    if base == "gauss":
        return rng.standard_normal(n)
    return (rng.gamma(2.0, 1.0, n) - 2.0) / np.sqrt(2.0)        # skewed, mean 0, sd 1


def drift_map(fam, d, rng, n_clean, base):
    """Draw n_clean drifted clean scores.  Returns array."""
    z = base_draw(base, rng, n_clean)
    if fam == "none" or d == 0:
        return z
    if fam == "affine":
        return 0.5 * d + (1 + 0.15 * d) * z
    if fam == "tailstretch":                                     # monotone, non-affine
        return 0.25 * d + z + 0.10 * d * z * np.abs(z)
    if fam == "tailheavy":                                       # upper tail bends only (shape drift)
        return z + 0.25 * d * np.maximum(z, 0) ** 2
    if fam == "mixture":                                         # a fraction of mass shifted
        w = 0.15 * d
        sh = rng.random(n_clean) < w
        return z + sh * 1.0
    raise ValueError(fam)


def laws(base, fam, d, seed=12345, n=400_000):
    rng = np.random.default_rng(seed)
    clean = np.sort(drift_map(fam, d, rng, n, base))
    anom = np.sort(drift_map(fam, d, rng, n, base) + 2.5)
    return clean, anom


def run_configs(cfgs, trials, out, drcc_kw_list=None, m=200, n_ref=200):
    rows = []; t0 = time.time(); cache = {}
    for ci, (base, fam, d, alpha, kind, q, sp, st, eps, m_, nref_) in enumerate(cfgs):
        key = (base, fam, d)
        if key not in cache:
            cache[key] = laws(base, fam, d)
        tn, ta = cache[key]
        for t in range(trials):
            rng = np.random.default_rng([zlib.crc32(repr(key).encode()), zlib.crc32(repr(cfgs[ci]).encode()), t])
            R = base_draw(base, rng, nref_)
            W = drift_map(fam, d, rng, m_, base)
            Wc, isp, _, _ = make_poison(kind, rng, eps, W, q=q, spread=sp or 0.05, stretch=st or 1.6)
            for kwname, kw in (drcc_kw_list or [("default", None)]):
                th, parts = all_thresholds(R, Wc, isp, alpha, eps, drcc_kw=kw)
                if kwname != "default":
                    th = {"drcc": th["drcc"], "drcc_safe": th["drcc_safe"], "oracle": th["oracle"], "standard": th["standard"]}
                base_d = dict(base=base, drift=fam, sev=d, alpha=alpha, attack=kind, q=q, spread=sp,
                              stretch=st, eps=eps, m=m_, n_ref=nref_, trial=t, variant=kwname)
                rows += rows_for(base_d, th, parts, tn, ta)
        if ci % 20 == 0:
            print(f"cfg {ci}/{len(cfgs)}  {time.time()-t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out, len(rows), f"{time.time()-t0:.0f}s")


def cfgs_F():
    out = []
    for base, (fam, d), alpha in itertools.product(BASES, DRIFTS, (0.10, 0.05)):
        for kind, q, sp, st in attack_list(dense=False):
            for eps in (EPS_F if kind != "none" else (0.0,)):
                out.append((base, fam, d, alpha, kind, q, sp, st, eps, 200, 200))
    return out


def cfgs_I():
    out = []
    for base, (fam, d) in itertools.product(BASES, [("none", 0), ("affine", 2), ("tailstretch", 2), ("tailheavy", 2)]):
        for kind, q, sp, st in attack_list(dense=True):
            if kind == "none":
                continue
            for eps in (0.05, 0.10, 0.20):
                out.append((base, fam, d, 0.10, kind, q, sp, st, eps, 200, 200))
    return out


def cfgs_J_size():
    out = []
    for base, (fam, d) in itertools.product(BASES, [("none", 0), ("affine", 2), ("tailstretch", 2)]):
        for kind, q in (("none", None), ("top", None), ("band", 0.8), ("band", 0.95)):
            for eps in ((0.0,) if kind == "none" else (0.10, 0.20)):
                for m_ in (100, 200, 500, 1000):
                    out.append((base, fam, d, 0.10, kind, q, 0.05, None, eps, m_, 200))
                for nr in (50, 100, 500, 1000):
                    out.append((base, fam, d, 0.10, kind, q, 0.05, None, eps, 200, nr))
                out.append((base, fam, d, 0.01, kind, q, 0.05, None, eps, 1000, 1000))
    return out


DRCC_VARIANTS = [("default", None)] + \
    [(f"B{B}", dict(B=B)) for B in (10, 40)] + \
    [(f"delta{dl}", dict(delta=dl)) for dl in (0.01, 0.10)] + \
    [(f"phi{ph}", dict(p_hi=ph)) for ph in (0.65, 0.85)] + \
    [(f"iters{it}", dict(iters=it)) for it in (1, 8)] + \
    [(f"lam{lm}", dict(lam=lm)) for lm in (0.25, 1.0)] + \
    [("single", dict(init="single", transport="affine")), ("e0", dict(init="e0", transport="affine")),
     ("affine", dict(transport="affine")), ("auto", dict(transport="auto")),
     ("union_corr", dict(transport="union")), ("guard_k3", dict(transport="guard"))] + \
    [(f"gam{g}", dict(gam=g, transport="auto")) for g in (0.05, 0.20)]


def cfgs_J_knobs():
    out = []
    for base, (fam, d) in itertools.product(BASES, [("none", 0), ("affine", 2), ("tailstretch", 2), ("tailheavy", 2), ("mixture", 2)]):
        for kind, q in (("none", None), ("top", None), ("band", 0.5), ("band", 0.8), ("band", 0.95), ("mimic", None)):
            for eps in ((0.0,) if kind == "none" else (0.10, 0.20)):
                out.append((base, fam, d, 0.10, kind, q, 0.05, 1.6 if kind == "mimic" else None, eps, 200, 200))
    return out


def cfgs_T_eps():
    """Theory check 1: type-I / power as a function of the poison fraction around eps = alpha."""
    out = []
    for base, (fam, d) in itertools.product(BASES, [("none", 0), ("affine", 2), ("tailheavy", 2)]):
        for kind, q in (("top", None), ("bottom", None), ("band", 0.5), ("band", 0.9)):
            for eps in np.round(np.arange(0.0, 0.305, 0.02), 2):
                out.append((base, fam, d, 0.10, kind if eps > 0 else "none", q if eps > 0 else None,
                            0.05, None, float(eps), 200, 200))
    return sorted(set(out), key=str)


def run_topt(task, ntask, trials, out):
    """Theory check 2: white-box targeted attacks against each rule, eps grid."""
    import attacks_opt as AO
    from baselines import conf_q
    units = list(itertools.product(BASES, [("none", 0), ("affine", 2), ("tailheavy", 2), ("mixture", 2)],
                                   (0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30), ("lo", "hi")))[task::ntask]
    rows = []; t0 = time.time(); cache = {}
    for base, (fam, d), eps, direction in units:
        key = (base, fam, d)
        if key not in cache:
            cache[key] = laws(base, fam, d)
        tn, ta = cache[key]
        for t in range(trials):
            rng = np.random.default_rng([zlib.crc32(repr((key, eps, direction)).encode()), t])
            R = base_draw(base, rng, 200); W = drift_map(fam, d, rng, 200, base)
            for rule in AO.RULES:
                Wc, isp = AO.opt_attack(rule, R, W, eps, direction, rng)
                th = AO.RULES[rule](R, Wc, 0.10)
                o = conf_q(Wc[~isp], 0.10)
                ti = lambda x: 1 - np.searchsorted(tn, x, "right") / tn.size
                pw = lambda x: 1 - np.searchsorted(ta, x, "right") / ta.size
                rows.append(dict(base=base, drift=fam, sev=d, eps=eps, direction=direction, trial=t, rule=rule,
                                 theta=th, theta_oracle=o, typeI=ti(th), power=pw(th), typeI_or=ti(o), power_or=pw(o)))
        print(base, fam, eps, direction, f"{time.time()-t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)


if __name__ == "__main__":
    exp, task, ntask = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    os.makedirs("results_drift", exist_ok=True)
    trials = int(os.environ.get("TRIALS", "200"))
    if exp == "F":
        cf = cfgs_F()[task::ntask]
        run_configs(cf, trials, f"results_drift/F_{task:02d}.csv")
    elif exp == "I":
        cf = cfgs_I()[task::ntask]
        run_configs(cf, trials, f"results_drift/Isyn_{task:02d}.csv")
    elif exp == "J":
        cf = cfgs_J_size()[task::ntask]
        run_configs(cf, trials, f"results_drift/Jsize_{task:02d}.csv")
    elif exp == "K":                                       # DRCC knob ablation
        cf = cfgs_J_knobs()[task::ntask]
        run_configs(cf, trials, f"results_drift/Jknob_{task:02d}.csv", drcc_kw_list=DRCC_VARIANTS)
    elif exp == "T":
        cf = cfgs_T_eps()[task::ntask]
        run_configs(cf, trials, f"results_drift/Teps_{task:02d}.csv")
    elif exp == "O":
        run_topt(task, ntask, trials, f"results_drift/Topt_{task:02d}.csv")
    print("done")

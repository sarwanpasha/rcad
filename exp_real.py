"""expG: injected feature drift on tabular benchmarks.
expH: real temporal drift streams (gas, creditcard, news, jailbreak).

For each (dataset, detector, seed) the detector is fit once on old normals, the
reference is a held-out set of old normals, and each later period supplies a
clean window, test normals, test anomalies and an attacker candidate pool.
Poison is either score-space (exact placement) or feasible (real inputs chosen
from the candidate pool, whose scores are as close as possible to the target).

usage: python exp_real.py G TASK NTASKS     |   python exp_real.py H TASK NTASKS
"""
import sys, os, time, itertools, zlib
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler

import rcad
from common import make_detector, make_poison, rows_for, attack_list, candidate_pool
from baselines import all_thresholds, conf_q
from drcc import dequantize

SEEDS = rcad.SEEDS
EPS = (0.0, 0.05, 0.10, 0.20, 0.30)
INPUT_JITTER = 1e-2
M, NREF, NTEST = 200, 200, 2000


def seed_of(*parts):
    return zlib.crc32(repr(parts).encode())


ALPHAS = tuple(float(a) for a in os.environ.get("ALPHAS", "0.10").split(","))
OPT_ATTACK = os.environ.get("OPT_ATTACK", "0") == "1"
OPT_EPS = (0.05, 0.10, 0.20)


def run_opt(base, R, w, rng, alpha=0.10):
    """White-box targeted attacks against each rule (score space)."""
    import attacks_opt as AO
    rows = []
    nanparts = dict(E=np.nan, flagged=np.nan, a=np.nan, b=np.nan, family="")
    for eps, direction, rule in itertools.product(OPT_EPS, ("lo", "hi"), AO.RULES):
        Wc, isp = AO.opt_attack(rule, R, w["W"], eps, direction, rng, alpha=alpha)
        th = {rule: AO.RULES[rule](R, Wc, alpha, eps), "oracle": conf_q(Wc[~isp], alpha)}
        d = dict(base); d.update(window=w["tag"], protocol=w.get("protocol", "stale"), attack="opt_" + direction,
                                 target=rule, q=np.nan, spread=np.nan, stretch=np.nan, eps=eps, mode="score",
                                 alpha=alpha, n_tn=len(w["tn"]), n_ta=0 if w["ta"] is None else len(w["ta"]))
        rows += rows_for(d, th, nanparts, w["tn"], w["ta"])
    return rows


def run_windows(base, R, refX, wins, attacks, rng, modes=("score", "feasible"), alpha=None):
    """wins: list of dicts(tag, Wscore, WX, tn, ta, cand_scores, candX)."""
    rows = []
    R0 = R
    for w, alpha in itertools.product(wins, ALPHAS if alpha is None else (alpha,)):
        R = w.get("R", R0)
        refX_w = w.get("refX", refX)
        if OPT_ATTACK and alpha == 0.10:
            rows += run_opt(base, R, w, rng, alpha)
        for (kind, q, sp, st), eps in itertools.product(attacks, EPS):
            if (kind == "none") != (eps == 0.0):
                continue
            for mode in modes:
                if mode == "feasible" and w["cand_scores"] is None:
                    continue
                cs = w["cand_scores"] if mode == "feasible" else None
                Wc, isp, picked, idx = make_poison(kind, rng, eps, w["W"], q=q, spread=sp or 0.05,
                                                   stretch=st or 1.6, cand_scores=cs)
                winX = None
                if mode == "feasible" or eps == 0.0:
                    winX = w["WX"].copy()
                    if picked is not None and len(idx):
                        winX[idx] = w["candX"][picked]
                th, parts = all_thresholds(R, Wc, isp, alpha, eps, ref_X=refX_w, win_X=winX,
                                           R_prev=w.get("R_prev"))
                gap = np.nan
                if picked is not None and len(idx):
                    gap = float(np.mean(np.abs(np.sort(Wc[isp]) - np.sort(w["cand_scores"][picked]))))
                d = dict(base); d.update(window=w["tag"], protocol=w.get("protocol", "stale"), attack=kind, q=q, spread=sp, stretch=st,
                                         eps=eps, mode=mode, alpha=alpha, n_tn=len(w["tn"]),
                                         n_ta=0 if w["ta"] is None else len(w["ta"]))
                rows += rows_for(d, th, parts, w["tn"], w["ta"])
    return rows


# ============================================================ expG: injected drift
def drift_fn(kind, sev, rng, d):
    u = rng.standard_normal(d); u /= np.linalg.norm(u)
    sc = np.ones(d); sub = rng.random(d) < 0.3; sc[sub] = 1 + 0.4 * sev
    if kind == "shift":
        return lambda Z: Z + sev * u
    if kind == "scale":
        return lambda Z: Z * sc
    if kind == "both":
        return lambda Z: Z * sc + sev * u
    raise ValueError(kind)


def expG_unit(ds, det, seed):
    rng = np.random.default_rng(seed_of("G", ds, det, seed))
    X, y = rcad.load_dataset(ds, rng)
    # exact duplicate rows are removed (they make the score law atomic)
    _, keep = np.unique(X, axis=0, return_index=True); keep = np.sort(keep)
    X, y = X[keep], y[keep]
    X = StandardScaler().fit_transform(X)
    # randomized tie-breaking at the input: duplicated rows (about 30 percent of the
    # mammography normals) would otherwise give atoms in the score law
    X = X + INPUT_JITTER * np.random.default_rng(seed_of("jit", ds, seed)).standard_normal(X.shape)
    nrm, anm = np.flatnonzero(y == 0), np.flatnonzero(y == 1); rng.shuffle(nrm); rng.shuffle(anm)
    avail = nrm.size - NREF - M
    n_test = min(NTEST, avail // 3); n_spare = min(1000, avail // 6)
    n_fit = min(rcad.MAX_FIT, avail - n_test - n_spare)
    fit, ref = nrm[:n_fit], nrm[n_fit:n_fit + NREF]
    new = nrm[n_fit + NREF:n_fit + NREF + M + n_test]
    spare = nrm[n_fit + NREF + M + n_test:n_fit + NREF + M + n_test + n_spare]
    win, tst = new[:M], new[M:]
    a_tst, a_cand = anm[: (anm.size + 1) // 2], anm[(anm.size + 1) // 2:]
    fit_fn, score_fn = make_detector(det, seed); fit_fn(X[fit])
    R = score_fn(X[ref]); refX = X[ref]
    wins = []
    for kind, sev in [("none", 0)] + [(k, s) for k in ("shift", "scale", "both") for s in (1, 2)]:
        D = (lambda Z: Z) if kind == "none" else drift_fn(kind, sev, np.random.default_rng(seed_of(ds, seed, kind, sev)), X.shape[1])
        WX = D(X[win])
        crng = np.random.default_rng(seed_of("cand", ds, seed, kind, sev))
        candX = candidate_pool(crng, D(X[spare] if spare.size >= 50 else X[win]), D(X[a_cand]), n_pert=3000)
        wins.append(dict(tag=f"{kind}{sev}", W=score_fn(WX), WX=WX, tn=score_fn(D(X[tst])),
                         ta=score_fn(D(X[a_tst])), cand_scores=score_fn(candX), candX=candX))
    base = dict(exp="G", dataset=ds, detector=det, seed=seed)
    return run_windows(base, R, refX, wins, attack_list(dense=False), rng)


G_DATASETS = ["synthetic", "digits", "mammography", "satellite"]


def expG(task, ntask):
    units = list(itertools.product(G_DATASETS, rcad.DETECTORS + ["DSVDD"], SEEDS))[task::ntask]
    rows = []; t0 = time.time()
    for u in units:
        try:
            rows += expG_unit(*u)
        except Exception as ex:
            print("FAIL", u, repr(ex), flush=True)
        print(u, f"{time.time()-t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(f"results_drift/G_{task:02d}.csv", index=False)


# ============================================================ expH: real streams
TAB_DETS = ["IF", "LOF", "OCSVM", "ECOD", "AE", "DSVDD"]
EMB_DETS = ["kNN", "Maha", "IF", "LOF", "OCSVM", "ECOD", "DSVDD"]
STREAM_DETS = dict(gas=TAB_DETS, creditcard=TAB_DETS, news=EMB_DETS, jailbreak=EMB_DETS, arxiv=EMB_DETS, cifar=EMB_DETS, cicids=TAB_DETS)


def load_stream(name):
    import data_drift
    return data_drift.LOADERS[name]()


def expH_unit(D, det, seed, old_periods=(0,)):
    X, y, per = D["X"], D["y"], D["period"]
    rng = np.random.default_rng(seed_of("H", D["name"], det, seed))
    tab = D["name"] in ("gas", "creditcard", "cicids")
    old_n = np.flatnonzero(np.isin(per, old_periods) & (y == 0)); rng.shuffle(old_n)
    n_fit = min(rcad.MAX_FIT, old_n.size - NREF - M - 100)
    fit, ref, old_rest = old_n[:n_fit], old_n[n_fit:n_fit + NREF], old_n[n_fit + NREF:]
    if tab:
        sc = StandardScaler().fit(X[fit]); T = sc.transform
    else:
        T = lambda Z: Z
    fit_fn, score_fn = make_detector(det, seed); fit_fn(T(X[fit]))
    R = score_fn(T(X[ref])); refX = T(X[ref])
    wins = []
    prev = None
    periods = sorted(set(per.tolist()))
    for p in periods:
        if p in old_periods:
            nn = old_rest
            an = np.flatnonzero(np.isin(per, old_periods) & (y == 1))
            tag = "p0_holdout"
        else:
            nn = np.flatnonzero((per == p) & (y == 0)); an = np.flatnonzero((per == p) & (y == 1))
            tag = f"p{p}"
        nn = nn.copy(); an = an.copy(); rng.shuffle(nn); rng.shuffle(an)
        if nn.size < M + 100:
            continue
        win = nn[:M]; tst = nn[M:M + NTEST]; spare = nn[M + NTEST:]
        a_tst, a_cand = an[: (an.size + 1) // 2], an[(an.size + 1) // 2:]
        if spare.size < 50:                     # attacker draws normals from the test pool's complement
            spare = tst[: tst.size // 2]; tst = tst[tst.size // 2:]
        crng = np.random.default_rng(seed_of("candH", D["name"], seed, p))
        candX = candidate_pool(crng, T(X[spare]), T(X[a_cand]) if a_cand.size else None,
                               n_pert=3000, scales=(0.25, 0.5, 1.0, 2.0) if tab else (0.02, 0.05, 0.1, 0.2))
        if not tab:                             # keep perturbed embeddings on the sphere
            candX = candX / np.linalg.norm(candX, axis=1, keepdims=True)
        WX = T(X[win])
        wd = dict(tag=tag, W=score_fn(WX), WX=WX, tn=score_fn(T(X[tst])),
                  ta=score_fn(T(X[a_tst])) if a_tst.size >= 10 else None,
                  cand_scores=score_fn(candX), candX=candX, protocol="stale",
                  R_prev=score_fn(T(X[prev])) if prev is not None else None)
        wins.append(wd)
        # chained protocol: the reference is a vetted clean sample of the previous period
        if prev is not None:
            wins.append(dict(wd, protocol="chain", R=score_fn(T(X[prev])), refX=T(X[prev]), R_prev=None))
        if p not in old_periods:
            rest = nn[M:]
            prev = rest[-NREF:] if rest.size >= 100 else None
    base = dict(exp="H", dataset=D["name"], detector=det, seed=seed)
    atk = [a for a in attack_list(dense=False) if a[0] != "band" or a[1] in (0.5, 0.8, 0.9, 0.95)]
    return run_windows(base, R, refX, wins, atk, rng, modes=("feasible", "score"))


H_STREAMS = tuple(os.environ.get("H_STREAMS", "gas,creditcard,news,arxiv,cifar,cicids").split(","))


def expH(task, ntask):
    units = [(ds, det, s) for ds in H_STREAMS
             for det in STREAM_DETS[ds] for s in SEEDS][task::ntask]
    rows = []; t0 = time.time(); cache = {}
    for ds, det, s in units:
        try:
            if ds not in cache:
                cache.clear(); cache[ds] = load_stream(ds)
            rows += expH_unit(cache[ds], det, s)
        except Exception as ex:
            import traceback; traceback.print_exc()
            print("FAIL", ds, det, s, repr(ex), flush=True)
        print(ds, det, s, f"{time.time()-t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(f"results_drift/H_{task:02d}.csv", index=False)


if __name__ == "__main__":
    exp, task, ntask = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    os.makedirs("results_drift", exist_ok=True)
    {"G": expG, "H": expH}[exp](task, ntask)
    print("done")

"""
Tabular datasets, classical anomaly detectors and simple calibration rules
(standard split conformal, trimming, reference-based tail correction).

Convention everywhere: larger score = more anomalous.
A test point is flagged anomalous when score > theta.
  type-I error = P(score > theta | normal)
  power        = P(score > theta | anomaly)
"""
import math
import warnings
import numpy as np

warnings.filterwarnings("ignore")

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer, load_digits, fetch_openml

MAX_FIT = 2000          # cap on fitting-set size, for runtime
MAX_TEST_NORMAL = 3000
SEEDS = [42, 1, 7, 13, 21, 100, 200, 300]


# --------------------------------------------------------------- datasets
def _synthetic(rng):
    d, n_norm, n_anom = 12, 3000, 330          # ~10 percent anomalies
    X0 = rng.standard_normal((n_norm, d))
    # deliberately overlapping, so the problem is not trivially separable
    X1 = rng.standard_normal((n_anom, d)) * 1.25 + 0.95
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(n_norm, int), np.ones(n_anom, int)])
    return X, y


def _minority(target):
    """Binary label: 1 for the rarer class, 0 otherwise."""
    lab = np.array([str(t).strip().strip("'") for t in np.asarray(target)])
    vals, cnt = np.unique(lab, return_counts=True)
    return (lab == vals[np.argmin(cnt)]).astype(int)


def load_dataset(name, rng):
    """Return (X, y) with y == 1 marking anomalies."""
    if name == "synthetic":
        return _synthetic(rng)
    if name == "breast_cancer":
        d = load_breast_cancer()
        # sklearn target: 0 = malignant (the minority, 212/569 = 37.3 percent)
        return d.data.astype(float), (d.target == 0).astype(int)
    if name == "digits":
        d = load_digits()
        return d.data.astype(float), (d.target == 8).astype(int)
    if name == "mammography":
        d = fetch_openml("mammography", version=1, as_frame=False,
                         parser="liac-arff", data_home="./sk_data")
        return d.data.astype(float), _minority(d.target)
    if name == "satellite":
        # OpenML data_id 40900: 5,100 x 36, 75 anomalies (1.47 percent)
        d = fetch_openml(data_id=40900, as_frame=False,
                         parser="liac-arff", data_home="./sk_data")
        return d.data.astype(float), _minority(d.target)
    raise ValueError(name)


DATASETS = ["synthetic", "breast_cancer", "digits", "mammography", "satellite"]


# --------------------------------------------------------------- detectors
class ECOD:
    """Empirical cumulative distribution based outlier scores (tail aggregation)."""
    def fit(self, X):
        self.Xs_ = np.sort(X, axis=0)
        self.n_ = X.shape[0]
        return self

    def score(self, X):
        n = self.n_
        out = np.zeros(X.shape[0])
        for j in range(X.shape[1]):
            col = self.Xs_[:, j]
            left = np.searchsorted(col, X[:, j], side="right") / n
            left = np.clip(left, 1.0 / n, 1.0)
            right = np.clip(1.0 - left + 1.0 / n, 1.0 / n, 1.0)
            out += np.maximum(-np.log(left), -np.log(right))
        return out


class AEScorer:
    def __init__(self, seed):
        self.seed = seed

    def fit(self, X):
        d = X.shape[1]
        h = (max(2, d // 2), max(2, d // 4), max(2, d // 2))
        self.m_ = MLPRegressor(hidden_layer_sizes=h, max_iter=300,
                               random_state=self.seed, early_stopping=False)
        self.m_.fit(X, X)
        return self

    def score(self, X):
        return np.mean((self.m_.predict(X) - X) ** 2, axis=1)


def make_detector(name, seed):
    if name == "IF":
        m = IsolationForest(n_estimators=100, random_state=seed)
        return (lambda X: m.fit(X)), (lambda X: -m.decision_function(X))
    if name == "LOF":
        m = LocalOutlierFactor(n_neighbors=20, novelty=True)
        return (lambda X: m.fit(X)), (lambda X: -m.score_samples(X))
    if name == "OCSVM":
        m = OneClassSVM(nu=0.1, gamma="scale")
        return (lambda X: m.fit(X)), (lambda X: -m.decision_function(X))
    if name == "ECOD":
        m = ECOD()
        return (lambda X: m.fit(X)), (lambda X: m.score(X))
    if name == "AE":
        m = AEScorer(seed)
        return (lambda X: m.fit(X)), (lambda X: m.score(X))
    raise ValueError(name)


DETECTORS = ["IF", "LOF", "OCSVM", "ECOD", "AE"]


# --------------------------------------------------------- calibration rules
def idx_standard(m, alpha):
    return int(np.clip(math.ceil((1 - alpha) * (m + 1)), 1, m))


def theta_standard(scores, alpha):
    s = np.sort(scores)
    return s[idx_standard(len(s), alpha) - 1]


def theta_naive_trim(scores, alpha, eps):
    """Drop the top k scores, then take the (1-alpha) quantile of the rest.
    No level correction: this is the Naive-Trim baseline."""
    s = np.sort(scores)
    m = len(s)
    k = int(math.floor(eps * m))
    kept = s[: m - k] if k > 0 else s
    mk = len(kept)
    j = int(np.clip(math.ceil((1 - alpha) * (mk + 1)), 1, mk))
    return kept[j - 1]


def theta_rcad(scores, alpha, eps):
    """Symmetric trim with the level correction alpha' = alpha(1 + 2 eps)."""
    s = np.sort(scores)
    m = len(s)
    k = int(math.floor(eps * m))
    ap = alpha * (1 + 2 * eps)
    j = math.ceil((1 - ap) * (m - 2 * k)) + k
    j = int(np.clip(j, 1, m))
    return s[j - 1]


from statistics import NormalDist as _ND
_Z = _ND().inv_cdf


def tail_excess_ks(cal_scores, ref_scores, delta=0.05):
    """Full-range one-sided two-sample KS excess, debiased by its critical value.
    Kept for comparison only: at m = n = 200 the critical value (0.122) exceeds a
    10 percent tail signal, so this estimator cannot see eps = 0.10."""
    grid = np.unique(np.concatenate([cal_scores, ref_scores]))
    cs, rs = np.sort(cal_scores), np.sort(ref_scores)
    Fc = np.searchsorted(cs, grid, side="right") / cs.size
    Fr = np.searchsorted(rs, grid, side="right") / rs.size
    crit = math.sqrt(math.log(1.0 / delta) / 2.0) * math.sqrt(1.0 / cs.size + 1.0 / rs.size)
    return (max(0.0, float(np.max(Fr - Fc)) - crit),
            max(0.0, float(np.max(Fc - Fr)) - crit))


def tail_excess(cal_scores, ref_scores, qs=(0.80, 0.90, 0.95), delta=0.05, mode="point"):
    """Tail-local test-then-estimate for the contaminated mass in each tail.

    Upper tail.  For t = the q-quantile of the clean reference, fixed-budget
    contamination lying above t gives  S_cal(t) - S_P(t) = eps * F_P(t),  so
        eps_hat(q) = (S_cal(t) - S_ref(t)) / F_ref(t)
    is unbiased.  Its null standard error is sqrt(S(1-S)(1/m + 1/n)) with S the
    pooled tail mass, which is small in the tail.  We test each q at level
    delta / |qs| (Bonferroni) and return the estimate at the most significant
    quantile, or 0 if none is significant.  The lower tail is symmetric.

    mode="lcb" instead returns the one-sided lower confidence bound
        max_q (S_cal - S_ref - z * se) / F_ref,
    which by a union bound over the grid satisfies P(eps_hat > eps) <= delta
    when the contamination lies above the grid quantiles.  Never over-trimming
    is what gives RCAD-U its finite-sample coverage statement.
    """
    m, n = cal_scores.size, ref_scores.size
    z = _Z(1.0 - delta / len(qs))

    def one_side(cal, ref):
        best_stat, best = -np.inf, 0.0
        for q in qs:
            t = np.quantile(ref, q)
            Sc, Sr = float(np.mean(cal > t)), float(np.mean(ref > t))
            Sp = (m * Sc + n * Sr) / (m + n)
            se = math.sqrt(max(Sp * (1 - Sp), 1e-12) * (1.0 / m + 1.0 / n))
            stat = (Sc - Sr) / se
            if mode == "lcb":
                lcb = (Sc - Sr - z * se) / max(1.0 - Sr, 1e-9)
                best = max(best, lcb)
            elif stat > z and stat > best_stat:
                best_stat, best = stat, (Sc - Sr) / max(1.0 - Sr, 1e-9)
        return max(0.0, best)

    e_up = one_side(cal_scores, ref_scores)
    e_low = one_side(-cal_scores, -ref_scores)   # mirror image for the lower tail
    return e_up, e_low


def theta_rcad_u(cal_scores, ref_scores, alpha, estimator="tail"):
    """One-sided, reference driven: trim k_up from the top, shift the index by k_low.
    Uses the conformal (m' + 1) finite-sample index on the retained block."""
    s = np.sort(cal_scores)
    m = len(s)
    if estimator == "tail":
        e_up, e_low = tail_excess(cal_scores, ref_scores, mode="point")
    elif estimator == "lcb":
        e_up, e_low = tail_excess(cal_scores, ref_scores, mode="lcb")
    else:
        e_up, e_low = tail_excess_ks(cal_scores, ref_scores)
    k_up = int(math.floor(e_up * m))
    k_low = int(math.floor(e_low * m))
    k_up = min(k_up, m - 2)
    k_low = min(k_low, m - k_up - 2)
    j = math.ceil((1 - alpha) * (m - k_up - k_low + 1)) + k_low
    j = int(np.clip(j, 1, m))
    return s[j - 1], e_up, e_low


def theta_label_trim(cal_scores, is_poison, alpha):
    """Oracle: remove the points an oracle labels as contamination, then calibrate."""
    clean = np.sort(cal_scores[~is_poison])
    if clean.size == 0:
        return np.inf
    return clean[idx_standard(clean.size, alpha) - 1]


def theta_ref_only(ref_scores, alpha):
    """Ignore the contaminated calibration set; calibrate on the clean reference."""
    return theta_standard(ref_scores, alpha)


# ------------------------------------------------------------------ attacks
def build_candidate_pool(rng, k, cal_clean_X, anom_X, sigma_mult=8.0):
    """Heavy-tailed draws far outside P, plus resampled real anomalies."""
    n_heavy = max(20 * k, 200)
    mu = cal_clean_X.mean(axis=0)
    sd = cal_clean_X.std(axis=0) + 1e-9
    direc = rng.standard_normal((n_heavy, cal_clean_X.shape[1]))
    direc /= (np.linalg.norm(direc, axis=1, keepdims=True) + 1e-12)
    heavy = mu + sigma_mult * sd * direc
    if anom_X.shape[0] > 0 and k > 0:
        idx = rng.integers(0, anom_X.shape[0], size=min(5 * k, 5 * k))
        return np.vstack([heavy, anom_X[idx]])
    return heavy


def poison_scores(kind, rng, k, pool_scores, clean_cal_scores, q=None):
    """Return k poison SCORES placed according to `kind`."""
    if k == 0:
        return np.empty(0)
    if kind == "uppertail":
        return np.sort(pool_scores)[-k:]
    if kind == "bottom":
        lo = np.min(clean_cal_scores)
        spread = np.std(clean_cal_scores) + 1e-9
        return lo - spread * rng.uniform(0.5, 1.5, size=k)
    if kind == "quantile_band":
        # place poison just above the q-quantile of the clean calibration scores
        base = np.quantile(clean_cal_scores, q)
        spread = (np.quantile(clean_cal_scores, 0.99) -
                  np.quantile(clean_cal_scores, 0.50) + 1e-9)
        return base + 0.05 * spread * rng.uniform(0.0, 1.0, size=k)
    raise ValueError(kind)


# ------------------------------------------------------------------- driver
def one_trial(dataset, detector, seed, alpha, eps, attack="uppertail",
              m=200, n_ref=200, q=None, ref_contam=0.0, assumed_eps=None):
    """Run a single (dataset, detector, seed, eps) configuration.

    Returns a dict of per-rule (type-I error, power) plus diagnostics.
    """
    rng = np.random.default_rng(seed)
    X, y = load_dataset(dataset, rng)
    X = StandardScaler().fit_transform(X)

    norm_idx = np.flatnonzero(y == 0)
    anom_idx = np.flatnonzero(y == 1)
    rng.shuffle(norm_idx)
    rng.shuffle(anom_idx)

    n_need = MAX_FIT + m + n_ref
    if norm_idx.size < n_need + 200:
        # scale the budget down for small datasets
        avail = norm_idx.size
        m = min(m, max(40, avail // 5))
        n_ref = min(n_ref, max(40, avail // 5))
    n_fit = min(MAX_FIT, max(50, norm_idx.size - m - n_ref - 100))

    fit_i = norm_idx[:n_fit]
    cal_i = norm_idx[n_fit:n_fit + m]
    ref_i = norm_idx[n_fit + m:n_fit + m + n_ref]
    test_norm_i = norm_idx[n_fit + m + n_ref:][:MAX_TEST_NORMAL]
    half = anom_idx.size // 2
    pool_anom_i, test_anom_i = anom_idx[:half], anom_idx[half:]
    if test_anom_i.size == 0 or test_norm_i.size < 20 or cal_i.size < 20:
        return None

    fit_fn, score_fn = make_detector(detector, seed)
    fit_fn(X[fit_i])

    cal_clean = score_fn(X[cal_i])
    ref_scores = score_fn(X[ref_i])
    test_norm = score_fn(X[test_norm_i])
    test_anom = score_fn(X[test_anom_i])

    m_eff = cal_clean.size
    k = int(math.floor(eps * m_eff))
    a_eps = eps if assumed_eps is None else assumed_eps

    pool = build_candidate_pool(rng, max(k, 1), X[cal_i], X[pool_anom_i])
    pool_scores = score_fn(pool)

    cal = cal_clean.copy()
    is_poison = np.zeros(m_eff, bool)
    if k > 0:
        ps = poison_scores(attack, rng, k, pool_scores, cal_clean, q=q)
        repl = rng.choice(m_eff, size=k, replace=False)   # fixed-budget replacement
        cal[repl] = ps
        is_poison[repl] = True

    # optionally contaminate the reference set as well
    ref_used = ref_scores.copy()
    if ref_contam > 0:
        kr = int(math.floor(ref_contam * ref_used.size))
        if kr > 0:
            pr = poison_scores(attack, rng, kr, pool_scores, ref_scores, q=q)
            ridx = rng.choice(ref_used.size, size=kr, replace=False)
            ref_used[ridx] = pr

    def metrics(theta):
        return (float(np.mean(test_norm > theta)), float(np.mean(test_anom > theta)))

    th_u, e_up, e_low = theta_rcad_u(cal, ref_used, alpha, estimator="tail")
    th_uks, e_up_ks, _ = theta_rcad_u(cal, ref_used, alpha, estimator="ks")
    th_ulcb, e_up_lcb, _ = theta_rcad_u(cal, ref_used, alpha, estimator="lcb")
    rules = {
        "standard":   theta_standard(cal, alpha),
        "naive_trim": theta_naive_trim(cal, alpha, a_eps),
        "rcad":       theta_rcad(cal, alpha, a_eps),
        "rcad_u":     th_u,
        "rcad_u_ks":  th_uks,
        "rcad_u_lcb": th_ulcb,
        "label_trim": theta_label_trim(cal, is_poison, alpha),
        "ref_only":   theta_ref_only(ref_used, alpha),
    }
    out = {"dataset": dataset, "detector": detector, "seed": seed,
           "alpha": alpha, "eps": eps, "assumed_eps": a_eps,
           "attack": attack, "q": q,
           "m": m_eff, "n_ref": int(ref_used.size), "ref_contam": ref_contam,
           "eps_up_hat": e_up, "eps_low_hat": e_low, "eps_up_hat_ks": e_up_ks,
           "eps_up_hat_lcb": e_up_lcb, "k_true": k}
    for name, th in rules.items():
        t1, pw = metrics(th)
        out[f"t1_{name}"] = t1
        out[f"pow_{name}"] = pw
        out[f"theta_{name}"] = float(th)
    return out

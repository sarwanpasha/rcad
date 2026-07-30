"""
RCAD Rebuttal Experiments (E1-E4). Self-contained.
"""
from __future__ import annotations
import argparse
import os
import sys
import time
import warnings
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.datasets import (
    load_breast_cancer, load_digits, make_classification, fetch_openml,
)
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
warnings.filterwarnings("ignore")


def get_anomaly_scores(method, X, method_name):
    n = method_name.lower()
    if "isolation" in n or "ocsvm" in n or "one-class" in n or n == "if":
        return -method.decision_function(X)
    if n == "lof" or "local outlier" in n:
        return -method.score_samples(X)
    if "knn" in n or "k-nearest" in n:
        return method.kneighbors(X)[0][:, -1]
    if "rcad" in n or "bashari" in n or "conformal" in n:
        return method.decision_function(X)
    if hasattr(method, "decision_function"):
        return -method.decision_function(X)
    if hasattr(method, "score_samples"):
        return -method.score_samples(X)
    raise ValueError("Cannot extract anomaly scores from %s" % method_name)


def clopper_pearson(k, n, alpha=0.05):
    if n == 0:
        return (0.0, 1.0)
    lo = stats.beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return (float(lo), float(hi))


@dataclass
class RCADConfig:
    base_detector: str = "isolation_forest"
    alpha: float = 0.10
    epsilon: float = 0.10
    calibration_size: float = 0.30
    robust: bool = True
    random_state: int = 42


class RCAD:
    def __init__(self, cfg):
        self.cfg = cfg
        self.base_detector_ = None
        self.calibration_scores_ = None
        self.threshold_ = None

    def _make_base(self):
        cfg = self.cfg
        cont = 0.1  # PATCH v2: fixed, decoupled from epsilon
        if cfg.base_detector == "isolation_forest":
            return IsolationForest(n_estimators=100, contamination=cont,
                                   random_state=cfg.random_state)
        if cfg.base_detector == "lof":
            return LocalOutlierFactor(n_neighbors=20, contamination=cont,
                                      novelty=True)
        if cfg.base_detector == "ocsvm":
            return OneClassSVM(nu=cont, kernel="rbf", gamma="scale")
        if cfg.base_detector == "knn":
            return NearestNeighbors(n_neighbors=10)
        raise ValueError("Unknown base detector: %s" % cfg.base_detector)

    def _nonconformity(self, X):
        if self.cfg.base_detector == "knn":
            d, _ = self.base_detector_.kneighbors(X)
            return d[:, -1]
        if self.cfg.base_detector == "lof":
            return -self.base_detector_.score_samples(X)
        return -self.base_detector_.decision_function(X)

    def _trimmed_quantile(self, scores, alpha, eps):
        m = len(scores)
        if not self.cfg.robust or eps <= 0:
            q = np.clip(np.ceil((m + 1) * (1 - alpha)) / m, 0.0, 1.0)
            return float(np.quantile(scores, q))
        k = int(np.floor(eps * m))
        if 2 * k >= m - 2:
            return float(np.quantile(scores, 1 - alpha))
        sorted_s = np.sort(scores)
        trimmed = sorted_s[k:m - k]
        alpha_adj = min(alpha * (1 + 2 * eps), 0.99)
        return float(np.quantile(trimmed, 1 - alpha_adj))

    def fit(self, X):
        rng = np.random.RandomState(self.cfg.random_state)
        idx = rng.permutation(len(X))
        m = int(self.cfg.calibration_size * len(X))
        cal_idx, fit_idx = idx[:m], idx[m:]
        return self.fit_with_split(X[fit_idx], X[cal_idx])

    def fit_with_split(self, X_fit, X_cal):
        self.base_detector_ = self._make_base()
        self.base_detector_.fit(X_fit)
        self.calibration_scores_ = self._nonconformity(X_cal)
        self.threshold_ = self._trimmed_quantile(
            self.calibration_scores_, self.cfg.alpha, self.cfg.epsilon)
        return self

    def decision_function(self, X):
        return self._nonconformity(X)

    def predict(self, X):
        s = self._nonconformity(X)
        return np.where(s > self.threshold_, -1, 1)

    def p_values(self, X):
        s = self._nonconformity(X)
        m = len(self.calibration_scores_)
        return np.array([
            (np.sum(self.calibration_scores_ >= s_i) + 1) / (m + 1)
            for s_i in s])


class BashariBaseline:
    def __init__(self, mode="standard", base_detector="isolation_forest",
                 alpha=0.10, contamination=0.10, random_state=42):
        assert mode in {"standard", "naive_trim", "label_trim"}
        self.mode = mode
        self.alpha = alpha
        self.contamination = contamination
        self.base_name = base_detector
        self.random_state = random_state
        self.base_detector_ = None
        self.threshold_ = None
        self.calibration_scores_ = None

    def _make_base(self):
        cont = 0.1  # PATCH v2: fixed, decoupled from epsilon
        if self.base_name == "isolation_forest":
            return IsolationForest(n_estimators=100, contamination=cont,
                                   random_state=self.random_state)
        if self.base_name == "lof":
            return LocalOutlierFactor(n_neighbors=20, contamination=cont,
                                      novelty=True)
        if self.base_name == "ocsvm":
            return OneClassSVM(nu=cont, kernel="rbf", gamma="scale")
        raise ValueError(self.base_name)

    def _nonconformity(self, X):
        if self.base_name == "lof":
            return -self.base_detector_.score_samples(X)
        return -self.base_detector_.decision_function(X)

    def fit_with_split(self, X_fit, X_cal, y_cal=None):
        self.base_detector_ = self._make_base()
        self.base_detector_.fit(X_fit)
        scores = self._nonconformity(X_cal)
        if self.mode == "standard":
            self.calibration_scores_ = scores
        elif self.mode == "naive_trim":
            k = int(np.floor(self.contamination * len(scores)))
            self.calibration_scores_ = (
                np.sort(scores)[:len(scores) - k] if k > 0 else scores)
        elif self.mode == "label_trim":
            if y_cal is None:
                raise ValueError("label_trim requires y_cal")
            self.calibration_scores_ = scores[y_cal == 0]
        m = len(self.calibration_scores_)
        q = np.clip(np.ceil((m + 1) * (1 - self.alpha)) / m, 0.0, 1.0)
        self.threshold_ = float(np.quantile(self.calibration_scores_, q))
        return self

    def decision_function(self, X):
        return self._nonconformity(X)

    def predict(self, X):
        s = self._nonconformity(X)
        return np.where(s > self.threshold_, -1, 1)

    def p_values(self, X):
        s = self._nonconformity(X)
        m = len(self.calibration_scores_)
        return np.array([
            (np.sum(self.calibration_scores_ >= s_i) + 1) / (m + 1)
            for s_i in s])


def gen_in_distribution(X_norm, X_anom, k, rng):
    if k <= 0 or len(X_anom) == 0:
        return np.empty((0, X_norm.shape[1]))
    idx = rng.choice(len(X_anom), size=k, replace=True)
    return X_anom[idx]


def gen_heavy_tail(X_norm, X_anom, k, rng, scale=8.0):
    if k <= 0:
        return np.empty((0, X_norm.shape[1]))
    d = X_norm.shape[1]
    mu = X_norm.mean(axis=0)
    sigma = X_norm.std(axis=0) + 1e-6
    direction = rng.randn(k, d)
    direction /= (np.linalg.norm(direction, axis=1, keepdims=True) + 1e-9)
    out = mu + scale * sigma * direction
    out += 0.5 * sigma * rng.randn(k, d)
    return out


def gen_adversarial_whitebox(X_norm, X_anom, k, rng, defender_detector):
    if k <= 0:
        return np.empty((0, X_norm.shape[1]))
    n_pool = max(20 * k, 200)
    heavy = gen_heavy_tail(X_norm, X_anom, n_pool, rng, scale=8.0)
    if len(X_anom) > 0:
        idx = rng.choice(len(X_anom), size=min(n_pool, 5 * k), replace=True)
        pool = np.vstack([heavy, X_anom[idx]])
    else:
        pool = heavy
    scores = -defender_detector.decision_function(pool)
    top = np.argsort(scores)[-k:]
    return pool[top]


def gen_adversarial_partial(X_norm, X_anom, k, rng, attacker_detector):
    if k <= 0:
        return np.empty((0, X_norm.shape[1]))
    n_pool = max(20 * k, 200)
    heavy = gen_heavy_tail(X_norm, X_anom, n_pool, rng, scale=8.0)
    if len(X_anom) > 0:
        idx = rng.choice(len(X_anom), size=min(n_pool, 5 * k), replace=True)
        pool = np.vstack([heavy, X_anom[idx]])
    else:
        pool = heavy
    scores = -attacker_detector.decision_function(pool)
    top = np.argsort(scores)[-k:]
    return pool[top]


def _binarize_labels(y_raw):
    y = np.asarray(y_raw).ravel()
    try:
        y_num = y.astype(float)
        unique = np.unique(y_num)
        if len(unique) != 2:
            return None
        counts = {v: int(np.sum(y_num == v)) for v in unique}
        anomaly_val = min(counts, key=counts.get)
        return (y_num == anomaly_val).astype(int)
    except (ValueError, TypeError):
        pass
    y_str = y.astype(str)
    unique = np.unique(y_str)
    if len(unique) < 2:
        return None
    if len(unique) > 2:
        counts = {v: int(np.sum(y_str == v)) for v in unique}
        majority = max(counts, key=counts.get)
        return (y_str != majority).astype(int)
    counts = {v: int(np.sum(y_str == v)) for v in unique}
    anomaly_val = min(counts, key=counts.get)
    return (y_str == anomaly_val).astype(int)


def _try_openml(name, min_rate=0.005, max_rate=0.45):
    try:
        ds = fetch_openml(name=name, as_frame=False, parser="liac-arff")
        X = np.asarray(ds.data, dtype=float)
        y = _binarize_labels(ds.target)
        if y is None:
            return None
        rate = float(y.mean())
        if not (min_rate <= rate <= max_rate):
            if min_rate <= 1 - rate <= max_rate:
                y = 1 - y
            else:
                return None
        return X, y
    except Exception as e:
        print("  [OpenML fetch failed for %s: %s] using synthetic." %
              (name, type(e).__name__))
        return None


def _synthetic_surrogate(name, n_samples, n_features, anomaly_rate, seed=42):
    Xs, ys = make_classification(
        n_samples=n_samples, n_features=n_features,
        n_informative=max(2, int(0.7 * n_features)),
        n_redundant=max(0, int(0.15 * n_features)),
        weights=[1 - anomaly_rate, anomaly_rate], flip_y=0,
        random_state=seed)
    return Xs, ys



_LOCAL_NPZ = {
    "mammography": "data/23_mammography.npz",
    "satellite": "data/30_satellite.npz",
}
# --- artifact portability patch (added for anonymous reviewer release) ---
# Prefer the datasets bundled next to this script (code/data/*.npz). These are
# the standard public ODDS/OpenML anomaly-detection benchmarks. If absent, the
# code falls back to OpenML (use_openml=True) or a seeded synthetic surrogate.
import os as _os_patch
_DATA_DIR = _os_patch.path.join(_os_patch.path.dirname(_os_patch.path.abspath(__file__)), "data")
for _k, _fn in (("mammography", "23_mammography.npz"), ("satellite", "30_satellite.npz")):
    _cand = _os_patch.path.join(_DATA_DIR, _fn)
    if _os_patch.path.exists(_cand):
        _LOCAL_NPZ[_k] = _cand
# --- end patch ---



def _load_local(name):
    import os as _os
    p = _LOCAL_NPZ.get(name)
    if p and _os.path.exists(p):
        d = np.load(p)
        X = np.asarray(d["X"], dtype=float)
        y = np.asarray(d["y"]).astype(int).ravel()
        if not (0.005 <= float(y.mean()) <= 0.45):
            if 0.005 <= 1 - float(y.mean()) <= 0.45:
                y = 1 - y
        return X, y
    return None


def load_datasets(use_openml=True):
    out = {}
    bc = load_breast_cancer()
    out["breast_cancer"] = (bc.data, (bc.target == 0).astype(int))
    d = load_digits()
    out["digits"] = (d.data, (d.target == 8).astype(int))
    Xs, ys = make_classification(
        n_samples=2000, n_features=20, n_informative=15, n_redundant=5,
        n_clusters_per_class=2, weights=[0.9, 0.1], flip_y=0,
        random_state=42)
    out["synthetic"] = (Xs, ys)
    mammo = _load_local("mammography")
    if mammo is None:
        mammo = _try_openml("mammography") if use_openml else None
    if mammo is None:
        print("  Using synthetic mammography surrogate.")
        mammo = _synthetic_surrogate("mammography", 11183, 6, 0.023, seed=1)
    out["mammography"] = mammo
    sat = _load_local("satellite")
    if sat is None and use_openml:
        for candidate in ("satellite", "Satellite"):
            sat = _try_openml(candidate)
            if sat is not None:
                break
    if sat is None:
        print("  Using synthetic satellite surrogate.")
        sat = _synthetic_surrogate("satellite", 5100, 36, 0.015, seed=2)
    out["satellite"] = sat
    print("Datasets loaded: %s" % list(out.keys()))
    for n, (X, y) in out.items():
        print("  %s: n=%d, d=%d, anomaly_rate=%.3f" %
              (n, len(X), X.shape[1], float(np.mean(y))))
    return out


def _fit_base(base_detector, X_fit, seed):
    if base_detector == "isolation_forest":
        return IsolationForest(n_estimators=100, contamination=0.1,
                               random_state=seed).fit(X_fit)
    if base_detector == "lof":
        return LocalOutlierFactor(n_neighbors=20, contamination=0.1,
                                  novelty=True).fit(X_fit)
    if base_detector == "ocsvm":
        return OneClassSVM(nu=0.1, kernel="rbf", gamma="scale").fit(X_fit)
    raise ValueError(base_detector)


def one_trial(X, y, seed, alpha, eps_true, eps_assumed, dataset_name,
              base_detector="isolation_forest", regime="adversarial",
              m_cal_target=200, attacker_kind="whitebox",
              include_bashari=True):
    rng = np.random.RandomState(seed)
    try:
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.30, random_state=seed, stratify=y)
    except ValueError:
        return []
    X_norm = Xtr[ytr == 0]
    X_anom_train = Xtr[ytr == 1]
    if len(X_norm) < 150 or len(X_anom_train) < 5 or yte.sum() < 5:
        return []
    sc = StandardScaler().fit(X_norm)
    X_norm = sc.transform(X_norm)
    X_anom_train = sc.transform(X_anom_train)
    Xte_s = sc.transform(Xte)
    n = len(X_norm)
    m_cal = min(m_cal_target, n - 50)
    perm = rng.permutation(n)
    X_fit = X_norm[perm[m_cal:]]
    X_cal_clean = X_norm[perm[:m_cal]]
    if attacker_kind in ("partial", "whitebox_half"):
        n_fit = len(X_fit)
        if n_fit < 100:
            return []
        half = n_fit // 2
        X_fit_def = X_fit[:half]
        X_fit_pub = X_fit[half:]
        defender_base = _fit_base(base_detector, X_fit_def, seed)
        if attacker_kind == "partial":
            attacker_base = IsolationForest(
                n_estimators=100, contamination=0.1,
                random_state=seed + 9999).fit(X_fit_pub)
        else:
            attacker_base = defender_base
        X_fit_for_methods = X_fit_def
    else:
        defender_base = _fit_base(base_detector, X_fit, seed)
        attacker_base = defender_base
        X_fit_for_methods = X_fit
    k_inject = int(np.floor(eps_true * m_cal))
    if regime == "in_distribution":
        X_inject = gen_in_distribution(X_norm, X_anom_train, k_inject, rng)
    elif regime == "heavy_tail":
        X_inject = gen_heavy_tail(X_norm, X_anom_train, k_inject, rng, 8.0)
    elif regime == "adversarial":
        X_inject = gen_adversarial_whitebox(
            X_norm, X_anom_train, k_inject, rng, attacker_base)
    else:
        raise ValueError(regime)
    if k_inject > 0 and len(X_inject) > 0:
        keep = rng.permutation(m_cal)[:m_cal - k_inject]
        X_cal = np.vstack([X_cal_clean[keep], X_inject])
        y_cal = np.concatenate([
            np.zeros(m_cal - k_inject, dtype=int),
            np.ones(k_inject, dtype=int)])
    else:
        X_cal = X_cal_clean.copy()
        y_cal = np.zeros(m_cal, dtype=int)
    methods = [
        ("RCAD (Robust)",
         RCAD(RCADConfig(alpha=alpha, epsilon=max(eps_assumed, 1e-6),
                         robust=True, base_detector=base_detector,
                         random_state=seed)), False),
        ("RCAD (Standard)",
         RCAD(RCADConfig(alpha=alpha, epsilon=0.0, robust=False,
                         base_detector=base_detector,
                         random_state=seed)), False),
    ]
    if include_bashari:
        methods.extend([
            ("Bashari (Standard)",
             BashariBaseline(mode="standard", base_detector=base_detector,
                             alpha=alpha, contamination=eps_assumed,
                             random_state=seed), False),
            ("Bashari (Label-Trim)",
             BashariBaseline(mode="label_trim", base_detector=base_detector,
                             alpha=alpha, contamination=eps_assumed,
                             random_state=seed), True),
        ])
    rows = []
    for name, mthd, needs_y in methods:
        try:
            if isinstance(mthd, BashariBaseline):
                mthd.fit_with_split(X_fit_for_methods, X_cal,
                                    y_cal=(y_cal if needs_y else None))
            else:
                mthd.fit_with_split(X_fit_for_methods, X_cal)
            scores = get_anomaly_scores(mthd, Xte_s, name)
            yhat_anom = (scores > mthd.threshold_).astype(int)
            normal_mask = (yte == 0)
            anom_mask = (yte == 1)
            n_norm = int(normal_mask.sum())
            n_anom = int(anom_mask.sum())
            p = mthd.p_values(Xte_s)
            covered = int(np.sum(p[normal_mask] > alpha))
            cov = covered / n_norm if n_norm else np.nan
            ci_lo, ci_hi = clopper_pearson(covered, n_norm)
            tp = int(np.sum(yhat_anom[anom_mask] == 1))
            power = tp / n_anom if n_anom else np.nan
            fp = int(np.sum(yhat_anom[normal_mask] == 1))
            fpr = fp / n_norm if n_norm else np.nan
            try:
                auc = roc_auc_score(yte, scores)
            except ValueError:
                auc = np.nan
            try:
                pr = average_precision_score(yte, scores)
            except ValueError:
                pr = np.nan
            rows.append(dict(
                dataset=dataset_name, seed=seed, alpha=alpha,
                eps_true=eps_true, eps_assumed=eps_assumed,
                base_detector=base_detector, regime=regime,
                attacker_kind=attacker_kind, method=name,
                coverage=cov, coverage_ci_lo=ci_lo, coverage_ci_hi=ci_hi,
                type1=fpr, power=power, auc_roc=auc, auc_pr=pr,
                threshold=float(mthd.threshold_),
                m_cal=m_cal, n_test=len(yte)))
        except Exception as e:
            print("  [skip] %s/%s/%s: %s: %s" %
                  (dataset_name, seed, name, type(e).__name__, e),
                  file=sys.stderr)
    return rows


def _agg_and_write(df, out_dir, exp, group_cols, single_seed):
    suffix = "_seed%d" % single_seed if single_seed is not None else ""
    out = os.path.join(out_dir, exp, "%s_raw%s.csv" % (exp, suffix))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print("[%s] wrote %d rows -> %s" % (exp.upper(), len(df), out))
    if len(df) and single_seed is None:
        agg = (df.groupby(group_cols)
               .agg(coverage_mean=("coverage", "mean"),
                    coverage_std=("coverage", "std"),
                    power_mean=("power", "mean"),
                    power_std=("power", "std"),
                    type1_mean=("type1", "mean"),
                    threshold_mean=("threshold", "mean"),
                    n=("power", "count")).reset_index())
        agg.to_csv(os.path.join(out_dir, exp, "%s_summary.csv" % exp),
                   index=False)
        print("[%s] summary rows: %d" % (exp.upper(), len(agg)))


def _loop(datasets, ds_names, combos, exp_tag, total):
    rows = []
    done = 0
    t0 = time.time()
    for r in combos(datasets, ds_names):
        rows.extend(r)
        done += 1
        if done % 20 == 0 or done == total:
            dt = time.time() - t0
            eta = dt / done * (total - done) if done else 0
            print("  [%s] %d/%d (%.1fs elapsed, ETA %.1fs)" %
                  (exp_tag, done, total, dt, eta))
    return rows


def run_e1(datasets, out_dir, seeds, single_seed):
    print("\n" + "=" * 72 + "\n[E1] epsilon-misspecification sweep\n" + "=" * 72)
    ds_names = [d for d in ["breast_cancer", "mammography", "satellite"]
                if d in datasets]
    eps_trues = [0.05, 0.10, 0.20]
    eps_assumeds = [0.0, 0.05, 0.10, 0.20, 0.30]
    alphas = [0.05, 0.10]
    use_seeds = [single_seed] if single_seed is not None else seeds
    total = (len(ds_names) * len(use_seeds) * len(alphas)
             * len(eps_trues) * len(eps_assumeds))
    print("  %d trials" % total)
    rows = []
    done = 0
    t0 = time.time()
    for ds_name in ds_names:
        X, y = datasets[ds_name]
        for seed in use_seeds:
            for alpha in alphas:
                for et in eps_trues:
                    for ea in eps_assumeds:
                        rows.extend(one_trial(
                            X, y, seed, alpha, et, ea, ds_name,
                            base_detector="isolation_forest",
                            regime="adversarial",
                            attacker_kind="whitebox",
                            include_bashari=True))
                        done += 1
                        if done % 20 == 0 or done == total:
                            dt = time.time() - t0
                            eta = dt / done * (total - done) if done else 0
                            print("  [E1] %d/%d (%.1fs, ETA %.1fs)" %
                                  (done, total, dt, eta))
    df = pd.DataFrame(rows)
    _agg_and_write(df, out_dir, "e1",
                   ["dataset", "alpha", "eps_true", "eps_assumed", "method"],
                   single_seed)
    return df


def run_e2(datasets, out_dir, seeds, single_seed):
    print("\n" + "=" * 72 + "\n[E2] multiple base detectors\n" + "=" * 72)
    ds_names = [d for d in ["breast_cancer", "mammography", "satellite",
                            "synthetic", "digits"] if d in datasets]
    base_detectors = ["isolation_forest", "lof", "ocsvm"]
    alphas = [0.05, 0.10]
    eps_grid = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    use_seeds = [single_seed] if single_seed is not None else seeds
    total = (len(base_detectors) * len(ds_names) * len(use_seeds)
             * len(alphas) * len(eps_grid))
    print("  %d trials" % total)
    rows = []
    done = 0
    t0 = time.time()
    for bd in base_detectors:
        for ds_name in ds_names:
            X, y = datasets[ds_name]
            for seed in use_seeds:
                for alpha in alphas:
                    for eps in eps_grid:
                        rows.extend(one_trial(
                            X, y, seed, alpha, eps, eps, ds_name,
                            base_detector=bd, regime="adversarial",
                            attacker_kind="whitebox",
                            include_bashari=True))
                        done += 1
                        if done % 20 == 0 or done == total:
                            dt = time.time() - t0
                            eta = dt / done * (total - done) if done else 0
                            print("  [E2] %d/%d (%.1fs, ETA %.1fs)" %
                                  (done, total, dt, eta))
    df = pd.DataFrame(rows)
    _agg_and_write(df, out_dir, "e2",
                   ["base_detector", "dataset", "alpha", "eps_true", "method"],
                   single_seed)
    return df


def run_e3(datasets, out_dir, seeds, single_seed):
    print("\n" + "=" * 72 + "\n[E3] partial-knowledge attacker\n" + "=" * 72)
    ds_names = [d for d in ["breast_cancer", "mammography", "satellite",
                            "synthetic", "digits"] if d in datasets]
    eps_grid = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    alpha = 0.10
    use_seeds = [single_seed] if single_seed is not None else seeds
    total = len(ds_names) * len(use_seeds) * len(eps_grid) * 2
    print("  %d trials" % total)
    rows = []
    done = 0
    t0 = time.time()
    for ds_name in ds_names:
        X, y = datasets[ds_name]
        for seed in use_seeds:
            for eps in eps_grid:
                for ak in ("whitebox", "partial", "whitebox_half"):
                    rows.extend(one_trial(
                        X, y, seed, alpha, eps, eps, ds_name,
                        base_detector="isolation_forest",
                        regime="adversarial", attacker_kind=ak,
                        include_bashari=False))
                    done += 1
                    if done % 20 == 0 or done == total:
                        dt = time.time() - t0
                        eta = dt / done * (total - done) if done else 0
                        print("  [E3] %d/%d (%.1fs, ETA %.1fs)" %
                              (done, total, dt, eta))
    df = pd.DataFrame(rows)
    _agg_and_write(df, out_dir, "e3",
                   ["dataset", "attacker_kind", "eps_true", "method"],
                   single_seed)
    return df


def _e4_trial_at_config(X, y, seed, alpha, eps, m_cal_target, regime):
    return one_trial(
        X, y, seed, alpha, eps_true=eps, eps_assumed=eps,
        dataset_name="synthetic_big",
        base_detector="isolation_forest",
        regime=regime, attacker_kind="whitebox",
        include_bashari=False, m_cal_target=m_cal_target)


def run_e4(datasets, out_dir, seeds, single_seed):
    print("\n" + "=" * 72 + "\n[E4] two-term coverage bound validation\n" + "=" * 72)
    Xbig, ybig = make_classification(
        n_samples=8000, n_features=20, n_informative=15, n_redundant=5,
        n_clusters_per_class=2, weights=[0.9, 0.1], flip_y=0,
        random_state=42)
    alpha = 0.10
    use_seeds = [single_seed] if single_seed is not None else seeds
    m_grid = [50, 100, 200, 400, 800, 1600, 3200]
    eps_fixed = 0.10
    m_fixed = 200
    eps_grid = [0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30]
    sub_configs = (
        [("A", "in_distribution", "m", m, eps_fixed) for m in m_grid]
        + [("B", "in_distribution", "eps", m_fixed, e) for e in eps_grid]
        + [("C", "heavy_tail", "eps", m_fixed, e) for e in eps_grid])
    total = len(use_seeds) * len(sub_configs)
    print("  %d trials (A=sweep m in-dist, B=sweep eps in-dist, C=sweep eps heavy-tail)" % total)
    rows = []
    done = 0
    t0 = time.time()
    for seed in use_seeds:
        for sweep_id, regime, sweep_var, m_target, eps in sub_configs:
            rs = _e4_trial_at_config(Xbig, ybig, seed, alpha, eps, m_target, regime)
            for r in rs:
                r["sweep_id"] = sweep_id
                r["sweep_var"] = sweep_var
                r["regime_e4"] = regime
                r["m_cal_target"] = m_target
            rows.extend(rs)
            done += 1
            if done % 5 == 0 or done == total:
                dt = time.time() - t0
                eta = dt / done * (total - done) if done else 0
                print("  [E4-%s] %d/%d (%.1fs, ETA %.1fs)" %
                      (sweep_id, done, total, dt, eta))
    df = pd.DataFrame(rows)
    if len(df):
        df["coverage_error"] = (1 - df["alpha"]) - df["coverage"]
        df["coverage_error_pos"] = df["coverage_error"].clip(lower=0)
        df["predicted_bias_term"] = df["eps_true"]
        df["predicted_sampling_term"] = 1.0 / np.sqrt(df["m_cal"].clip(lower=1))
    suffix = "_seed%d" % single_seed if single_seed is not None else ""
    out = os.path.join(out_dir, "e4", "e4_raw%s.csv" % suffix)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print("[E4] wrote %d rows -> %s" % (len(df), out))
    if len(df) and single_seed is None:
        for sweep_id in ("A", "B", "C"):
            sub = df[df["sweep_id"] == sweep_id]
            if sub.empty:
                continue
            group_cols = (["method", "m_cal", "m_cal_target"] if sweep_id == "A"
                          else ["method", "eps_true"])
            agg = (sub.groupby(group_cols)
                   .agg(coverage_mean=("coverage", "mean"),
                        coverage_std=("coverage", "std"),
                        coverage_err_mean=("coverage_error", "mean"),
                        coverage_err_pos_mean=("coverage_error_pos", "mean"),
                        power_mean=("power", "mean"),
                        threshold_mean=("threshold", "mean"),
                        n=("power", "count")).reset_index())
            agg.to_csv(os.path.join(out_dir, "e4", "e4_summary_%s.csv" % sweep_id),
                       index=False)
            print("[E4-%s] summary rows: %d" % (sweep_id, len(agg)))
    return df


DEFAULT_SEEDS = [42, 1, 7, 13, 21, 100, 200, 300]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", "-e",
                    choices=["e1", "e2", "e3", "e4", "all"], default="all")
    ap.add_argument("--out-dir", "-o", default="./rcad_rebuttal_results")
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--single-seed", type=int, default=None)
    ap.add_argument("--no-openml", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    print("=" * 72 + "\nRCAD Rebuttal Experiments\n" + "=" * 72)
    print("experiment=%s out_dir=%s seeds=%s single_seed=%s openml=%s" %
          (args.experiment, args.out_dir, args.seeds, args.single_seed,
           not args.no_openml))
    t0 = time.time()
    print("Loading datasets...")
    datasets = load_datasets(use_openml=not args.no_openml)
    to_run = ([args.experiment] if args.experiment != "all"
              else ["e1", "e2", "e3", "e4"])
    fns = {"e1": run_e1, "e2": run_e2, "e3": run_e3, "e4": run_e4}
    for exp in to_run:
        fns[exp](datasets, args.out_dir, args.seeds, args.single_seed)
    dt = time.time() - t0
    print("\n" + "=" * 72)
    print("All done in %.1fs (%.1f min). Results in %s/" %
          (dt, dt / 60, args.out_dir))
    print("=" * 72)


if __name__ == "__main__":
    main()

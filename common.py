"""Shared pieces for the drift experiments: detectors, poisoning of a window in
score space or with real inputs, and the per-rule metrics."""
import math
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors

import rcad
from attacks import poison_window
from baselines import all_thresholds

ALPHA = 0.10
RULES_ORDER = ["standard", "ref_only", "pooled", "naive_trim", "naive_trim_e05", "naive_trim_e20",
               "rcad", "rcad_u", "qtc", "weighted", "drcc", "oracle"]


class KNNScorer:
    """Mean distance to the k nearest training points (cosine geometry on unit vectors)."""
    def __init__(self, k=10):
        self.k = k

    def fit(self, X):
        self.nn_ = NearestNeighbors(n_neighbors=self.k).fit(X)
        return self

    def score(self, X):
        d, _ = self.nn_.kneighbors(X)
        return d.mean(1)


class MahaScorer:
    """Mahalanobis distance under a Ledoit-Wolf shrunk covariance."""
    def fit(self, X):
        self.mu_ = X.mean(0)
        self.P_ = LedoitWolf().fit(X).precision_
        return self

    def score(self, X):
        Z = X - self.mu_
        return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", Z, self.P_, Z), 0))


class DeepSVDD:
    """One-class deep SVDD (Ruff et al. 2018): a bias-free MLP mapping normals close to a
    fixed centre; score = squared distance to the centre."""
    def __init__(self, seed, epochs=40, h=(128, 32)):
        self.seed, self.epochs, self.h = seed, epochs, h

    def fit(self, X):
        import torch
        torch.manual_seed(self.seed)
        d = X.shape[1]
        self.net = torch.nn.Sequential(torch.nn.Linear(d, self.h[0], bias=False), torch.nn.LeakyReLU(),
                                       torch.nn.Linear(self.h[0], self.h[1], bias=False))
        Xt = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            c = self.net(Xt).mean(0); c[c.abs() < 0.01] = 0.01
        self.c = c
        opt = torch.optim.Adam(self.net.parameters(), lr=1e-3, weight_decay=1e-5)
        n = Xt.shape[0]; g = torch.Generator().manual_seed(self.seed)
        for _ in range(self.epochs):
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, 128):
                z = self.net(Xt[perm[i:i + 128]])
                loss = ((z - self.c) ** 2).sum(1).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        return self

    def score(self, X):
        import torch
        with torch.no_grad():
            z = self.net(torch.tensor(X, dtype=torch.float32))
            return ((z - self.c) ** 2).sum(1).numpy().astype(float)


def make_detector(name, seed):
    if name == "DSVDD":
        m = DeepSVDD(seed)
        return (lambda X: m.fit(X)), (lambda X: m.score(X))
    if name == "kNN":
        m = KNNScorer()
    elif name == "Maha":
        m = MahaScorer()
    else:
        return rcad.make_detector(name, seed)
    return (lambda X: m.fit(X)), (lambda X: m.score(X))


def candidate_pool(rng, Xn, Xa, n_pert=4000, scales=(0.25, 0.5, 1.0, 2.0, 4.0)):
    """Real or near-real inputs an attacker could submit: held-out normals and
    anomalies of the current period, plus noisy copies of normals at several scales.
    Returns the candidate feature matrix."""
    sd = Xn.std(0) + 1e-9
    base = Xn[rng.integers(0, Xn.shape[0], n_pert)]
    sc = rng.choice(scales, n_pert)[:, None]
    pert = base + sc * sd * rng.standard_normal(base.shape)
    parts = [Xn, pert] + ([Xa] if Xa is not None and len(Xa) else [])
    return np.vstack(parts)


def attack_list(dense=False):
    """(kind, q, spread, stretch) tuples."""
    out = [("none", None, None, None), ("top", None, None, None), ("bottom", None, None, None),
           ("mimic", None, None, 1.6)]
    qs = (0.3, 0.5, 0.7, 0.8, 0.9, 0.95)
    out += [("band", q, 0.05, None) for q in qs]
    if dense:
        out += [("band", q, sp, None) for q in (0.2, 0.4, 0.6, 0.75, 0.85, 0.92, 0.97, 0.99)
                for sp in (0.05,)]
        out += [("band", q, sp, None) for q in (0.5, 0.8, 0.9, 0.95) for sp in (0.3, 1.0, 2.0)]
        out += [("mimic", None, None, st) for st in (1.2, 1.4, 2.0, 2.5)]
        out += [("split", q, 0.05, None) for q in (0.5, 0.8, 0.9)]
    return out


def make_poison(kind, rng, eps, W, q=None, spread=0.05, stretch=1.6, cand_scores=None):
    """Wrapper around poison_window with the extra adaptive knobs."""
    import attacks
    m = W.size; k = int(math.floor(eps * m))
    if kind == "none" or k == 0:
        return W.copy(), np.zeros(m, bool), None, np.zeros(0, int)
    if kind == "split":                       # half on the top, half in a band
        sd = np.std(W) + 1e-9
        k1 = k // 2
        want = np.r_[np.max(W) + sd * rng.uniform(0.5, 3.0, k1),
                     np.quantile(W, q) + spread * sd * rng.uniform(0, 1, k - k1)]
    elif kind == "mimic":
        sd = np.std(W) + 1e-9; med = np.median(W)
        base = rng.choice(W[W > med], size=k, replace=True)
        want = med + stretch * (base - med) + 0.02 * sd * rng.standard_normal(k)
    else:
        want = attacks.target_scores(kind, rng, k, W, q=q, spread=spread)
    Wc = W.copy(); isp = np.zeros(m, bool); picked = None
    if cand_scores is not None:
        picked = attacks.feasible_pick(want, cand_scores, rng)
        vals = cand_scores[picked]
    else:
        vals = want
    idx = rng.choice(m, k, replace=False)
    Wc[idx] = vals; isp[idx] = True
    return Wc, isp, picked, idx


def evaluate(th, tn, ta):
    """type-I on test normals, power on test anomalies, per rule."""
    tn = np.sort(tn); ta = np.sort(ta) if ta is not None and len(ta) else None
    res = {}
    for r, t in th.items():
        if t is None or not np.isfinite(t):
            fa = 0.0 if t == np.inf else np.nan
            pw = 0.0 if t == np.inf else np.nan
        else:
            fa = 1.0 - np.searchsorted(tn, t, side="right") / tn.size
            pw = (1.0 - np.searchsorted(ta, t, side="right") / ta.size) if ta is not None else np.nan
        res[r] = (fa, pw)
    return res


def rows_for(base, th, parts, tn, ta):
    ev = evaluate(th, tn, ta)
    rows = []
    for r, (fa, pw) in ev.items():
        d = dict(base); d.update(rule=r, typeI=fa, power=pw, theta=th[r])
        if r == "drcc":
            d.update(E_hat=parts["E"], flagged=parts["flagged"], map_a=parts["a"], map_b=parts["b"],
                     family=parts.get("family", "affine"), Ec=parts.get("Ec", np.nan), Pund=parts.get("Pund", np.nan))
        rows.append(d)
    return rows

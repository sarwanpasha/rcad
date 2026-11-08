"""Summaries of the experiment outputs.  Writes CSV tables to results_drift/tables/ and prints a
compact text report.

  main_<exp>.csv        per dataset x rule (alpha = 0.10, standard attacks): maxI, viol, meanI,
                        dpow (power - oracle power) with 95% bootstrap CIs over seeds
  alpha_<exp>.csv       per alpha x dataset x rule: viol with tolerance 0.2 alpha, mean type-I
  detector_<exp>.csv    per dataset x detector x rule
  opt_<exp>.csv         white-box attacks: per dataset x target x eps x direction
  poisondet_<exp>.csv   AUROC of the DRCC excess estimate for poisoned vs clean windows
  ablation_<exp>.csv    DRCC components on real data
  timeline_H.csv        per stream x period: type-I and power of the main rules
  teps.csv, topt.csv    theory checks on synthetic data
"""
import glob, os, sys
import numpy as np, pandas as pd

OUT = "results_drift/tables"; os.makedirs(OUT, exist_ok=True)
MAIN = ["standard", "ref_only", "naive_trim", "rcad_u", "qtc", "weighted", "aci", "nexcp", "robust_kl",
        "mad_filter", "drcc_affine", "drcc", "drcc_safe", "oracle"]
ABL = ["drcc", "drcc_affine", "drcc_single", "drcc_auto", "drcc_unioncorr", "drcc_B10", "drcc_B40", "drcc_safe"]
_gb = pd.DataFrame.groupby
pd.DataFrame.groupby = lambda self, *a, **k: _gb(self, *a, **{"observed": True, **k})
_pt = pd.DataFrame.pivot_table
pd.DataFrame.pivot_table = lambda self, *a, **k: _pt(self, *a, **{"observed": True, **k})
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500); pd.set_option("display.max_columns", 40)


DROPC = {"map_a", "map_b", "theta", "n_tn", "n_ta", "spread", "stretch", "family", "gap"}


def _read(f, drop):
    d = pd.read_csv(f, low_memory=False, usecols=(lambda c: c not in DROPC) if drop else None)
    for c in d.columns:
        if d[c].dtype == object:
            d[c] = d[c].astype("category")
        elif c in ("typeI", "power", "E_hat", "flagged"):
            d[c] = d[c].astype(np.float32)
    return d


def load(prefix):
    fs = sorted(glob.glob(f"results_drift/{prefix}_*.csv"))
    if not fs:
        return None
    d = pd.concat([_read(f, prefix in ("G", "H")) for f in fs], ignore_index=True)
    for c in d.columns:
        if d[c].dtype == object:
            d[c] = d[c].astype("category")
    return d


def boot_ci(x, B=1000, seed=0):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if x.size < 2:
        return (np.nan, np.nan)
    r = np.random.default_rng(seed); bs = r.choice(x, (B, x.size)).mean(1)
    return tuple(np.quantile(bs, [0.025, 0.975]))


def cell_table(d, cell, by, rules, alpha_col="alpha"):
    """Average over seeds within a cell (setting), then summarise cells by `by`."""
    d = d[d.rule.isin(rules)]
    g = d.groupby(cell + ["rule"])[["typeI", "power"]].mean().reset_index()
    orc = d[d.rule == "oracle"].groupby(cell)["power"].mean().rename("p_or")
    g = g.join(orc, on=cell); g["dpow"] = g.power - g.p_or
    a = g[alpha_col] if alpha_col in g else 0.10
    g["viol"] = (g.typeI > a * 1.2).astype(float)
    return g.groupby(by + ["rule"]).agg(maxI=("typeI", "max"), meanI=("typeI", "mean"), viol=("viol", "mean"),
                                        dpow=("dpow", "mean"), dpow_p10=("dpow", lambda x: np.nanpercentile(x, 10)),
                                        n_cells=("typeI", "size")).reset_index()


def seed_ci(d, cell, by, rules, seed_col="seed"):
    """95% CI over seeds of the per-seed mean type-I and dpow."""
    d = d[d.rule.isin(rules)]
    g = d.groupby(by + [seed_col, "rule"])[["typeI"]].mean().reset_index()
    pw = d.pivot_table(index=cell + [seed_col], columns="rule", values="power", aggfunc="mean")
    rows = []
    for (key, s) in g.groupby(by + ["rule"]):
        lo, hi = boot_ci(s.typeI)
        rows.append(dict(zip(by + ["rule"], key if isinstance(key, tuple) else (key,)), I_lo=lo, I_hi=hi))
    return pd.DataFrame(rows)


def real(exp):
    d = load(exp)
    if d is None:
        print("no", exp); return
    d["q"] = d["q"].fillna(-1); d["alpha"] = d.get("alpha", 0.10)
    if "target" not in d:
        d["target"] = np.nan
    std = d[d.attack.isin(["none", "top", "bottom", "band", "mimic"])]
    cell = ["dataset", "detector", "window", "protocol", "mode", "attack", "q", "eps", "alpha"]
    # main table
    m = std[std.alpha == 0.10]
    t = cell_table(m, cell, ["dataset"], MAIN)
    ci = seed_ci(m, cell, ["dataset"], MAIN)
    t = t.merge(ci, on=["dataset", "rule"], how="left")
    t.to_csv(f"{OUT}/main_{exp}.csv", index=False)
    print(f"\n######## {exp}: main (alpha 0.10)  maxI / viol / meanI [95% CI] / dpow")
    for ds, s in t.groupby("dataset"):
        print(f" [{ds}] " + "  ".join(f"{r.rule}:{r.maxI:.2f}/{r.viol:.2f}/{r.meanI:.3f}[{r.I_lo:.3f},{r.I_hi:.3f}]/{r.dpow:+.3f}"
                                    for r in s.set_index("rule").reindex([x for x in MAIN if x in set(s.rule)]).reset_index().itertuples()))
    # by protocol / mode
    t2 = cell_table(m, cell, ["dataset", "protocol", "mode"], ["standard", "drcc", "drcc_safe", "drcc_affine", "oracle"])
    t2.to_csv(f"{OUT}/main_bymode_{exp}.csv", index=False)
    # alpha
    ta = cell_table(std, cell, ["alpha", "dataset"], MAIN)
    ta.to_csv(f"{OUT}/alpha_{exp}.csv", index=False)
    print(f"\n-- {exp}: viol (type-I > 1.2 alpha) by alpha")
    print(ta.pivot_table(index=["dataset", "alpha"], columns="rule", values="viol").reindex(
        columns=[x for x in MAIN if x in set(ta.rule)]).round(2).to_string())
    # detector
    td = cell_table(m, cell, ["dataset", "detector"], ["standard", "drcc", "drcc_safe", "drcc_affine", "aci", "mad_filter", "oracle"])
    td.to_csv(f"{OUT}/detector_{exp}.csv", index=False)
    # ablation
    if "drcc_single" in set(d.rule):
        tb = cell_table(m, cell, ["dataset"], ABL + ["oracle"])
        tb.to_csv(f"{OUT}/ablation_{exp}.csv", index=False)
        print(f"\n-- {exp}: ablation maxI/viol/dpow")
        print(tb.pivot_table(index="dataset", columns="rule", values="viol").round(3).to_string())
    # poison detection by DRCC excess estimate
    dr = m[(m.rule == "drcc") & (m["mode"] == "score")].copy()
    if "E_hat" in dr:
        from sklearn.metrics import roc_auc_score
        rows = []
        for (ds, atk), s in dr[dr.attack != "none"].groupby(["dataset", "attack"]):
            clean = dr[(dr.dataset == ds) & (dr.eps == 0)]
            for eps, p in s.groupby("eps"):
                y = np.r_[np.zeros(len(clean)), np.ones(len(p))]; sc = np.r_[clean.E_hat, p.E_hat]
                if len(clean) and len(p):
                    rows.append(dict(dataset=ds, attack=atk, eps=eps, auroc=roc_auc_score(y, sc),
                                     flag_clean=float((clean.E_hat > 0).mean()), flag_poison=float((p.E_hat > 0).mean())))
        pdt = pd.DataFrame(rows); pdt.to_csv(f"{OUT}/poisondet_{exp}.csv", index=False)
        print(f"\n-- {exp}: poisoning detection by DRCC excess (AUROC; flag rate clean / poisoned)")
        print(pdt.groupby(["dataset", "eps"])[["auroc", "flag_clean", "flag_poison"]].mean().round(3).to_string())
    # white-box attacks
    o = d[d.attack.isin(["opt_lo", "opt_hi"])]
    if len(o):
        oc = ["dataset", "detector", "window", "protocol", "attack", "eps", "target"]
        tgt = o[o.rule.astype(str) == o.target.astype(str)].groupby(oc)[["typeI", "power"]].mean()
        orc = o[o.rule == "oracle"].groupby(oc)[["typeI", "power"]].mean().add_suffix("_or")
        j = tgt.join(orc).reset_index(); j["dI"] = j.typeI - j.typeI_or; j["dP"] = j.power - j.power_or
        to = j.groupby(["dataset", "attack", "eps", "target"]).agg(I=("typeI", "mean"), maxI=("typeI", "max"),
                                                                dI=("dI", "mean"), dP=("dP", "mean")).reset_index()
        to.to_csv(f"{OUT}/opt_{exp}.csv", index=False)
        print(f"\n-- {exp}: white-box attacks: opt_lo -> mean type-I; opt_hi -> mean power gap vs oracle")
        lo = to[to.attack == "opt_lo"].pivot_table(index=["dataset", "eps"], columns="target", values="I").round(3)
        hi = to[to.attack == "opt_hi"].pivot_table(index=["dataset", "eps"], columns="target", values="dP").round(3)
        print("type-I under opt_lo\n" + lo.to_string()); print("power gap under opt_hi\n" + hi.to_string())
    if exp == "H":
        tl = m[(m.attack.isin(["none", "band"])) & (m.eps.isin([0.0, 0.10])) & (m["mode"] == "score") &
               m.rule.isin(["standard", "drcc", "drcc_safe", "oracle"])]
        tl.groupby(["dataset", "protocol", "window", "attack", "eps", "rule"])[["typeI", "power"]].mean().reset_index() \
          .to_csv(f"{OUT}/timeline_H.csv", index=False)


def synth():
    for p in ("F", "Isyn"):
        d = load(p)
        if d is None:
            continue
        for c in ("q", "spread", "stretch"):
            if c in d:
                d[c] = d[c].astype(float).fillna(-1)
        cell = [c for c in ("base", "drift", "sev", "alpha", "attack", "q", "spread", "stretch", "eps") if c in d]
        t = cell_table(d, cell, [], MAIN); t.to_csv(f"{OUT}/main_{p}.csv", index=False)
        print(f"\n######## {p}"); print(t.set_index("rule").reindex([x for x in MAIN if x in set(t.rule)]).round(3).to_string())
    d = load("Teps")
    if d is not None:
        d["q"] = d["q"].fillna(-1)
        t = d.groupby(["base", "drift", "attack", "q", "eps", "rule"])[["typeI", "power"]].mean().reset_index()
        t.to_csv(f"{OUT}/teps.csv", index=False)
        s = t[(t.base == "gauss") & (t.drift == "none") & t.rule.isin(["standard", "drcc", "drcc_safe", "drcc_affine", "oracle"])]
        print("\n-- Teps (gauss, no drift): power under top attack vs eps")
        print(s[s.attack.isin(["top", "none"])].pivot_table(index="eps", columns="rule", values="power").round(3).to_string())
        print("-- Teps: type-I under bottom attack vs eps")
        print(s[s.attack.isin(["bottom", "none"])].pivot_table(index="eps", columns="rule", values="typeI").round(3).to_string())
    d = load("Topt")
    if d is not None:
        d["dI"] = d.typeI - d.typeI_or; d["dP"] = d.power - d.power_or; d["dth"] = d.theta - d.theta_oracle
        t = d.groupby(["drift", "eps", "direction", "rule"])[["typeI", "dI", "dP", "dth"]].mean().reset_index()
        t.to_csv(f"{OUT}/topt.csv", index=False)
        print("\n-- Topt: type-I under opt_lo (mean over bases/drifts) by eps")
        print(t[t.direction == "lo"].pivot_table(index="eps", columns="rule", values="typeI").round(3).to_string())
        print("-- Topt: power gap under opt_hi by eps")
        print(t[t.direction == "hi"].pivot_table(index="eps", columns="rule", values="dP").round(3).to_string())


if __name__ == "__main__":
    which = sys.argv[1:] or ["synth", "G", "H"]
    for w in which:
        synth() if w == "synth" else real(w)

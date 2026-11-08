"""Real-drift streams.  All downloads are cached in ./cache next to this file.

Each loader returns a dict with
  X       : feature matrix (tabular features or sentence embeddings)
  y       : 1 = anomaly, 0 = normal
  period  : integer time period, 0 = oldest
  name    : dataset name
"""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
os.makedirs(CACHE, exist_ok=True)
for var in ("HF_HOME", "HF_DATASETS_CACHE", "TRANSFORMERS_CACHE", "TORCH_HOME", "XDG_CACHE_HOME"):
    os.environ[var] = CACHE
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import numpy as np
from sklearn.datasets import fetch_openml

SK_DATA = os.path.join(CACHE, "sk_data")

# ------------------------------------------------------------------ gas drift
# UCI gas sensor array drift (Vergara et al. 2012): 10 batches over 36 months,
# read from the original batch files so the batch of every row is known exactly.
# Gases: 1 ethanol, 2 ethylene, 3 ammonia, 4 acetaldehyde, 5 acetone, 6 toluene.
GAS_URLS = ["https://archive.ics.uci.edu/static/public/224/gas+sensor+array+drift+dataset.zip",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00224/Dataset.zip"]
GAS_NORMAL = (2,)                  # normal: ethylene; every other gas is an anomaly
# batches -> periods: b1-3 old (months 1-12), b4-5, b6, b7, b8-9, b10
GAS_PERIOD = {1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 2, 7: 3, 8: 4, 9: 4, 10: 5}


GAS_CONC_URLS = ["https://archive.ics.uci.edu/static/public/270/gas+sensor+array+drift+dataset+at+different+concentrations.zip"]


def _gas_raw(with_conc=False):
    """Batch files of the gas drift data.  The concentration-annotated release
    (UCI 270, lines 'class;conc idx:val ...') is used when with_conc is set."""
    import zipfile, urllib.request, io
    path = os.path.join(CACHE, "gas_drift_conc.zip" if with_conc else "gas_drift.zip")
    if not os.path.exists(path):
        last = None
        for u in (GAS_CONC_URLS if with_conc else GAS_URLS):
            try:
                with urllib.request.urlopen(u, timeout=120) as r:
                    data = r.read()
                open(path, "wb").write(data); break
            except Exception as ex:
                last = ex
        if not os.path.exists(path):
            raise RuntimeError(f"gas download failed: {last!r}")

    def walk(zf):
        for nm in zf.namelist():
            low = nm.lower()
            if low.endswith(".zip"):
                yield from walk(zipfile.ZipFile(io.BytesIO(zf.read(nm))))
            elif low.endswith(".dat") and "batch" in low:
                yield nm, zf.read(nm).decode("utf-8", "ignore")
    rows, cls, bat, conc = [], [], [], []
    for nm, txt in walk(zipfile.ZipFile(path)):
        b = int("".join(ch for ch in os.path.basename(nm).split(".")[0] if ch.isdigit()))
        for line in txt.splitlines():
            parts = line.split()
            if not parts:
                continue
            c = int(parts[0].split(";")[0])
            cc = float(parts[0].split(";")[1]) if ";" in parts[0] else np.nan
            v = np.zeros(128)
            for kv in parts[1:]:
                k, x = kv.split(":"); v[int(k) - 1] = float(x)
            rows.append(v); cls.append(c); bat.append(b); conc.append(cc)
    if with_conc:
        return np.array(rows), np.array(cls), np.array(bat), np.array(conc)
    return np.array(rows), np.array(cls), np.array(bat)


def load_gas(normal=GAS_NORMAL):
    X, c, b = _gas_raw()
    X = np.sign(X) * np.log1p(np.abs(X))
    y = (~np.isin(c, normal)).astype(int)
    period = np.array([GAS_PERIOD[int(v)] for v in b])
    counts = {int(k): np.bincount(c[b == k], minlength=7)[1:].tolist() for k in sorted(set(b.tolist()))}
    return dict(X=X, y=y, period=period, name="gas",
                info=dict(n=len(c), normal_classes=list(normal), class_counts_by_batch=counts))


# ------------------------------------------------------------------ credit card
CC_URL = "https://www.openml.org/data/v1/download/1673544/creditcard.arff"


def _creditcard_frame():
    """Read the OpenML creditcard ARFF directly (sklearn's md5 check fails on the
    current server copy), keeping the row order, which is time order."""
    import pandas as pd, urllib.request
    path = os.path.join(CACHE, "creditcard.arff")
    if not os.path.exists(path) or os.path.getsize(path) < 100_000_000:
        urllib.request.urlretrieve(CC_URL, path)
    cols, skip = [], 0
    with open(path, errors="ignore") as f:
        for i, line in enumerate(f):
            t = line.strip()
            if t.lower().startswith("@attribute"):
                cols.append(t.split()[1].strip("'\""))
            if t.lower().startswith("@data"):
                skip = i + 1; break
    df = pd.read_csv(path, skiprows=skip, header=None, names=cols, quotechar="'",
                     comment="%", skipinitialspace=True)
    return df


def load_creditcard(n_periods=8):
    df = _creditcard_frame()
    tcol = [c for c in df.columns if c.lower() == "time"]
    ycol = [c for c in df.columns if c.lower() == "class"][0]
    y = np.array([str(v).strip().strip("'") for v in df[ycol]])
    y = (y == "1").astype(int)
    feats = [c for c in df.columns if c not in tcol + [ycol]]
    X = df[feats].to_numpy(dtype=float)
    X[:, -1] = np.log1p(np.abs(X[:, -1]))                      # Amount
    if tcol:
        t = df[tcol[0]].to_numpy(dtype=float)
    else:
        t = np.arange(len(df), dtype=float)                    # rows are time-ordered
    edges = np.quantile(t, np.linspace(0, 1, n_periods + 1))
    period = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, n_periods - 1)
    return dict(X=X, y=y, period=period, name="creditcard",
                info=dict(has_time_column=bool(tcol), n=len(df), frauds=int(y.sum())))


# ------------------------------------------------------------------ text: embeddings
def embed(texts, tag, batch=64, max_length=256):
    path = os.path.join(CACHE, f"emb_{tag}.npy")
    if os.path.exists(path):
        E = np.load(path)
        if E.shape[0] == len(texts):
            return E
    import torch
    from transformers import AutoTokenizer, AutoModel
    torch.set_num_threads(max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "2"))))
    name = "sentence-transformers/all-MiniLM-L6-v2"
    tok = AutoTokenizer.from_pretrained(name, cache_dir=CACHE)
    mdl = AutoModel.from_pretrained(name, cache_dir=CACHE).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok(list(texts[i:i + batch]), padding=True, truncation=True,
                      max_length=max_length, return_tensors="pt")
            h = mdl(**enc).last_hidden_state
            msk = enc["attention_mask"].unsqueeze(-1).float()
            v = (h * msk).sum(1) / msk.sum(1).clamp(min=1e-9)
            out.append(torch.nn.functional.normalize(v, dim=1).numpy())
            if (i // batch) % 50 == 0:
                print(f"  embed {tag}: {i}/{len(texts)}", flush=True)
    E = np.vstack(out).astype(np.float32)
    np.save(path, E)
    return E


def _hf_file(repo, filename):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset", cache_dir=CACHE)


# normal stream: political headlines, whose topics drift strongly from year to year;
# anomalies: topically adjacent desks (business, crime, media)
NEWS_NORMAL = ["POLITICS"]
NEWS_ANOMALY = ["BUSINESS", "CRIME", "MEDIA"]


def load_news(per_year_normal=2500, per_year_anom=300, seed=0):
    path = _hf_file("heegyu/news-category-dataset", "data.json")
    rows = []
    with open(path) as f:
        txt = f.read().strip()
    recs = json.loads(txt) if txt.startswith("[") else [json.loads(l) for l in txt.splitlines() if l.strip()]
    for r in recs:
        cat = r.get("category", "")
        if cat not in NEWS_NORMAL + NEWS_ANOMALY:
            continue
        yr = int(str(r.get("date", "0000"))[:4])
        text = (r.get("headline", "") + ". " + r.get("short_description", "")).strip()
        rows.append((yr, int(cat in NEWS_ANOMALY), text))
    rng = np.random.default_rng(seed)
    years = sorted({y for y, _, _ in rows})
    keep = []
    for yr in years:
        for lab, cap in ((0, per_year_normal), (1, per_year_anom)):
            idx = [i for i, (y, l, _) in enumerate(rows) if y == yr and l == lab]
            rng.shuffle(idx); keep += idx[:cap]
    keep.sort()
    yrs = np.array([rows[i][0] for i in keep]); y = np.array([rows[i][1] for i in keep])
    texts = [rows[i][2] for i in keep]
    X = embed(texts, f"newspol_{per_year_normal}_{per_year_anom}_{seed}")
    # periods: <= 2014 old; then 2015, 2016, 2017; years >= 2018 pooled
    yr_c = np.clip(yrs, 2014, 2018)            # political desk is sparse before 2014
    uniq = sorted(set(yr_c)); period = np.array([uniq.index(v) for v in yr_c])
    return dict(X=X, y=y, period=period, name="news",
                info=dict(years=uniq, n=len(texts), anomalies=int(y.sum())))


def load_jailbreak():
    import pandas as pd
    reg = pd.read_parquet(_hf_file("TrustAIRLab/in-the-wild-jailbreak-prompts",
                                   "regular_2023_12_25/train-00000-of-00001.parquet"))
    jb = pd.read_parquet(_hf_file("TrustAIRLab/in-the-wild-jailbreak-prompts",
                                  "jailbreak_2023_12_25/train-00000-of-00001.parquet"))
    reg = reg.assign(y=0); jb = jb.assign(y=1)
    df = pd.concat([reg, jb], ignore_index=True)
    dcol = next(c for c in ("date", "created_at") if c in df.columns)
    t = pd.to_datetime(df[dcol], errors="coerce", utc=True)
    df = df.assign(t=t).dropna(subset=["t", "prompt"]).sort_values("t").reset_index(drop=True)
    texts = df["prompt"].astype(str).str.slice(0, 2000).tolist()
    X = embed(texts, f"jailbreak_{len(texts)}")
    # periods: quarters of the observed time range; earliest quarter = old
    q = df["t"].dt.year.astype(int) * 4 + (df["t"].dt.month.astype(int) - 1) // 3
    uniq = sorted(set(q)); period = np.array([uniq.index(v) for v in q])
    period = np.maximum(period - 1, 0)          # first quarter is tiny: merge it into the old period
    return dict(X=X, y=df["y"].to_numpy(), period=period, name="jailbreak",
                info=dict(columns=list(reg.columns), n=len(df), jailbreaks=int(df["y"].sum()),
                          per_period=np.bincount(period).tolist(),
                          per_period_jb=np.bincount(period, weights=df["y"].to_numpy()).astype(int).tolist()))


# arXiv stream: normal = papers whose primary category is cs.LG; anomalies = off-topic
# submissions from other fields.  Time is read from the new-style identifier (YYMM.NNNNN).
ARXIV_NORMAL = ("cs.LG",)
ARXIV_ANOMALY_PREFIX = ("q-bio.", "econ.", "cs.CR", "cs.DB", "math.AP", "astro-ph", "cond-mat.", "stat.ME")


def load_arxiv(per_year_normal=2500, per_year_anom=300, seed=0, years=(2012, 2021)):
    import gzip, re
    path = _hf_file("gfissore/arxiv-abstracts-2021", "arxiv-abstracts.jsonl.gz")
    pat = re.compile(r"^(\d{2})(\d{2})\.\d{4,5}")
    rows = []; sub = np.random.default_rng(seed + 7)
    with gzip.open(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            m = pat.match(str(r.get("id", "")))
            if not m:
                continue
            yr = 2000 + int(m.group(1))
            if yr < years[0] or yr > years[1]:
                continue
            cats = r.get("categories", "")
            if isinstance(cats, list):
                cats = " ".join(cats)
            prim = cats.split()[0] if cats.split() else ""
            if prim in ARXIV_NORMAL:
                lab = 0
            elif prim.startswith(ARXIV_ANOMALY_PREFIX):
                lab = 1
                if sub.random() > 0.1:          # thin the large fields before storing text
                    continue
            else:
                continue
            text = (str(r.get("title", "")).strip() + ". " + " ".join(str(r.get("abstract", "")).split()))
            rows.append((yr, lab, text))
    rng = np.random.default_rng(seed)
    yrs_all = np.array([a for a, _, _ in rows]); lab_all = np.array([b for _, b, _ in rows])
    keep = []
    for yr in sorted(set(yrs_all.tolist())):
        for lab, cap in ((0, per_year_normal), (1, per_year_anom)):
            idx = np.flatnonzero((yrs_all == yr) & (lab_all == lab)); rng.shuffle(idx)
            keep += idx[:cap].tolist()
    keep.sort()
    yrs = yrs_all[keep]; y = lab_all[keep]; texts = [rows[i][2] for i in keep]
    X = embed(texts, f"arxivlg_{per_year_normal}_{per_year_anom}_{seed}_{years[0]}")
    # periods: <= 2016 old (cs.LG is small before then); then one period per year
    yr_c = np.clip(yrs, 2016, years[1])
    uniq = sorted(set(yr_c.tolist())); period = np.array([uniq.index(v) for v in yr_c])
    return dict(X=X, y=y, period=period, name="arxiv",
                info=dict(years=uniq, n=len(texts), anomalies=int(y.sum()),
                          per_period=np.bincount(period).tolist(),
                          per_period_anom=np.bincount(period, weights=y).astype(int).tolist()))


# Image stream: CIFAR-10 seen through a progressively degrading camera (blur, dimming and
# sensor noise grow with the period, as in the CIFAR-10-C severity levels).  Normal = the
# four vehicle classes, anomalies = animals.  Each period uses disjoint images.
CIFAR_NORMAL = (0, 1, 8, 9)


def _degrade(img, p, rng):
    """img: float array (N,32,32,3) in [0,1]; severity p = 0..5."""
    from scipy.ndimage import gaussian_filter
    if p == 0:
        return img
    out = gaussian_filter(img, sigma=(0, 0.12 * p, 0.12 * p, 0))
    out = out * (1 - 0.05 * p) + 0.01 * p
    out = out + rng.normal(0, 0.006 * p, out.shape)
    return np.clip(out, 0, 1)


def embed_images(X, tag, batch=128, res=128):
    path = os.path.join(CACHE, f"emb_{tag}.npy")
    if os.path.exists(path):
        E = np.load(path)
        if E.shape[0] == len(X):
            return E
    import torch
    from transformers import ResNetModel
    torch.set_num_threads(max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "2"))))
    mdl = ResNetModel.from_pretrained("microsoft/resnet-18", cache_dir=CACHE).eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); sd = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            x = torch.tensor(X[i:i + batch], dtype=torch.float32).permute(0, 3, 1, 2)
            x = torch.nn.functional.interpolate(x, size=(res, res), mode="bilinear", align_corners=False)
            v = mdl(pixel_values=(x - mean) / sd).pooler_output.flatten(1)
            out.append(torch.nn.functional.normalize(v, dim=1).numpy())
            if (i // batch) % 20 == 0:
                print(f"  embed {tag}: {i}/{len(X)}", flush=True)
    E = np.vstack(out).astype(np.float32); np.save(path, E)
    return E


def load_cifar(per_period_normal=2500, per_period_anom=300, n_periods=6, seed=0):
    import pandas as pd, io
    from PIL import Image
    df = pd.concat([pd.read_parquet(_hf_file("uoft-cs/cifar10", f"plain_text/{s}-00000-of-00001.parquet"))
                    for s in ("train", "test")], ignore_index=True)
    icol = next(c for c in df.columns if c in ("img", "image")); lcol = next(c for c in df.columns if "label" in c)
    lab = df[lcol].to_numpy(); rng = np.random.default_rng(seed)
    nrm = rng.permutation(np.flatnonzero(np.isin(lab, CIFAR_NORMAL)))
    anm = rng.permutation(np.flatnonzero(~np.isin(lab, CIFAR_NORMAL)))
    idx, per, y = [], [], []
    for p in range(n_periods):
        a = nrm[p * per_period_normal:(p + 1) * per_period_normal]; b = anm[p * per_period_anom:(p + 1) * per_period_anom]
        idx += a.tolist() + b.tolist(); per += [p] * (a.size + b.size); y += [0] * a.size + [1] * b.size
    def dec(v):
        return np.asarray(Image.open(io.BytesIO(v["bytes"] if isinstance(v, dict) else v)).convert("RGB"), dtype=np.float32) / 255.
    imgs = np.stack([dec(df[icol].iloc[i]) for i in idx]); per = np.array(per); y = np.array(y)
    drng = np.random.default_rng(seed + 1)
    for p in range(n_periods):
        sel = per == p; imgs[sel] = _degrade(imgs[sel], p, drng)
    X = embed_images(imgs, f"cifardeg2_{per_period_normal}_{per_period_anom}_{n_periods}_{seed}")
    return dict(X=X, y=y, period=per, name="cifar",
                info=dict(n=len(y), anomalies=int(y.sum()), per_period=np.bincount(per).tolist()))


# Network intrusion stream: CIC-IDS2017 flows in capture order (Monday -> Friday afternoon).
# Normal = BENIGN flows, anomalies = attack flows; one period per capture file.
CICIDS_FILES = ["Monday-WorkingHours.pcap_ISCX.csv", "Tuesday-WorkingHours.pcap_ISCX.csv",
                "Wednesday-workingHours.pcap_ISCX.csv", "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
                "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv", "Friday-WorkingHours-Morning.pcap_ISCX.csv",
                "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv", "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"]


def load_cicids(per_period_normal=4000, per_period_anom=600, seed=0):
    import pandas as pd
    rng = np.random.default_rng(seed)
    Xs, ys, ps, cols = [], [], [], None
    for p, fn in enumerate(CICIDS_FILES):
        df = pd.read_csv(_hf_file("c01dsnap/CIC-IDS2017", fn), low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        lab = df["Label"].astype(str).str.strip()
        num = df.drop(columns=["Label"]).apply(pd.to_numeric, errors="coerce")
        if cols is None:
            cols = list(num.columns)
        num = num.reindex(columns=cols)
        V = num.to_numpy(dtype=float); V[~np.isfinite(V)] = 0.0
        b = np.flatnonzero(lab.to_numpy() == "BENIGN"); a = np.flatnonzero(lab.to_numpy() != "BENIGN")
        b = rng.choice(b, min(per_period_normal, b.size), replace=False)
        a = rng.choice(a, min(per_period_anom, a.size), replace=False) if a.size else a
        Xs.append(V[np.r_[b, a]]); ys.append(np.r_[np.zeros(b.size), np.ones(a.size)]); ps.append(np.full(b.size + a.size, p))
        print(" cicids", fn, b.size, a.size, flush=True)
    X = np.vstack(Xs); X = np.sign(X) * np.log1p(np.abs(X))
    keep = X.std(0) > 0
    y = np.concatenate(ys).astype(int); per = np.concatenate(ps)
    return dict(X=X[:, keep], y=y, period=per, name="cicids",
                info=dict(n=len(y), anomalies=int(y.sum()), per_period=np.bincount(per).tolist(),
                          per_period_anom=np.bincount(per, weights=y).astype(int).tolist(), d=int(keep.sum())))


LOADERS = dict(gas=load_gas, creditcard=load_creditcard, news=load_news, jailbreak=load_jailbreak,
               arxiv=load_arxiv, cifar=load_cifar, cicids=load_cicids)

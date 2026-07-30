"""Extract embeddings for Track 2 (embedding-space generalization).
Vision : CLIP ViT-B/32 (transformers) on CIFAR-10, one-vs-rest.
         raw-pixel counterpart (flattened, standardized) for paired comparison.
Text   : sentence-transformers all-MiniLM-L6-v2 on 20 Newsgroups, one-vs-rest.
         TF-IDF + TruncatedSVD(256) counterpart.
Writes ADBench-format .npz {X, y} so the existing loader treats each as a dataset.
y=1 means anomaly (the held-out minority class), y=0 normal (the 'rest', subsampled).
Run on a GPU node AFTER assets are cached on the login node.
"""
import os, sys, numpy as np
OUT = os.environ.get("EMB_OUT", "track2/assets/emb")
os.makedirs(OUT, exist_ok=True)
SEED = 0
rng = np.random.RandomState(SEED)

# anomaly frac target in each one-vs-rest problem
ANOM_FRAC = 0.10
N_NORMAL = 2000     # subsample normals to keep problems mid-sized

def save(name, X, y):
    X = np.asarray(X, np.float32); y = np.asarray(y, np.int64)
    np.savez(os.path.join(OUT, name + ".npz"), X=X, y=y)
    print("saved %-28s X=%s anom=%d/%d" % (name, X.shape, int(y.sum()), len(y)), flush=True)

def make_ovr(feat, labels, cls, n_normal=N_NORMAL, anom_frac=ANOM_FRAC):
    labels = np.asarray(labels)
    norm_idx = np.where(labels != cls)[0]
    anom_idx = np.where(labels == cls)[0]
    nn = min(n_normal, len(norm_idx))
    n_anom = int(round(nn * anom_frac / (1 - anom_frac)))
    n_anom = min(n_anom, len(anom_idx))
    ni = rng.choice(norm_idx, nn, replace=False)
    ai = rng.choice(anom_idx, n_anom, replace=False)
    idx = np.concatenate([ni, ai])
    y = np.concatenate([np.zeros(len(ni)), np.ones(len(ai))]).astype(int)
    p = rng.permutation(len(idx))
    return feat[idx[p]], y[p]

def do_vision(classes=(3,5,1,9)):  # cat, dog, automobile, truck (harder in CLIP space)
    import torch, torchvision
    from transformers import CLIPModel, CLIPProcessor
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("vision device:", dev, flush=True)
    root = os.environ.get("TORCHVISION_ROOT", "track2/assets/cifar")
    ds = torchvision.datasets.CIFAR10(root=root, train=True, download=False)
    imgs = ds.data           # (50000,32,32,3) uint8
    labels = np.array(ds.targets)
    # subsample the full pool once for tractability
    keep = rng.choice(len(imgs), 12000, replace=False)
    imgs, labels = imgs[keep], labels[keep]

    # ---- CLIP embeddings ----
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(dev).eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    feats = []
    import torch
    with torch.no_grad():
        B = 256
        for i in range(0, len(imgs), B):
            batch = [imgs[j] for j in range(i, min(i+B, len(imgs)))]
            inp = proc(images=batch, return_tensors="pt").to(dev)
            f = model.get_image_features(**inp)
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu().numpy())
            if i % 2560 == 0: print("  clip %d/%d" % (i, len(imgs)), flush=True)
    clip_feat = np.concatenate(feats).astype(np.float32)
    # ---- raw pixels (flattened + standardized) ----
    raw = imgs.reshape(len(imgs), -1).astype(np.float32)
    raw = (raw - raw.mean(0)) / (raw.std(0) + 1e-6)
    for cls in classes:
        Xe, ye = make_ovr(clip_feat, labels, cls)
        save("clipvit_cifar10_c%d" % cls, Xe, ye)
        Xr, yr = make_ovr(raw, labels, cls)     # same rng sequence order differs; acceptable, paired by class
        save("rawpix_cifar10_c%d" % cls, Xr, yr)

def do_text(classes=(7,)):  # will map to actual 20NG category indices below
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sentence_transformers import SentenceTransformer
    data = fetch_20newsgroups(subset="train", remove=("headers","footers","quotes"),
                              data_home=os.environ.get("SK_DATA","track2/assets/sk"))
    texts = data.data; labels = np.array(data.target)
    # sentence embeddings
    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    emb = st.encode(texts, batch_size=128, show_progress_bar=False, normalize_embeddings=True).astype(np.float32)
    # tfidf + svd
    tf = TfidfVectorizer(max_features=20000, min_df=3).fit_transform(texts)
    svd = TruncatedSVD(n_components=256, random_state=SEED).fit_transform(tf).astype(np.float32)
    # pick a few categories that are semantically distinctive as anomalies
    for cls in classes:
        Xe, ye = make_ovr(emb, labels, cls)
        save("minilm_news20_c%d" % cls, Xe, ye)
        Xs, ys = make_ovr(svd, labels, cls)
        save("tfidfsvd_news20_c%d" % cls, Xs, ys)

if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("all","vision"): do_vision()
    if what in ("all","text"): do_text()
    print("EXTRACT DONE", flush=True)

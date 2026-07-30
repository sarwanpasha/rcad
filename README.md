# RCAD Rebuttal Artifact (anonymous)

End-to-end, runnable code and results. This package is self-contained: the two tabular benchmarks
(mammography, satellite; standard public ODDS/OpenML anomaly-detection sets)
are bundled under `code/data/`, so no network access is required for the
tabular tracks.

## Quick start
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt        # numpy scipy pandas scikit-learn
bash run_all.sh                        # runs Tracks 0, 1, 2; ~40 min on a CPU node
```
To use your own interpreter: `PYTHON=/path/to/python bash run_all.sh`.

## What each track answers
- **Track 0 - ablations** (`code/rcad_v2.py`): reference-contamination sweep
  (e10) and reference-size sweep (e11). Raw CSVs: `results/ablation_results/`.
- **Track 1 - deployment/temporal** (`code/track1_deploy.py`, `code/track1_grid.py`):
  Stage 1a drift-baseline gate and Stage 1b five-arm streaming grid with an
  onset attacker (attack begins at window 3). Results: `results/track1/`.
- **Track 2 - embeddings** (`code/embed_extract.py`, `code/calib2a.py`, `code/grid2b.py`):
  CLIP ViT-B/32 on CIFAR-10 and all-MiniLM-L6-v2 on 20 Newsgroups, kNN and
  Mahalanobis embedding-space detectors, raw-pixel / tfidf-svd baselines.
  `embed_extract.py` (re)builds embeddings and needs torch+transformers;
  the pre-computed embedding grid is already in `results/assets/` so
  `grid2b.py` reproduces the reported table without a GPU.

## Reproducibility
- All randomness is seeded; seeds are set inside each driver.
- Detector: Isolation Forest; conformal level alpha = 0.10 (tabular tracks).
- See `MANIFEST.md` for the mapping from every number in the response to the
  exact CSV and column it came from.

## Layout
```
code/       harness + all experiment drivers + bundled data/
results/    per-trial raw CSVs and summaries for every track
run_all.sh  single-command driver
requirements.txt
MANIFEST.md
```

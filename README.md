# DRCC: drift-robust conformal calibration under calibration poisoning

Anomaly detectors are deployed with a score threshold that is recalibrated on a fresh
window of recent normal data. When the score distribution drifts and part of that window
may be poisoned, the standard split conformal threshold can be moved by an attacker.
DRCC explains the fresh window as a monotone transport of a stale, vetted reference sample
plus sparse excess mass, removes the excess and computes a weighted conformal rank on
what remains.

This repository contains the method, all baselines, the poisoning attacks and the scripts
for the synthetic and real-data experiments.

## Files

| file | content |
|---|---|
| `drcc.py` | DRCC: robust affine and piecewise-linear transport fits, binned excess scan with an exact tail test, keep-weights and weighted conformal threshold (`theta_drcc`, default `transport="union3"`), safe variant |
| `baselines.py` | all calibration rules (`all_thresholds`): standard split conformal, reference-only, pooled, trimming, weighted conformal, ACI, NexCP, KL-robust conformal, MAD filter, DRCC-affine, DRCC, DRCC-safe, oracle; `ABLATE=1` adds DRCC ablations |
| `attacks.py`, `common.py` | oblivious poisoning in score space and with real inputs, detectors (kNN, Mahalanobis, Deep SVDD and the classical detectors), per-rule metrics |
| `attacks_opt.py` | white-box attacks optimized against each rule (`opt_lo` lowers the threshold, `opt_hi` raises it) |
| `rcad.py` | tabular datasets, classical detectors (IF, LOF, OCSVM, ECOD, autoencoder) and simple calibration rules |
| `data_drift.py` | real drift streams: gas sensors (UCI), credit card (OpenML), news headlines, arXiv abstracts, degraded CIFAR-10 (ResNet-18 features), CIC-IDS2017 |
| `exp_synth.py` | synthetic experiments: `F` main grid, `I` dense adaptive attacks, `J` sample sizes, `K` knob ablation, `T` poison fraction around alpha, `O` white-box attacks |
| `exp_real.py` | `G` tabular benchmarks with injected drift, `H` real streams |
| `runtime.py` | wall-clock cost per rule and window size |
| `analyze.py`, `fig_data.py` | summary tables in `results_drift/tables/` |
| `figures/` | plotting scripts |
| `slurm/run_array.sh` | generic Slurm array job |

## Setup

```bash
pip install -r requirements.txt
```

Python 3.10 or newer. Datasets and pretrained encoders are downloaded on first use into
`cache/` (Hugging Face Hub, UCI, OpenML).

## Usage

```python
import numpy as np
from drcc import theta_drcc

ref = np.random.randn(200)                 # stale, vetted reference scores
win = 0.5 + 1.1 * np.random.randn(200)     # fresh (drifted, possibly poisoned) window
theta = theta_drcc(ref, win, alpha=0.10)   # flag a test point when its score > theta
```

## Reproducing the experiments

Each script takes `<experiment> <task id> <number of tasks>` and writes
`results_drift/<experiment>_<task>.csv`, so experiments can be split over a job array.

```bash
TRIALS=200 python exp_synth.py F 0 1
TRIALS=200 python exp_synth.py I 0 1
TRIALS=200 python exp_synth.py J 0 1
TRIALS=200 python exp_synth.py K 0 1
TRIALS=200 python exp_synth.py T 0 1
TRIALS=30  python exp_synth.py O 0 1
ALPHAS=0.10,0.05,0.01 OPT_ATTACK=1 ABLATE=1 python exp_real.py G 0 1
ALPHAS=0.10,0.05,0.01 OPT_ATTACK=1 ABLATE=1 python exp_real.py H 0 1
python runtime.py
python analyze.py synth G H
```

On a Slurm cluster, for example:

```bash
sbatch --array=0-19 slurm/run_array.sh exp_synth.py F 20
```

All randomness is seeded from the experiment configuration (CRC32 of the configuration
tuple), so the split into tasks does not change the results.

## Metrics

Per setting (dataset, detector, window, protocol, mode, attack, poison fraction, alpha),
results are averaged over seeds. `maxI` is the worst type-I error over settings, `viol` the
fraction of settings with type-I error above 1.2 alpha, and `dpow` the power minus the power
of the oracle threshold computed on the clean window entries.

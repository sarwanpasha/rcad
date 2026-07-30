# MANIFEST - number-to-file mapping

## Track 0 (ablations)
- Reference-contamination sweep (e10): `results/ablation_results/e10_refcontam_raw.csv`
  summarized in `results/ablation_results/e10_summary_caleps10.csv`.
- Reference-size sweep (e11): `results/ablation_results/e11_refsize_raw.csv`.

## Track 1 (deployment / temporal)
- Stage 1a drift-baseline gate (all delta pass, spread ~0.02 vs thr 0.03):
  `results/track1/stage1a_gate.csv`, decision map `results/track1/passmap.json`.
- Stage 1b five-arm grid, onset attacker (W_START=3): `results/track1/stage1b_grid_v2.csv`
  (per-trial), `results/track1/stage1b_summary.csv`, and phase split (pre/post
  onset) `results/track1/stage1b_phase_summary.csv`.
  v1 grid (attack all windows) kept for reference: `results/track1/stage1b_grid.csv`.

## Track 2 (embeddings)
- Difficulty calibration (clean power / coverage per config):
  `results/assets/calib2a_summary.csv`.
- Contamination grid (CLIP-CIFAR10, MiniLM-20news; kNN/Maha; Standard/Robust/Auto;
  cal_eps in {0,0.05,0.10,0.20}; per seed): `results/assets/grid2b_raw.csv`.

## Fixed configuration
- Detector: Isolation Forest. Conformal level alpha = 0.10 (tabular tracks).
- Seeds are set inside each driver in `code/`.

#!/usr/bin/env bash
# RCAD rebuttal artifact - end-to-end reproduction driver.
# Usage:  bash run_all.sh
# Runs the three rebuttal experiment tracks and writes CSV outputs under ./out/.
# Only requires: numpy scipy pandas scikit-learn  (see requirements.txt).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/code"
export PYTHONPATH="$HERE/code:${PYTHONPATH:-}"
OUT="$HERE/out"; mkdir -p "$OUT"
PY="${PYTHON:-python3}"
echo "[env] using interpreter: $($PY --version 2>&1)"

echo "===== TRACK 0: ablations (reference contamination e10 + reference size e11) ====="
$PY rcad_v2.py --help >/dev/null 2>&1 || true
$PY rcad_v2.py || true

echo "===== TRACK 1a: deployment drift gate ====="
$PY track1_deploy.py
echo "===== TRACK 1b: five-arm deployment grid ====="
$PY track1_grid.py

echo "===== TRACK 2a: embedding calibration ====="
$PY calib2a.py
echo "===== TRACK 2b: embedding generalization grid ====="
$PY grid2b.py

echo "[done] outputs written next to each script; see README.md for the mapping to reported numbers."

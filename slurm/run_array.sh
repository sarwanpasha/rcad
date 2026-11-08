#!/bin/bash
# Generic Slurm array job. Adjust resources (and add --partition/--account) for your cluster.
# Usage:  sbatch --array=0-19 slurm/run_array.sh exp_synth.py F 20
#         sbatch --array=0-87 slurm/run_array.sh exp_real.py H 88
#SBATCH --job-name=drcc
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%A_%a.out
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs results_drift
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
SCRIPT=$1; EXP=$2; NTASK=$3
python3 -u "$SCRIPT" "$EXP" "$SLURM_ARRAY_TASK_ID" "$NTASK"

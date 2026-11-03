#!/bin/bash
#SBATCH --job-name=anm-ten-retry
#SBATCH --partition=generalaccess
#SBATCH --array=0-3%4
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=01:00:00
#SBATCH --output=/home/coulson/oc/lina4225/_runtime/jax-Ka/anm-pilot-v1/masif-audit-v1/modes10-retry-%A_%a.log
set -euo pipefail
pilot=/home/coulson/oc/lina4225/_runtime/jax-Ka/anm-pilot-v1
runtime="$pilot/masif-audit-v1"
export PYTHONPATH=/home/coulson/oc/lina4225/jax-Ka/src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
IFS=, read -r -a protein_indices <<< "${3:?protein indices}"
protein=${protein_indices[$((SLURM_ARRAY_TASK_ID / 4))]}
shard=$((SLURM_ARRAY_TASK_ID % 4))
"$pilot/env/bin/python" "${1:?frozen code}/augmentation_modes10.py" --runtime "$runtime" --calibration "$runtime/results/surface-filter-final-751543" --exhaustive "$runtime/results/surface-filter-751526" --output "$runtime/results/modes10-${2:?original array}" --index "$protein" --shard-count 4 --shard-index "$shard"

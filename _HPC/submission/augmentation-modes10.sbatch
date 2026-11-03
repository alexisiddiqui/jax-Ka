#!/bin/bash
#SBATCH --job-name=anm-ten-modes
#SBATCH --partition=generalaccess
#SBATCH --array=0-11%12
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=02:00:00
#SBATCH --output=/home/coulson/oc/lina4225/_runtime/jax-Ka/anm-pilot-v1/masif-audit-v1/modes10-%A_%a.log
set -euo pipefail
pilot=/home/coulson/oc/lina4225/_runtime/jax-Ka/anm-pilot-v1
runtime="$pilot/masif-audit-v1"
export PYTHONPATH=/home/coulson/oc/lina4225/jax-Ka/src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
code=${1:?frozen code directory}
"$pilot/env/bin/python" "$code/augmentation_modes10.py" --runtime "$runtime" --calibration "$runtime/results/surface-filter-final-751543" --exhaustive "$runtime/results/surface-filter-751526" --output "$runtime/results/modes10-${2:-$SLURM_ARRAY_JOB_ID}" --index "$SLURM_ARRAY_TASK_ID"

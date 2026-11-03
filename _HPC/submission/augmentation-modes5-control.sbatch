#!/bin/bash
#SBATCH --job-name=anm-five-mode-control
#SBATCH --partition=generalaccess
#SBATCH --array=0-11%12
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:40:00
#SBATCH --output=/home/coulson/oc/lina4225/_runtime/jax-Ka/anm-pilot-v1/masif-audit-v1/modes5-control-%A_%a.log
set -euo pipefail
pilot=/home/coulson/oc/lina4225/_runtime/jax-Ka/anm-pilot-v1
runtime="$pilot/masif-audit-v1"
export PYTHONPATH=/home/coulson/oc/lina4225/jax-Ka/src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
code=${1:?frozen control source}
"$pilot/env/bin/python" "$code/augmentation_modes10.py" --runtime "$runtime" --calibration "$runtime/results/surface-filter-final-751543" --exhaustive "$runtime/results/surface-filter-751526" --output "$runtime/results/modes5-repeat-$SLURM_ARRAY_JOB_ID" --index "$SLURM_ARRAY_TASK_ID" --control-five

#!/bin/bash
#SBATCH --job-name=surface-filter-tail
#SBATCH --partition=amd96
#SBATCH --cpus-per-task=96
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:30:00
#SBATCH --output=/home/coulson/oc/lina4225/_runtime/jax-Ka/anm-pilot-v1/masif-audit-v1/surface-filter-tail-%j.log
set -euo pipefail
repo=/home/coulson/oc/lina4225/jax-Ka
pilot=/home/coulson/oc/lina4225/_runtime/jax-Ka/anm-pilot-v1
runtime="$pilot/masif-audit-v1"
export PYTHONPATH="$repo/src" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
code="$runtime/code/$SLURM_JOB_ID"
mkdir -p "$code"
cp "$repo/benchmarks/augmentation_surface_filter_accelerate.py" "$code/"
"$pilot/env/bin/python" "$code/augmentation_surface_filter_accelerate.py" --runtime "$runtime" --input "$runtime/results/surface-filter-${1:?pilot job}" --original-code "$runtime/code/$1" --output "$runtime/results/surface-filter-tail-$SLURM_JOB_ID"

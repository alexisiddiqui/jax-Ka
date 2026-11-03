#!/bin/bash
#SBATCH --job-name=surface-filter-pilot
#SBATCH --partition=amd96
#SBATCH --cpus-per-task=96
#SBATCH --mem-per-cpu=2G
#SBATCH --time=02:00:00
#SBATCH --output=/home/coulson/oc/lina4225/_runtime/jax-Ka/anm-pilot-v1/masif-audit-v1/surface-filter-%j.log
set -euo pipefail
repo=/home/coulson/oc/lina4225/jax-Ka
pilot=/home/coulson/oc/lina4225/_runtime/jax-Ka/anm-pilot-v1
runtime="$pilot/masif-audit-v1"
export PYTHONPATH="$repo/src" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
code="$runtime/code/$SLURM_JOB_ID"
mkdir -p "$code"
cp "$repo/benchmarks/augmentation_surface_filter.py" "$repo/benchmarks/augmentation_patch_scale.py" "$repo/benchmarks/augmentation_masif_audit.py" "$repo/benchmarks/augmentation_masif_bridge.py" "$repo/benchmarks/augmentation_masif_report.py" "$repo/benchmarks/augmentation_surface_audit.py" "$code/"
"$pilot/env/bin/python" "$code/augmentation_surface_filter.py" --runtime "$runtime" --output "$runtime/results/surface-filter-$SLURM_JOB_ID" --workers 24

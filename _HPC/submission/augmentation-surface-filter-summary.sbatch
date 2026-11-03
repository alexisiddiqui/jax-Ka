#!/bin/bash
#SBATCH --job-name=surface-filter-report
#SBATCH --partition=amd96
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:10:00
#SBATCH --output=/home/coulson/oc/lina4225/_runtime/jax-Ka/anm-pilot-v1/masif-audit-v1/surface-filter-report-%j.log
set -euo pipefail
repo=/home/coulson/oc/lina4225/jax-Ka
pilot=/home/coulson/oc/lina4225/_runtime/jax-Ka/anm-pilot-v1
runtime="$pilot/masif-audit-v1"
export PYTHONPATH="$repo/src" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
code="$runtime/code/$SLURM_JOB_ID"
mkdir -p "$code"
cp "$repo/benchmarks/augmentation_surface_filter_summary.py" "$repo/benchmarks/augmentation_surface_filter.py" "$repo/benchmarks/augmentation_patch_scale.py" "$repo/benchmarks/augmentation_masif_audit.py" "$repo/benchmarks/augmentation_masif_bridge.py" "$repo/benchmarks/augmentation_masif_report.py" "$repo/benchmarks/augmentation_surface_audit.py" "$code/"
extra=()
if [ -n "${3:-}" ]; then extra=(--memory "$runtime/results/surface-filter-cpu-$3"); fi
"$pilot/env/bin/python" "$code/augmentation_surface_filter_summary.py" --input "$runtime/results/surface-filter-${1:?pilot job}" --cpu "$runtime/results/surface-filter-cpu-${2:?CPU job}" --output "$runtime/results/surface-filter-final-$SLURM_JOB_ID" "${extra[@]}"

#!/bin/bash
#SBATCH --job-name=anm-modes-report
#SBATCH --partition=generalaccess
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:10:00
#SBATCH --output=/home/coulson/oc/lina4225/_runtime/jax-Ka/anm-pilot-v1/masif-audit-v1/modes10-report-%j.log
set -euo pipefail
repo=/home/coulson/oc/lina4225/jax-Ka
pilot=/home/coulson/oc/lina4225/_runtime/jax-Ka/anm-pilot-v1
runtime="$pilot/masif-audit-v1"
export PYTHONPATH="$repo/src" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
code="$runtime/code/modes10-report-$SLURM_JOB_ID"
mkdir -p "$code"
cp "$repo/benchmarks/augmentation_modes10_report.py" "$repo/benchmarks/augmentation_surface_audit.py" "$repo/benchmarks/augmentation_masif_report.py" "$code/"
extra=()
if [ -n "${2:-}" ]; then extra=(--control "$runtime/results/modes5-repeat-$2"); fi
"$pilot/env/bin/python" "$code/augmentation_modes10_report.py" --output "$runtime/results/modes10-${1:?array id}" --baseline "$runtime/results/surface-filter-final-751543" --exhaustive "$runtime/results/surface-filter-751526" --runtime "$runtime" "${extra[@]}"

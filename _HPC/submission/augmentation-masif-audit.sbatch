#!/bin/bash
#SBATCH --job-name=masif-surface-audit
#SBATCH --partition=amd96
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=2G
#SBATCH --time=04:00:00
#SBATCH --output=/home/coulson/oc/lina4225/_runtime/jax-Ka/anm-pilot-v1/masif-audit-v1/audit-%j.log
set -euo pipefail
repo=/home/coulson/oc/lina4225/jax-Ka
pilot=/home/coulson/oc/lina4225/_runtime/jax-Ka/anm-pilot-v1
runtime="$pilot/masif-audit-v1"
export PYTHONPATH="$repo/src" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python="$pilot/env/bin/python"
code="$runtime/code/$SLURM_JOB_ID"
mkdir -p "$code"
cp "$repo/benchmarks/augmentation_masif_audit.py" "$repo/benchmarks/augmentation_masif_bridge.py" \
   "$repo/benchmarks/augmentation_masif_report.py" "$repo/benchmarks/augmentation_surface_audit.py" "$code/"
entry="$code/augmentation_masif_audit.py"
cmd=${1:-smoke}
case "$cmd" in
  prepare)
    mkdir -p "$runtime/compat"
    cp /usr/lib64/libstdc++.so.6.0.25 "$runtime/compat/libstdc++.so.6"
    cp /usr/lib64/libgcc_s-8-20210514.so.1 "$runtime/compat/libgcc_s.so.1"
    "$python" "$entry" select --output "$runtime/results" \
      --store "$pilot/gh200-mixed-7379908/cpu-64" --historical "$pilot/native-large/units" \
      --previous-audit "$pilot/surface-audit-751294"
    "$python" "$entry" preflight --output "$runtime/results" --runtime "$runtime"
    ;;
  smoke)
    "$python" "$entry" preflight --output "$runtime/results" --runtime "$runtime"
    "$python" "$entry" run --output "$runtime/results" --runtime "$runtime" --workers 3 --smoke
    "$python" "$entry" report --output "$runtime/results" --smoke
    ;;
  full)
    "$python" "$entry" preflight --output "$runtime/results" --runtime "$runtime"
    "$python" "$entry" run --output "$runtime/results" --runtime "$runtime" --workers "${MASIF_WORKERS:-8}"
    "$python" "$entry" report --output "$runtime/results"
    ;;
  validate-staging)
    "$python" "$entry" validate-staging --output "$runtime/results" --runtime "$runtime"
    ;;
  report)
    "$python" "$entry" report --output "$runtime/results"
    ;;
  *) exit 2;;
esac

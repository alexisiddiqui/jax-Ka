#!/bin/bash
#SBATCH --job-name=masif-annotation
#SBATCH --partition=oc
#SBATCH --nodelist=comp1400
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=2G
#SBATCH --time=02:00:00
set -euo pipefail
root=/home/coulson/oc/lina4225/_runtime/jax-Ka/anm-pilot-v1
code="${1:?frozen code directory}"
output="${2:?pilot output}"
workers="${3:-8}"
export PYTHONPATH="$code/src" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export JAX_PLATFORMS=cpu
"$root/env/bin/python" -m pytest "$code/tests/test_masif_annotation_pilot.py" -q
exec "$root/env/bin/python" "$code/benchmarks/masif_annotation_pilot.py" \
  --coordinates "$root/pools-112-v1" --runtime "$root/../pkabench" \
  --masif "$root/masif-audit-v1" --output "$output" --workers "$workers" \
  --preprocessing paper --paper-runtime "$root/masif-paper-runtime-v1" \
  --dataset pkpdb --split train --separate-chains \
  --checkpoint "${MASIF_CHECKPOINT_PREFIX:-$root/masif-checkpoints/active/model}" "${@:4}"

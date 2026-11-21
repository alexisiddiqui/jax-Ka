#!/bin/bash
#SBATCH --job-name=masif-pkpdb-abag
#SBATCH --partition=amd96
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=2G
#SBATCH --time=12:00:00
set -euo pipefail
code="${1:?frozen code directory}"
inventory="${2:?pinned pKPDB Ab/Ag inventory.json}"
output="${3:?packed output directory}"
spool="${4:?shared GPU spool}"
base=/home/coulson/oc/lina4225/_runtime/jax-Ka
export PYTHONPATH="$code/src" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu
exec "$base/anm-pilot-v1/env/bin/python" "$code/benchmarks/masif_annotation_pilot.py" \
 --pkpdb-abag-inventory "$inventory" --runtime "$base/pkabench" \
 --masif "$base/anm-pilot-v1/masif-audit-v1" \
 --paper-runtime "$base/anm-pilot-v1/masif-paper-runtime-v1" \
 --checkpoint "$base/anm-pilot-v1/masif-checkpoints/active/model" \
 --preprocessing paper --dataset pkpdb --split all --full --workers 16 \
 --gpu-spool "$spool" --output "$output" "${@:5}"

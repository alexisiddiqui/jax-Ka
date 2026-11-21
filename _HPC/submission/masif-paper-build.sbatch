#!/bin/bash
#SBATCH --job-name=masif-paper-build
#SBATCH --partition=amd96
#SBATCH --cpus-per-task=96
#SBATCH --mem-per-cpu=2G
#SBATCH --time=12:00:00
set -euo pipefail
code="${1:?frozen code directory}"
output="${2:?node output under shared release root}"
workers="${3:-16}"
pilot=/home/coulson/oc/lina4225/_runtime/jax-Ka/anm-pilot-v1
export PYTHONPATH="$code/src" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu
exec "$pilot/env/bin/python" "$code/benchmarks/masif_annotation_pilot.py" \
 --coordinates "$pilot/pools-112-v1" --runtime "$pilot/../pkabench" --masif "$pilot/masif-audit-v1" \
 --paper-runtime "$pilot/masif-paper-runtime-v1" --checkpoint "$pilot/masif-checkpoints/active/model" \
 --preprocessing paper --dataset pkpdb --split train --separate-chains --full \
 --output "$output" --workers "$workers" "${@:4}"

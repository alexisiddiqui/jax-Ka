#!/bin/bash
#SBATCH --job-name=anm-feature-full
#SBATCH --partition=oc
#SBATCH --nodelist=comp1400
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=2G
#SBATCH --time=36:00:00
set -euo pipefail
pilot=/home/coulson/oc/lina4225/_runtime/jax-Ka/anm-pilot-v1
code="${1:?frozen source directory}"
output="${2:?output release directory}"
workers="${3:-24}"
export PYTHONPATH="$code/src" JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"$pilot/env/bin/python" -m pkatrain.augmentation_cache_generate \
  --coordinates "$pilot/pools-112-v1" --runtime "$pilot/../pkabench" \
  --output "$output" --workers "$workers" "${@:4}"

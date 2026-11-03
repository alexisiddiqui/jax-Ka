#!/bin/bash
#SBATCH --job-name=anm-feature-canary
#SBATCH --partition=oc
#SBATCH --nodelist=comp1400
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:45:00
set -euo pipefail
pilot=/home/coulson/oc/lina4225/_runtime/jax-Ka/anm-pilot-v1
code="${1:?frozen source directory}"
output="${2:?canary release directory}"
export PYTHONPATH="$code/src" JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"$pilot/env/bin/python" -m pytest "$code/tests/test_augmentation_cache_generate.py" \
  "$code/tests/test_augmentation_cache_pilot.py" "$code/tests/test_augmentation_release.py" -q
"$pilot/env/bin/python" -m pkatrain.augmentation_cache_generate \
  --coordinates "$pilot/pools-112-v1" --runtime "$pilot/../pkabench" \
  --output "$output" --workers 4 --canary-reference "$pilot/feature-cache-pilot-v2"

#!/bin/bash
#SBATCH --job-name=anm-feature-cache-pilot
#SBATCH --partition=oc
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --time=02:00:00
#SBATCH --output=/home/coulson/oc/lina4225/_runtime/jax-Ka/anm-pilot-v1/feature-cache-pilot-v2/build-%j.log
set -euo pipefail
pilot=/home/coulson/oc/lina4225/_runtime/jax-Ka/anm-pilot-v1
code="${1:?frozen source}"
export PYTHONPATH="$code/src" JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"$pilot/env/bin/python" -m pytest "$code/tests/test_augmentation_cache_pilot.py" "$code/tests/test_augmentation_release.py" -q
"$pilot/env/bin/python" -m pkatrain.augmentation_cache_pilot build --coordinates "$pilot/pools-112-v1" \
  --runtime "$pilot/../pkabench" --output "$pilot/feature-cache-pilot-v2" --sample 256 --workers 4 --allow-comp1400

#!/bin/bash
#SBATCH --job-name=masif-gpu-pilot
#SBATCH --partition=oc
#SBATCH --nodelist=comp1400
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:40:00
set -euo pipefail
base=/home/coulson/oc/lina4225/_runtime/jax-Ka
pilot="$base/anm-pilot-v1"
code="${1:?frozen code directory}"
output="${2:?pilot output directory}"
export PYTHONPATH="$code/src" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
JAX_PLATFORMS=cpu "$pilot/env/bin/python" "$code/benchmarks/masif_gpu_pilot.py" prepare \
  --output "$output" --native "$pilot/masif-annotation-v5/pilot" --masif "$pilot/masif-audit-v1" \
  --checkpoint "${MASIF_CHECKPOINT_PREFIX:-$pilot/masif-checkpoints/active/model}"
unset JAX_PLATFORMS
"$base/cuda12/python/bin/python" "$code/benchmarks/masif_gpu_pilot.py" gpu \
  --output "$output" --native "$pilot/masif-annotation-v5/pilot"

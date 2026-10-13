#!/bin/bash
#SBATCH --job-name=anm-loader-preflight
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=72
#SBATCH --mem-per-cpu=2G
#SBATCH --time=04:00:00
set -euo pipefail
scratch=/scratch/u6xq/hussain.u6xq
repo="$scratch/jax-Ka"
release="${AUGMENTATION_RELEASE_ROOT:-$scratch/_runtime/jax-Ka/anm-pilot-v1/infrastructure-v1}"
code="${AUGMENTATION_SOURCE_ROOT:-$repo}"
export PYTHONPATH="$code/src"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
protocol="${AUGMENTATION_PRODUCTION_PROTOCOL:-$scratch/_runtime/jax-Ka/pkabench/training/gqt-production-v2/runs/prod-100pct-w64-drop10-bur1-orient-dihed/protocol.json}"
"$repo/.venv/bin/python" -m pkatrain.augmentation_pilot verify-graphs "$release/graphs"
"$repo/.venv/bin/python" -m pkatrain.augmentation_pilot verify-native "$release/graphs" \
  --runtime "$scratch/_runtime/jax-Ka/pkabench"
"$repo/.venv/bin/python" -m pkatrain.augmentation_preflight reads "$release/graphs" "$release/host-read-preflight.json" \
  --workers 8 16 32 --prefetch 2 4 --batch 4
"$repo/.venv/bin/python" -m pkatrain.augmentation_preflight gpu "$release/graphs" "$release/gpu-preflight.json" \
  --workers 8 16 32 --prefetch 2 4 --protocol "$protocol"

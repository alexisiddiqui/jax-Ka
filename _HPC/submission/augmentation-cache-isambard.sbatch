#!/bin/bash
#SBATCH --job-name=anm-feature-cache-preflight
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=72
#SBATCH --mem-per-cpu=2G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/u6xq/hussain.u6xq/_runtime/jax-Ka/anm-pilot-v1/feature-cache-pilot-v2/gpu-%j.log
set -euo pipefail
base=/scratch/u6xq/hussain.u6xq
work="$base/_submission/augmentation-feature-cache-v1"
output="$base/_runtime/jax-Ka/anm-pilot-v1/feature-cache-pilot-v2"
python="$base/jax-Ka/.venv/bin/python"
local_work="${TMPDIR:-/tmp}/anm-feature-cache-$SLURM_JOB_ID"
mkdir -p "$local_work/base" "$local_work/overlay"
trap 'rm -rf "$local_work"' EXIT
tar -xzf "$work/training-src.tgz" -C "$local_work/base"
tar -xzf "${ANM_PILOT_ARCHIVE:-$work/pilot-overlay-final.tgz}" -C "$local_work/overlay"
export ANM_TRAINING_SOURCE="$local_work/base/src" ANM_PILOT_OVERLAY="$local_work/overlay"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONDONTWRITEBYTECODE=1 XDG_CACHE_HOME="$local_work/cache"
export TRITON_CACHE_DIR="$local_work/triton-cache" JAX_COMPILATION_CACHE_DIR="$local_work/jax-cache"
unset PYTHONPATH
"$python" "$work/augmentation_cache_launcher.py" verify-live --output "$output" --runtime "$base/_runtime/jax-Ka/pkabench"
"$python" "$work/augmentation_cache_launcher.py" benchmark --output "$output" --runtime "$base/_runtime/jax-Ka/pkabench" \
    --protocol "$work/protocol.json" --results "${ANM_CACHE_RESULTS:-$output/bench-current}" --loader-workers 16 --steps 20 "$@"

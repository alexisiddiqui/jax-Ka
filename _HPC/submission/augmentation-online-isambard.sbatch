#!/bin/bash
#SBATCH --job-name=anm-online-features
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=72
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:40:00
set -euo pipefail
base=/scratch/u6xq/hussain.u6xq
work="${ANM_ONLINE_SUBMISSION:-$base/_submission/augmentation-online-features-v1}"
output="${ANM_ONLINE_RESULTS:-$base/_runtime/jax-Ka/anm-pilot-v1/online-features-pilot-v1}"
python="$base/jax-Ka/.venv/bin/python"
local_work="$(mktemp -d /tmp/anm-online-$SLURM_JOB_ID-XXXXXX)"
mkdir -p "$local_work/base" "$local_work/overlay"
trap 'rm -rf "$local_work"' EXIT
tar -xzf "$base/_submission/augmentation-feature-cache-v1/training-src.tgz" -C "$local_work/base"
tar -xzf "$work/overlay.tgz" -C "$local_work/overlay"
export ANM_TRAINING_SOURCE="$local_work/base/src" ANM_PILOT_OVERLAY="$local_work/overlay"
mkdir -p "$local_work/tmp"
export TMPDIR="$local_work/tmp"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="$local_work/cache" TRITON_CACHE_DIR="$local_work/triton-cache" JAX_COMPILATION_CACHE_DIR="$local_work/jax-cache"
unset PYTHONPATH
"$python" "$local_work/overlay/tests/test_augmentation_online_features.py"
"$python" "$work/augmentation_online_launcher.py" \
  --pilot "$base/_runtime/jax-Ka/anm-pilot-v1/feature-cache-pilot-v2" \
  --compact "${ANM_ONLINE_COMPACT:-$output/compact}" --output "$output" --runtime "$base/_runtime/jax-Ka/pkabench" "$@"

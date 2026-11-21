#!/bin/bash
#SBATCH --job-name=masif-loader-check
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=72
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:15:00
set -euo pipefail
base=/scratch/u6xq/hussain.u6xq
submission="$base/_submission/masif-native-release-v1"
source="${1:?native MaSIF release}"
report="${2:?JSON output}"
limit="${3:-0}"
masif_loader_work="$(mktemp -d /tmp/masif-loader-$SLURM_JOB_ID-XXXXXX)"
trap 'rm -rf "$masif_loader_work"' EXIT
tar -xzf "$submission/loader-code.tgz" -C "$masif_loader_work"
export PYTHONPATH="$masif_loader_work/src" PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XDG_CACHE_HOME="$masif_loader_work/cache" JAX_COMPILATION_CACHE_DIR="$masif_loader_work/jax-cache"
unset JAX_PLATFORMS
exec "$base/jax-Ka/.venv/bin/python" "$masif_loader_work/masif_loader_pilot.py" \
 --source "$source" --output "$report" --gpu --max-assemblies "$limit"

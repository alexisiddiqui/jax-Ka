#!/bin/bash
#SBATCH --job-name=anm-pools-generate
#SBATCH --partition=generalaccess
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=/home/coulson/oc/lina4225/_runtime/jax-Ka/anm-pilot-v1/pools-112-v1/worker-%A_%a.log
set -euo pipefail
output=/home/coulson/oc/lina4225/_runtime/jax-Ka/anm-pilot-v1/pools-112-v1
pilot=/home/coulson/oc/lina4225/_runtime/jax-Ka/anm-pilot-v1
code="${1:?frozen source directory}"
offset="${2:-0}"
export PYTHONPATH="$code/src" JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENMM_CPU_THREADS=1
export TMPDIR="${SLURM_TMPDIR:-/tmp}"
"$pilot/env/bin/python" -m pkatrain.augmentation_generate worker --output "$output" --pilot "$pilot" --worker "$((SLURM_ARRAY_TASK_ID+offset))" "${@:3}"

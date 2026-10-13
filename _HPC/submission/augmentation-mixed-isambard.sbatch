#!/bin/bash
#SBATCH --job-name=anm-gh200-mixed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=72
#SBATCH --mem-per-cpu=2G
#SBATCH --time=02:00:00
set -euo pipefail
scratch=/scratch/u6xq/hussain.u6xq
repo="$scratch/jax-Ka"
work="$scratch/_submission/augmentation-gh200-repair-v1"
export PYTHONPATH="$repo/src"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENMM_CPU_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cpu
"$work/env/bin/python" -m pkatrain.augmentation_repair_benchmark run \
  "$work/inputs-mixed.tar" "$scratch/_runtime/jax-Ka/anm-pilot-v1/gh200-mixed-$SLURM_JOB_ID" \
  --mixed --cpu-workers 64 --mixed-cpu-workers 32 48 64

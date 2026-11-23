#!/bin/bash
#SBATCH --partition=amd96
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=2G
set -euo pipefail
B=/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench
CODE=$1; shift
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$CODE/src:$CODE/jobs JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_FLAGS="--xla_force_host_platform_device_count=1"
cd "$CODE"
exec "$B/envs/ogqt-main-7c3e3e9/bin/python" "$@"

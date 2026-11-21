#!/bin/bash
#SBATCH --job-name=masif-gpu-consumer
#SBATCH --partition=oc
#SBATCH --nodelist=comp1400
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --time=12:00:00
set -euo pipefail
code="${1:?frozen code directory}"
spool="${2:?shared inference spool}"
base=/home/coulson/oc/lina4225/_runtime/jax-Ka
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONDONTWRITEBYTECODE=1
unset JAX_PLATFORMS
exec "$base/cuda12/python/bin/python" "$code/benchmarks/masif_gpu_queue.py" \
 --runtime "$base/anm-pilot-v1/masif-paper-runtime-v1" --spool "$spool"

#!/bin/bash
#SBATCH --partition=oc
#SBATCH --nodelist=comp1400
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=2G
set -euo pipefail
B=/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench
CODE=$1; shift
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$CODE/jobs/a40site:$CODE/src:$CODE/jobs XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset LD_LIBRARY_PATH
cd "$CODE"
exec "$B/envs/ogqt-main-7c3e3e9/bin/python" "$@"

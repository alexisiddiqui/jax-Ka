#!/bin/bash
#SBATCH --partition=amd96
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2G
#SBATCH --output=/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/tierB/jaxpb-pipeline/energy/logs/%x-%j.out
# jaxpb-pipeline energy track on Coulson (CPU JAX, solver env). usage:
#   sbatch -J jpe-<name> -c <cpus> -t <time> energy_coulson.sbatch <code_dir> <test_script.py> <mode> [args...]
set -euo pipefail
PY=/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/envs/solver/bin/python
CODE=$1; shift
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1   # numpy BLAS single-threaded (cmf uses a process pool)
echo "job $SLURM_JOB_ID node $(hostname) cpus ${SLURM_CPUS_PER_TASK:-} code $CODE args $*"
cd "$CODE/tests"
$PY "$@"

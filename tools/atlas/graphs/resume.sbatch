#!/bin/bash
#SBATCH --job-name=atlas_graphs_sharded
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=72
#SBATCH --mem-per-cpu=2G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/u6xq/hussain.u6xq/_logs/atlas_graphs_sharded_%j.out
#SBATCH --error=/scratch/u6xq/hussain.u6xq/_logs/atlas_graphs_sharded_%j.err
set -euo pipefail
source /scratch/u6xq/hussain.u6xq/_submission/jaxka_env.sh
export PKABENCH_RUNTIME=$JAXKA_RUNTIME/pkabench
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export JAX_PLATFORMS=cpu
cd "$JAXKA_REPO"
A="$PKABENCH_RUNTIME/audits/atlas-graphs-v1"
.venv/bin/python "$A/inventory.py"
.venv/bin/python "$A/build.py"
.venv/bin/python "$A/finalize.py"
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
.venv/bin/python "$A/smoke.py"

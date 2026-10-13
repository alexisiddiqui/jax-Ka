#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/u6xq/hussain.u6xq/_logs/%x_%j.out
#SBATCH --error=/scratch/u6xq/hussain.u6xq/_logs/%x_%j.err
set -euo pipefail
source /scratch/u6xq/hussain.u6xq/_submission/jaxka_env.sh
export CODE=$S/_submission/invsqrt-v2
export PKABENCH_RUNTIME=$JAXKA_RUNTIME/pkabench XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH=$CODE/src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PKATRAIN_LOADER_WORKERS=15 PKATRAIN_PREFETCH_DEPTH=4
cd "$CODE"
"$JAXKA_REPO/.venv/bin/python" "$CODE/report_series.py"

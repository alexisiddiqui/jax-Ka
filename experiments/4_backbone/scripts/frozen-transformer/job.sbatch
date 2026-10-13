#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=2G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/u6xq/hussain.u6xq/_logs/%x_%j.out
#SBATCH --error=/scratch/u6xq/hussain.u6xq/_logs/%x_%j.err
set -euo pipefail
source /scratch/u6xq/hussain.u6xq/_submission/jaxka_env.sh
export CODE=$S/_submission/frozen-transformer-v1
export PKABENCH_RUNTIME=$JAXKA_RUNTIME/pkabench
export PYTHONPATH=$S/_submission/invsqrt-v2/src
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd "$CODE"
"$JAXKA_REPO/.venv/bin/python" "$CODE/train.py" "$@"

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
export CODE=$S/_submission/frozen-interface-v1
export PKABENCH_RUNTIME=$JAXKA_RUNTIME/pkabench
export PYTHONPATH=$S/_submission/invsqrt-v2/src
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd "$CODE"
if [ "$1" = smoke ]; then
  bash "$JAXKA_REPO/scripts/sqfs_run.sh" "$JAXKA_REPO/.venv/bin/python" "$CODE/probe.py" labels --smoke
  "$JAXKA_REPO/.venv/bin/python" "$CODE/probe.py" run --seed 17 --smoke
elif [ "$1" = labels ]; then
  bash "$JAXKA_REPO/scripts/sqfs_run.sh" "$JAXKA_REPO/.venv/bin/python" "$CODE/probe.py" labels
else
  "$JAXKA_REPO/.venv/bin/python" "$CODE/probe.py" run --seed "$1"
fi

#!/bin/bash
#SBATCH --partition=generalaccess,amd96
#SBATCH --exclude=comp1400
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2G
#SBATCH --time=02:00:00
#SBATCH --job-name=experimental-lofo
set -euo pipefail
source /home/coulson/oc/lina4225/_HPC/install/jax-Ka/pkabench/common.sh
export PKATRAIN_THREADS=2
"$PKABENCH_RUNTIME/envs/train-shared-v1/bin/python" "$PKABENCH_SOURCE/experiments/experimental_pilot/experimental_lofo.py" "$1"

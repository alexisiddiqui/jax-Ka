#!/bin/bash
#SBATCH --partition=generalaccess,amd96
#SBATCH --exclude=comp1400
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2G
#SBATCH --time=01:00:00
#SBATCH --job-name=exp-fit-select
set -euo pipefail
source /home/coulson/oc/lina4225/_HPC/install/jax-Ka/pkabench/common.sh
export PKATRAIN_THREADS=2
"$PKABENCH_RUNTIME/envs/train-shared-v1/bin/python" "$PKABENCH_SOURCE/experiments/experimental_pilot/select_experimental_fit.py"

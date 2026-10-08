#!/bin/bash
#SBATCH --partition=generalaccess,amd96
#SBATCH --exclude=comp1400
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:25:00
#SBATCH --array=0-2
#SBATCH --job-name=experimental-hybrid-mc
set -euo pipefail
source /home/coulson/oc/lina4225/_HPC/install/jax-Ka/pkabench/common.sh
export LD_LIBRARY_PATH="$PKABENCH_RUNTIME/fortran/lib"
structures=(1BNI 1IGD 1PGB)
for method in teacher catboost-17 catboost-29 catboost-43; do
    "$PKABENCH_RUNTIME/envs/pypka/bin/python" "$PKABENCH_SOURCE/experiments/experimental_pilot/hybrid_replay.py" "${structures[$SLURM_ARRAY_TASK_ID]}" "$method"
done

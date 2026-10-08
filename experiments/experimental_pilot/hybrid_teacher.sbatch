#!/bin/bash
#SBATCH --partition=generalaccess,amd96
#SBATCH --exclude=comp1400
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:35:00
#SBATCH --array=0-2
#SBATCH --job-name=experimental-pypka
set -euo pipefail
source /home/coulson/oc/lina4225/_HPC/install/jax-Ka/pkabench/common.sh
structures=(1BNI 1IGD 1PGB)
"$PKABENCH_RUNTIME/envs/runner/bin/python" "$PKABENCH_SOURCE/experiments/experimental_pilot/hybrid_teacher.py" "${structures[$SLURM_ARRAY_TASK_ID]}"

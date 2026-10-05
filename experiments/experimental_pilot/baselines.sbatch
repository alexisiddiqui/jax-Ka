#!/bin/bash
#SBATCH --partition=generalaccess,amd96
#SBATCH --exclude=comp1400
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:30:00
#SBATCH --array=0-8%9
#SBATCH --job-name=experimental-baselines
set -euo pipefail
source /home/coulson/oc/lina4225/_HPC/install/jax-Ka/pkabench/common.sh
structures=(1BNI 1IGD 1PGB)
methods=(propka pkai jaxka)
pdb=${structures[$((SLURM_ARRAY_TASK_ID / 3))]}
method=${methods[$((SLURM_ARRAY_TASK_ID % 3))]}
"$PKABENCH_RUNTIME/envs/runner/bin/python" "$PKABENCH_SOURCE/experiments/experimental_pilot/run_baseline.py" "$pdb" "$method"

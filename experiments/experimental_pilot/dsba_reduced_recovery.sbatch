#!/bin/bash
#SBATCH --partition=generalaccess,amd96
#SBATCH --exclude=comp1400
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2G
#SBATCH --time=01:00:00
#SBATCH --job-name=dsba-reduced
set -euo pipefail
source /home/coulson/oc/lina4225/_HPC/install/jax-Ka/pkabench/common.sh
env_name=runner
"$PKABENCH_RUNTIME/envs/$env_name/bin/python" "$PKABENCH_SOURCE/experiments/experimental_pilot/dsba_reduced_recovery.py" "$1"

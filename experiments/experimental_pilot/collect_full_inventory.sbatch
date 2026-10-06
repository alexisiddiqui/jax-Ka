#!/bin/bash
#SBATCH --partition=generalaccess,amd96
#SBATCH --exclude=comp1400
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:30:00
#SBATCH --job-name=pkadr-full-collect
set -euo pipefail
source /home/coulson/oc/lina4225/_HPC/install/jax-Ka/pkabench/common.sh
"$PKABENCH_RUNTIME/envs/runner/bin/python" "$PKABENCH_SOURCE/experiments/experimental_pilot/collect_full_inventory.py"

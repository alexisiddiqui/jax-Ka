#!/bin/bash
#SBATCH --partition=generalaccess,amd96
#SBATCH --exclude=comp1400
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2G
#SBATCH --time=01:00:00
#SBATCH --job-name=pkadr-full-audit
set -euo pipefail
source /home/coulson/oc/lina4225/_HPC/install/jax-Ka/pkabench/common.sh
if [[ ${1:-} == init ]]; then
 "$PKABENCH_RUNTIME/envs/runner/bin/python" "$PKABENCH_SOURCE/experiments/experimental_pilot/full_inventory.py" init
else
 "$PKABENCH_RUNTIME/envs/runner/bin/python" "$PKABENCH_SOURCE/experiments/experimental_pilot/full_inventory.py" shard "$SLURM_ARRAY_TASK_ID" 32
fi

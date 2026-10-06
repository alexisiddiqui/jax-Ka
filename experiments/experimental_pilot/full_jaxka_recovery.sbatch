#!/bin/bash
#SBATCH --partition=generalaccess,amd96
#SBATCH --exclude=comp1400
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:30:00
#SBATCH --job-name=pkadr-jax1024
set -euo pipefail
source /home/coulson/oc/lina4225/_HPC/install/jax-Ka/pkabench/common.sh
if [[ ${1:-} == init ]]; then
 "$PKABENCH_RUNTIME/envs/runner/bin/python" "$PKABENCH_SOURCE/experiments/experimental_pilot/full_jaxka_recovery.py" init
else
 "$PKABENCH_RUNTIME/envs/runner/bin/python" "$PKABENCH_SOURCE/experiments/experimental_pilot/full_jaxka_recovery.py" work
fi

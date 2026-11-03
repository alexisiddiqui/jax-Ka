#!/bin/bash
#SBATCH --job-name=anm-real-conformation
#SBATCH --partition=generalaccess
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:20:00
#SBATCH --output=/home/coulson/oc/lina4225/_runtime/jax-Ka/anm-pilot-v1/conformation-reference-v1/run-%j.log
set -euo pipefail
repo=/home/coulson/oc/lina4225/jax-Ka
pilot=/home/coulson/oc/lina4225/_runtime/jax-Ka/anm-pilot-v1
output="$pilot/conformation-reference-v1"
code="${2:?frozen source directory}"
export PYTHONPATH="$repo/src" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
args=()
if [[ "${1:?command}" == parent ]]; then args=(--index "${SLURM_ARRAY_TASK_ID:?}"); fi
"$pilot/env/bin/python" "$code/augmentation_conformation_reference.py" "$1" --pilot "$pilot" --output "$output" "${args[@]}"

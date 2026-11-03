#!/bin/bash
#SBATCH --job-name=anm-pools-finalize
#SBATCH --partition=generalaccess
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2G
#SBATCH --time=02:00:00
#SBATCH --output=/home/coulson/oc/lina4225/_runtime/jax-Ka/anm-pilot-v1/pools-112-v1/finalize-%j.log
set -euo pipefail
output=/home/coulson/oc/lina4225/_runtime/jax-Ka/anm-pilot-v1/pools-112-v1
pilot=/home/coulson/oc/lina4225/_runtime/jax-Ka/anm-pilot-v1
code="${1:?frozen source directory}"
export PYTHONPATH="$code/src" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"$pilot/env/bin/python" -m pkatrain.augmentation_generate finalize --output "$output" --pilot "$pilot" "${@:2}"

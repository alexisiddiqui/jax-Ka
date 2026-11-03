#!/bin/bash
#SBATCH --job-name=anm-quota-readiness
#SBATCH --partition=generalaccess
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:40:00
#SBATCH --output=/home/coulson/oc/lina4225/_runtime/jax-Ka/anm-pilot-v1/recipe-readiness-%j.log
set -euo pipefail
repo=/home/coulson/oc/lina4225/jax-Ka
pilot=/home/coulson/oc/lina4225/_runtime/jax-Ka/anm-pilot-v1
output="$pilot/recipe-112-v1"
code="${2:?frozen source directory}"
export PYTHONPATH="$code/src" JAX_PLATFORMS=cpu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENMM_CPU_THREADS=1
python="$pilot/env/bin/python"
if [[ "${1:?command}" == tests ]]; then
  "$python" -m pytest "$code/tests/test_augmentation_selection.py" "$code/tests/test_augmentation_release.py" -q
else
  "$python" "$code/benchmarks/augmentation_recipe_readiness.py" "$1" --pilot "$pilot" --output "$output" --base "$pilot/infrastructure-v1"
fi

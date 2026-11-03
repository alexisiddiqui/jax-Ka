#!/bin/bash
#SBATCH --job-name=anm-pools-preflight
#SBATCH --partition=oc
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:30:00
#SBATCH --output=/home/coulson/oc/lina4225/_runtime/jax-Ka/anm-pilot-v1/pools-112-v1/preflight-%j.log
set -euo pipefail
output=/home/coulson/oc/lina4225/_runtime/jax-Ka/anm-pilot-v1/pools-112-v1
pilot=/home/coulson/oc/lina4225/_runtime/jax-Ka/anm-pilot-v1
code="${1:?frozen source directory}"
export PYTHONPATH="$code/src" JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENMM_CPU_THREADS=1
export TMPDIR="${SLURM_TMPDIR:-/tmp}"
python="$pilot/env/bin/python"
"$python" -m pytest "$code/tests/test_augmentation_selection.py" "$code/tests/test_augmentation_release.py" "$code/tests/test_augmentation_generate.py" -q
"$python" -m pkatrain.augmentation_generate register --output "$output" --pilot "$pilot" --allow-comp1400
"$python" -m pkatrain.augmentation_generate worker --output "$output" --pilot "$pilot" --canary --allow-comp1400

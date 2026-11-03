#!/bin/bash
#SBATCH --job-name=ogqt-triton-check
#SBATCH --partition=oc
#SBATCH --nodelist=comp1400
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:10:00
#SBATCH --output=/home/coulson/oc/lina4225/_runtime/jax-Ka/logs/ogqt-triton-compat-%j.out

set -euo pipefail
repo=/home/coulson/oc/lina4225/jax-Ka
runtime=/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench
source=/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/sources/jax-Ka-main-7c3e3e9
python=/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/envs/train-shared-gpu-v1/bin/python
overlay=/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/overlays/ogqt-triton-v1
export PYTHONPATH="$source/src:$repo/src:$overlay:/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/envs/runner/lib/python3.11/site-packages"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=2
cd "$repo"
"$python" benchmarks/ogqt_triton_compat_check.py \
  --runtime "$runtime" --source "$source" \
  --output "$runtime/audits/ogqt-production-frozen-v1"

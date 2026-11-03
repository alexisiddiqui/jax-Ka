#!/bin/bash
#SBATCH --job-name=shift-bin-models
#SBATCH --partition=oc
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:10:00
#SBATCH --output=/home/coulson/oc/lina4225/_runtime/jax-Ka/logs/shift-bin-model-comparison-%j.out

set -euo pipefail
repo=/home/coulson/oc/lina4225/jax-Ka
runtime=/home/coulson/oc/lina4225/_runtime/jax-Ka
python=/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/envs/train-shared-gpu-v1/bin/python
export PYTHONPATH="$repo/src:/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/envs/runner/lib/python3.11/site-packages"
cd "$repo"
"$python" benchmarks/shift_bin_model_comparison.py \
  --runtime "$runtime" \
  --output "$runtime/pkabench/audits/shift-bin-model-comparison-v1"

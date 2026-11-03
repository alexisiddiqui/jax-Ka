#!/bin/bash
#SBATCH --job-name=ogqt-prod-post
#SBATCH --partition=oc
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:20:00
#SBATCH --output=/home/coulson/oc/lina4225/_runtime/jax-Ka/logs/ogqt-production-postprocess-%j.out

set -euo pipefail
repo=/home/coulson/oc/lina4225/jax-Ka
runtime=/home/coulson/oc/lina4225/_runtime/jax-Ka
python=/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/envs/train-shared-gpu-v1/bin/python
export PYTHONPATH="$repo/src:/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/envs/runner/lib/python3.11/site-packages"
cd "$repo"

"$python" benchmarks/kd_ratio_comparison_production_ogqt.py \
  --runtime "$runtime" \
  --output "$runtime/pkabench/audits/kd-ratio-comparison-v5-ogqt-production-exact" \
  --production-audit ogqt-production-frozen-v2-exact \
  --bootstrap 2000

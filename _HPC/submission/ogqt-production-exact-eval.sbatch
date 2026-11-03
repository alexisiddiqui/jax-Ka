#!/bin/bash
#SBATCH --job-name=ogqt-prod-exact
#SBATCH --partition=oc
#SBATCH --nodelist=comp1400
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:30:00
#SBATCH --output=/home/coulson/oc/lina4225/_runtime/jax-Ka/logs/ogqt-production-exact-%j.out

set -euo pipefail
repo=/home/coulson/oc/lina4225/jax-Ka
runtime=/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench
source=/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/sources/jax-Ka-main-7c3e3e9
python=/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/envs/ogqt-main-7c3e3e9/bin/python
export PYTHONPATH="$source/src:$repo/src"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=2
cd "$repo"
"$python" benchmarks/ogqt_production_frozen_eval.py \
  --runtime "$runtime" --source "$source" \
  --output "$runtime/audits/ogqt-production-frozen-v2-exact" \
  ${OGQT_EVAL_ARGS:-}

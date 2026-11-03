#!/bin/bash
#SBATCH --job-name=ogqt-prod-eval
#SBATCH --partition=oc
#SBATCH --nodelist=comp1400
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:30:00
#SBATCH --output=/home/coulson/oc/lina4225/_runtime/jax-Ka/logs/ogqt-production-frozen-%j.out

set -euo pipefail
runtime=/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench
source_root=$runtime/sources/jax-Ka-main-7c3e3e9
export PYTHONPATH=$source_root/src:/home/coulson/oc/lina4225/jax-Ka/src:$runtime/overlays/ogqt-triton-v1:$runtime/envs/runner/lib/python3.11/site-packages
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
$runtime/envs/train-shared-gpu-v1/bin/python /home/coulson/oc/lina4225/jax-Ka/benchmarks/ogqt_production_frozen_eval.py \
  --runtime $runtime --source $source_root \
  --output $runtime/audits/ogqt-production-frozen-v1 ${OGQT_EVAL_ARGS:-}

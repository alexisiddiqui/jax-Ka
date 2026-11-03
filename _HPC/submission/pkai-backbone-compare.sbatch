#!/bin/bash
#SBATCH --job-name=pkai-bb-cmp
#SBATCH --partition=oc
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:20:00
#SBATCH --output=/home/coulson/oc/lina4225/_runtime/jax-Ka/logs/pkai-backbone-compare-%j.out

set -euo pipefail
export PYTHONPATH=/home/coulson/oc/lina4225/jax-Ka/src
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/envs/runner/bin/python \
  /home/coulson/oc/lina4225/jax-Ka/benchmarks/pkai_backbone_compare.py

#!/bin/bash
#SBATCH --job-name=pkai-backbone
#SBATCH --partition=oc
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2G
#SBATCH --time=01:00:00
#SBATCH --output=/home/coulson/oc/lina4225/_runtime/jax-Ka/logs/pkai-backbone-%A_%a.out

set -euo pipefail
runtime=/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench
export PYTHONPATH=/home/coulson/oc/lina4225/jax-Ka/src:$runtime/sources/smooth_propka-pkai-backbone/src:$runtime/sources/pKAI-fork-backbone:$runtime/sources/pKAI-fork-backbone/pKAI:$runtime/overlays/smooth-propka-py311:$runtime/envs/runner/lib/python3.11/site-packages
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
$runtime/envs/smooth-propka-v1/bin/python -m pkabench.smooth_pkai_backbone_pinder_eval work \
  --runtime "$runtime" --index "${SLURM_ARRAY_TASK_ID:-0}"

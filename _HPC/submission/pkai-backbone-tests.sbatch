#!/bin/bash
#SBATCH --job-name=pkai-bb-tests
#SBATCH --partition=oc
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:30:00
#SBATCH --output=/home/coulson/oc/lina4225/_runtime/jax-Ka/logs/pkai-backbone-tests-%j.out

set -euo pipefail
runtime=/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench
export PYTHONPATH=$runtime/sources/smooth_propka-pkai-backbone/src:$runtime/sources/pKAI-fork-backbone:$runtime/sources/pKAI-fork-backbone/pKAI:$runtime/overlays/smooth-propka-py311:$runtime/envs/runner/lib/python3.11/site-packages
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
cd $runtime/sources/smooth_propka-pkai-backbone
$runtime/envs/smooth-propka-v1/bin/python -m pytest -q tests/test_pkai.py \
  -k 'hard_native_oracle_and_smooth_fixture or backbone_features_ignore_side_chains or default_estimator_masks_atom16 or invariance_batching_chunks_and_checkpoint'

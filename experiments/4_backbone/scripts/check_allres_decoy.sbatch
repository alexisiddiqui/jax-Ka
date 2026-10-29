#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=72
#SBATCH --mem-per-cpu=2G
#SBATCH --time=01:00:00
#SBATCH --exclude=nid010229,nid010639
#SBATCH --output=/scratch/u6xq/hussain.u6xq/_logs/%x_%j.out
#SBATCH --error=/scratch/u6xq/hussain.u6xq/_logs/%x_%j.err
# Decoy-ranking all-residue export (models/ogqt-allres15-siamese-decoy-50, run decoy-v1-50-both-w064-s17) vs its
# training run's stored validation predictions (2026-10-29), as check_allres_default.sbatch does for the default.
# Code: _submission/decoy-export-check-v1 (branch decoy-export of jax-Ka).
set -euo pipefail
source /scratch/u6xq/hussain.u6xq/_submission/jaxka_env.sh
CODE=$S/_submission/decoy-export-check-v1
export PKABENCH_RUNTIME=$JAXKA_RUNTIME/pkabench XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=$CODE/src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export GQT_DEFAULT_MODEL=$CODE/models/ogqt-allres15-siamese-decoy-50
PY=$JAXKA_REPO/.venv/bin/python
cd "$CODE"
sha256sum src/pkanet/{allres,attention_backend,cli,features}.py src/pkatrain/production_{defaults,position}.py "$GQT_DEFAULT_MODEL/params.npz"
A=$PKABENCH_RUNTIME/audits/decoy-export-check-v1
for b in triton jax; do bash "$JAXKA_REPO/scripts/sqfs_run.sh" "$PY" experiments/4_backbone/scripts/check_allres_default.py "$A/check-$b.json" "$b" 100; done
GQT_DEFAULT_MODEL= "$PY" -m pytest -q tests/test_allres_default.py tests/test_relational_structure.py 2>&1 | tail -5

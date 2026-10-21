#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=72
#SBATCH --mem-per-cpu=2G
#SBATCH --time=01:30:00
#SBATCH --exclude=nid010229,nid010639
#SBATCH --output=/scratch/u6xq/hussain.u6xq/_logs/%x_%j.out
#SBATCH --error=/scratch/u6xq/hussain.u6xq/_logs/%x_%j.err
# Default all-residue export vs its training run (2026-10-21, user: "make this the default weights and endpoint. Update
# the CLI etc to be able to handle this"). Code: _submission/allres-default-v1 (branch allres-default of jax-Ka).
set -euo pipefail
source /scratch/u6xq/hussain.u6xq/_submission/jaxka_env.sh
CODE=$S/_submission/allres-default-v1
export PKABENCH_RUNTIME=$JAXKA_RUNTIME/pkabench XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=$CODE/src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
PY=$JAXKA_REPO/.venv/bin/python
cd "$CODE"
sha256sum src/pkanet/{allres,attention_backend,cli,features}.py src/pkatrain/production_{defaults,position}.py models/ogqt-allres15-siamese-50/params.npz
A=$PKABENCH_RUNTIME/audits/allres-default-v1
for b in triton jax; do bash "$JAXKA_REPO/scripts/sqfs_run.sh" "$PY" experiments/4_backbone/scripts/check_allres_default.py "$A/check-$b.json" "$b" 100; done
"$PY" -m pytest -q tests/test_allres_default.py tests/test_relational_structure.py 2>&1 | tail -5

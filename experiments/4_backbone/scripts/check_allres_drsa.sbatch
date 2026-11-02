#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=72
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/u6xq/hussain.u6xq/_logs/%x_%j.out
#SBATCH --error=/scratch/u6xq/hussain.u6xq/_logs/%x_%j.err
# Release the selected drsa-within-v1 checkpoints (2026-11-02), without retraining.
# Submit STEP=export first, then STEP=check --array=0-2 --dependency=afterok:<export job>.
set -euo pipefail
source /scratch/u6xq/hussain.u6xq/_submission/jaxka_env.sh
CODE=$S/_submission/drsa-weight-release-v1
export PKABENCH_RUNTIME=$JAXKA_RUNTIME/pkabench XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 GQT_PARAMETER_SCALING=1
PY=$JAXKA_REPO/.venv/bin/python
cd "$CODE"
if [ "${STEP:-}" = export ]; then
  # Restore with the exact training implementation, not the newer public inference source.
  export PYTHONPATH=$S/_submission/drsa-within-v1/src
  "$PY" - "$CODE" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import jax
import numpy as np
from pkatrain.production_train import export_model, selected_model, load_exported

code = Path(__import__('sys').argv[1]); root = Path(os.environ['PKABENCH_RUNTIME'])
pins = {
    17: (18, '703e17a7e2fcb109d44e510eb58db802f4f0c1cf2a3e994fad73d1d7a2427607'),
    29: (20, '1de40443be8dcfc7e9294821a133c6ef794095481dbd481421b24adfa0dc12d2'),
    43: (16, 'd457cd9a6edaa0b619d0cfa063415d4a6e0c2d46f5da96ca4ecc5d8dce8ffc63'),
}
for seed, (epoch, sha) in pins.items():
    run = f'drsa-v1-50-within03-asym50-w064-s{seed}'
    checkpoint = root / 'training/gqt-production-v2/runs' / run / 'checkpoints' / f'epoch-{epoch:03d}' / 'state.npz'
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == sha
    name = 'ogqt-allres15-within-drsa-50' + (f'-s{seed}' if seed != 17 else '')
    dest = export_model(root, run, code / 'models' / name)
    info = json.loads((dest / 'model.json').read_text())
    assert info['checkpoint_sha256'] == sha and info['selected_epoch'] == epoch
    assert info['parameter_count'] == 180000 and info['protocol']['within_listwise']['target'] == 'drsa'
    original, _ = selected_model(root, run); exported = load_exported(dest)
    assert jax.tree.structure(original) == jax.tree.structure(exported)
    for a, b in zip(jax.tree.leaves(original), jax.tree.leaves(exported)):
        np.testing.assert_array_equal(a, b)
    (dest / 'verification-parameters.json').write_text(json.dumps({
        'run': run, 'selected_epoch': epoch, 'checkpoint_sha256': sha,
        'params_sha256': info['params_sha256'], 'parameter_count': info['parameter_count'],
        'restored_parameter_equality': 'bitwise', 'job': os.environ['SLURM_JOB_ID'],
    }, indent=2) + '\n')
    print(name, 'parameter equality passed', flush=True)
PY
elif [ "${STEP:-}" = check ]; then
  export PYTHONPATH=$CODE/src
  NAMES=(ogqt-allres15-within-drsa-50 ogqt-allres15-within-drsa-50-s29 ogqt-allres15-within-drsa-50-s43)
  export GQT_DEFAULT_MODEL=$CODE/models/${NAMES[${SLURM_ARRAY_TASK_ID:?}]}
  for backend in triton jax; do
    bash "$JAXKA_REPO/scripts/sqfs_run.sh" "$PY" experiments/4_backbone/scripts/check_allres_default.py \
      "$GQT_DEFAULT_MODEL/verification-$backend.json" "$backend" 20
  done
  "$PY" - "$GQT_DEFAULT_MODEL" <<'PY'
import json
from pathlib import Path
import sys
for backend in ('jax', 'triton'):
    r = json.loads((Path(sys.argv[1]) / f'verification-{backend}.json').read_text())
    assert r['complexes'] == 20 and r['matched_sites'] > 300, r
    assert r['unmatched_sites'] / (r['matched_sites'] + r['unmatched_sites']) < .05, r
    for state in ('bound', 'free'):
        assert r[f'{state}_abs_diff']['max'] < .01, r
        assert r[f'{state}_abs_diff']['mean'] < .001, r
PY
  if [ "$SLURM_ARRAY_TASK_ID" = 0 ]; then
    GQT_DEFAULT_MODEL= "$PY" -m pytest -q tests/test_allres_default.py tests/test_relational_structure.py \
      2>&1 | tee models/ogqt-allres15-within-drsa-50/verification-tests.txt
  fi
else
  echo 'Set STEP=export or STEP=check' >&2; exit 2
fi

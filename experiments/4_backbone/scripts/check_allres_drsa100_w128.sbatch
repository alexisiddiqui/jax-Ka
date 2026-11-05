#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=72
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/u6xq/hussain.u6xq/_logs/%x_%j.out
#SBATCH --error=/scratch/u6xq/hussain.u6xq/_logs/%x_%j.err
# Release the 100%-pool width-128 dRSA checkpoint (datascale-v1 run drsa100-v1-asym50-w128-s17, epoch 20; 2026-11-05),
# without retraining. Submit STEP=export first, then STEP=check --dependency=afterok:<export job>.
set -euo pipefail
source /scratch/u6xq/hussain.u6xq/_submission/jaxka_env.sh
CODE=$S/_submission/export-drsa100-w128-v1
NAME=ogqt-allres15-within-drsa-100-w128
export PKABENCH_RUNTIME=$JAXKA_RUNTIME/pkabench XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
PY=$JAXKA_REPO/.venv/bin/python
cd "$CODE"
if [ "${STEP:-}" = export ]; then
  # Restore with the exact training implementation (datascale-v1 = drsa-cap-v1 src), not the newer public inference
  # source; that snapshot opens width 128 only with GQT_PARAMETER_SCALING=1.
  export PYTHONPATH=$S/_submission/datascale-v1/src GQT_PARAMETER_SCALING=1
  "$PY" - "$CODE" "$NAME" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path
import jax
import numpy as np
from pkatrain.production_train import export_model, selected_model, load_exported

code, name = Path(sys.argv[1]), sys.argv[2]; root = Path(os.environ['PKABENCH_RUNTIME'])
run, epoch, sha = 'drsa100-v1-asym50-w128-s17', 20, '50d53402c43c7e330e9db9e3e0ee86f00450236b04247f9f82e8a51da45ab25b'
checkpoint = root / 'training/gqt-production-v2/runs' / run / 'checkpoints' / f'epoch-{epoch:03d}' / 'state.npz'
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == sha
dest = export_model(root, run, code / 'models' / name)
info = json.loads((dest / 'model.json').read_text())
assert info['checkpoint_sha256'] == sha and info['selected_epoch'] == epoch and info['epochs_run'] == 20
assert info['parameter_count'] == 693856 and info['architecture'] == {'dihedral_features': 6, 'ff': 256, 'orientation_edges': 13, 'width': 128}
protocol = info['protocol']
assert protocol['within_listwise']['target'] == 'drsa' and protocol['fraction'] == 1.0
assert protocol['production_variant'] == 'relations-4-1-allres-cut15' and protocol['hotspot_mask']['store'].endswith('/store-hotspot-v2')
assert hashlib.sha256((dest / 'params.npz').read_bytes()).hexdigest() == info['params_sha256']
original, _ = selected_model(root, run); exported = load_exported(dest)
assert jax.tree.structure(original) == jax.tree.structure(exported)
for a, b in zip(jax.tree.leaves(original), jax.tree.leaves(exported)):
    assert a.dtype == b.dtype and a.shape == b.shape
    np.testing.assert_array_equal(a, b)
(dest / 'verification-parameters.json').write_text(json.dumps({
    'run': run, 'selected_epoch': epoch, 'checkpoint_sha256': sha,
    'params_sha256': info['params_sha256'], 'parameter_count': info['parameter_count'],
    'architecture': info['architecture'], 'restored_parameter_equality': 'bitwise', 'job': os.environ['SLURM_JOB_ID'],
}, indent=2) + '\n')
print(name, 'parameter equality passed', flush=True)
PY
elif [ "${STEP:-}" = check ]; then
  export PYTHONPATH=$CODE/src
  export GQT_DEFAULT_MODEL=$CODE/models/$NAME
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
    assert r['run'] == 'drsa100-v1-asym50-w128-s17' and r['backend'] == backend and r['device'].startswith('cuda'), r
    assert r['complexes'] == 20 and r['matched_sites'] > 300, r
    assert r['unmatched_sites'] / (r['matched_sites'] + r['unmatched_sites']) < .05, r
    for state in ('bound', 'free'):
        assert r[f'{state}_abs_diff']['max'] < .01, r
        assert r[f'{state}_abs_diff']['mean'] < .001, r
print('parity checks passed')
PY
  GQT_DEFAULT_MODEL= "$PY" -m pytest -q tests/test_allres_default.py tests/test_relational_structure.py \
    2>&1 | tee models/$NAME/verification-tests.txt
else
  echo 'Set STEP=export or STEP=check' >&2; exit 2
fi

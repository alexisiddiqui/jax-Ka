#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=72
#SBATCH --mem-per-cpu=2G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/u6xq/hussain.u6xq/_logs/%x_%j.out
#SBATCH --error=/scratch/u6xq/hussain.u6xq/_logs/%x_%j.err
# abag-fv-check-v1 (2026-11-20): PINDER val Fab records scored as one antibody chain (training unit) vs whole Fv (POISK
# unit). Exports the pool-v5 s17 runs (control, q03 with quartet-rank-v1b; qc, conly with quartet-rank-v2) into
# $A/models (params.npz + model.json each), links the published ogqt-allres15-within-drsa-50 as a reference, then runs
# score_fv.py on node-local copies of the assembled structures. Writes $A/fv_check.json + $A/per_case.tsv.
set -euo pipefail
source /scratch/u6xq/hussain.u6xq/_submission/jaxka_env.sh
C=$S/_submission/abag-fv-check-v1; R=$JAXKA_RUNTIME/pkabench; A=$R/audits/abag-fv-check-v1
export PKABENCH_RUNTIME=$R PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false GQT_PARAMETER_SCALING=1
PY=$JAXKA_REPO/.venv/bin/python
echo "[job] $(date -Is) ${SLURMD_NODENAME:-}"; sha256sum "$C"/score_fv.py "$C"/fvcheck.tgz "$0"; (cd $JAXKA_REPO && git log --oneline -1)
mkdir -p "$A/models"
for arm in control q03; do
  [ -f "$A/models/$arm/model.json" ] || PYTHONPATH=$S/_submission/quartet-rank-v1b/src "$PY" -m pkatrain.production_train export pool5-v1-50-asym50-$arm-w064-s17 "$A/models/$arm"; done
for arm in qc conly; do
  [ -f "$A/models/$arm/model.json" ] || PYTHONPATH=$S/_submission/quartet-rank-v2/src "$PY" -m pkatrain.production_train export pool5-v2-50-asym50-$arm-w064-s17 "$A/models/$arm"; done
[ -e "$A/models/zz-published-drsa50" ] || ln -s "$JAXKA_REPO/models/ogqt-allres15-within-drsa-50" "$A/models/zz-published-drsa50"
W=/tmp/fvcheck-$SLURM_JOB_ID; mkdir -p $W; tar xzf "$C/fvcheck.tgz" -C $W
"$PY" "$C/score_fv.py" "$W" "$A/models" "$A"
rm -rf $W; echo "[job] $(date -Is) done"

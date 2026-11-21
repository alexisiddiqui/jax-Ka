#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=72
#SBATCH --mem-per-cpu=2G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/scratch/u6xq/hussain.u6xq/_logs/%x_%j.out
#SBATCH --error=/scratch/u6xq/hussain.u6xq/_logs/%x_%j.err
# quartet-rank-v2 (2026-11-19): pool-v5 asym50 dRSA recipe (quartet-rank-v1b) + cross-complex chain quartet loss (production_residue.chain_*).
#   STEP qc    -> pool5-v2-50-asym50-qc-w064-s$SEED     (--quartet-weight 0.3 --quartet-count 128 + chain CHAINW)
#   STEP conly -> pool5-v2-50-asym50-conly-w064-s$SEED  (chain CHAINW only; residue quartet off)
# CHAINW (required, > 0): the chain quartet weight set from the smoke; ABAGMIN (optional, default 2): Ab/Ag per regrouped batch.
# Bus-error guard and requeue as quartet-rank-v1 train.sbatch (a requeued run resumes from its history).
#   sbatch -J q2_qc_s17    --export=ALL,STEP=qc,SEED=17,CHAINW=<w> train.sbatch
#   sbatch -J q2_conly_s17 --export=ALL,STEP=conly,SEED=17,CHAINW=<w> train.sbatch
set -uo pipefail
source /scratch/u6xq/hussain.u6xq/_submission/quartet-rank-v2/jobs/common.sh
export PYTHONPATH=$CODE/src XLA_PYTHON_CLIENT_PREALLOCATE=false REQUEUE=1
STEP=${STEP:-${1:-}}; SEED=${SEED:-17}; CHAINW=${CHAINW:-}
if ! awk -v w="$CHAINW" 'BEGIN { exit !(w ~ /^[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$/ && w + 0 > 0) }'; then
  echo "[job] CHAINW must be set to a positive number (got '$CHAINW')" >&2; exit 2; fi
case "$STEP" in
  qc) ARM=(--quartet-weight 0.3 --quartet-count 128) ;;
  conly) ARM=() ;;
  *) echo "unknown step '$STEP' (qc | conly)" >&2; exit 2 ;;
esac
ulimit -c 0; cd "$CODE"; echo "[job] $(date -Is) ${SLURMD_NODENAME:-} restart=${SLURM_RESTART_COUNT:-0} step=$STEP seed=$SEED chainw=$CHAINW abagmin=${ABAGMIN:-2}"
sha256sum "$CODE"/src/pkatrain/production_{train,extra,loading,residue,position}.py "$0" 2>/dev/null || true
RUN=pool5-v2-50-asym50-$STEP-w064-s$SEED
for f in "$STORE_V2/verification.json" "$P/store-hotspot-v5abag/verification.json" "$R/training/pool-v5/pinder-manifest.json"; do
  [ -f "$f" ] || { echo "[job] missing $f" >&2; exit 3; }; done
A=$R/audits/quartet-rank-v2/train/$STEP-s$SEED; mkdir -p "$A"
echo "[job] run=$RUN ${ARM[*]} $(chain_flags "$CHAINW")"
tstart=$(date +%s)
gpu "$PY" -m pkatrain.production_train train "$RUN" $(recipe "$STORE_V2") --seed "$SEED" "${ARM[@]}" $(chain_flags "$CHAINW") "${POOL5[@]}"
rc=$?; t1=$(date +%s)
printf '{"run": "%s", "job": "%s", "node": "%s", "exit": %d, "wall_seconds": %d, "chain_weight": "%s", "abag_min": "%s"}\n' \
  "$RUN" "${SLURM_JOB_ID:-}" "${SLURMD_NODENAME:-}" "$rc" "$((t1 - tstart))" "$CHAINW" "${ABAGMIN:-2}" | tee "$A/train_resources.json"
echo "[job] $(date -Is) exit=$rc"; exit $rc

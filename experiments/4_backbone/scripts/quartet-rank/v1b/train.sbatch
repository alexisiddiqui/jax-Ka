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
# quartet-rank-v1b (2026-11-16): the asym50 dRSA recipe on pool-v5 (pool-v4 + Ab/Ag paratope screen; training/pool-v5) through
# --pinder-manifest / --pinder-extra / --mask-store-extra (production_extra; store-hotspot-v2 + store-hotspot-v5abag for the masks).
#   control -> pool5-v1-50-asym50-control-w064-s$SEED (quartet weight 0 = the drsa control objective, bit-identical by test)
#   q03     -> pool5-v1-50-asym50-q03-w064-s$SEED     (--quartet-weight 0.3 --quartet-count 128, as quartet-rank-v1 q03)
# Bus-error guard and requeue as quartet-rank-v1 train.sbatch.
#   sbatch -J p5_control_s17 --export=ALL,STEP=control,SEED=17 train.sbatch
#   sbatch -J p5_q03_s17     --export=ALL,STEP=q03,SEED=17 train.sbatch
set -uo pipefail
source /scratch/u6xq/hussain.u6xq/_submission/quartet-rank-v1b/jobs/common.sh
export PYTHONPATH=$CODE/src XLA_PYTHON_CLIENT_PREALLOCATE=false
STEP=${STEP:-${1:-}}; SEED=${SEED:-17}
ulimit -c 0; cd "$CODE"; echo "[job] $(date -Is) ${SLURMD_NODENAME:-} restart=${SLURM_RESTART_COUNT:-0} step=$STEP"
sha256sum "$CODE"/src/pkatrain/production_{train,extra,loading,residue,position}.py "$0" 2>/dev/null || true
case "$STEP" in
  control) ARM=() ;;
  q03) ARM=(--quartet-weight 0.3 --quartet-count 128) ;;
  *) echo "unknown step '$STEP' (control | q03)" >&2; exit 2 ;;
esac
RUN=pool5-v1-50-asym50-$STEP-w064-s$SEED
for f in "$STORE_V2/verification.json" "$P/store-hotspot-v5abag/verification.json" "$R/training/pool-v5/pinder-manifest.json"; do
  [ -f "$f" ] || { echo "[job] missing $f" >&2; exit 3; }; done
gpu() {
  for attempt in 1 2 3; do
    t0=$SECONDS; rc=0; "$@" || rc=$?
    [ $rc -eq 135 ] && [ $((SECONDS - t0)) -lt 120 ] || return $rc
    echo "[job] bus error at startup, attempt $attempt" >&2; sleep 20
  done
  if [ "${SLURM_RESTART_COUNT:-0}" -lt 4 ]; then
    ex=$(scontrol show job "$SLURM_JOB_ID" | sed -n 's/.*ExcNodeList=\([^ ]*\).*/\1/p'); [ "$ex" = "(null)" ] && ex=""
    scontrol update job="$SLURM_JOB_ID" ExcNodeList="${ex:+$ex,}$SLURMD_NODENAME"; echo "[job] requeue, excluding $SLURMD_NODENAME" >&2
    scontrol requeue "$SLURM_JOB_ID"; sleep 60
  fi
  return 135
}
A=$R/audits/abdata-v5/train/$STEP-s$SEED; mkdir -p "$A"
tstart=$(date +%s)
gpu "$PY" -m pkatrain.production_train train "$RUN" $(recipe "$STORE_V2") --seed "$SEED" "${ARM[@]}" "${POOL5[@]}"
rc=$?; t1=$(date +%s)
printf '{"run": "%s", "job": "%s", "node": "%s", "exit": %d, "wall_seconds": %d}\n' "$RUN" "${SLURM_JOB_ID:-}" "${SLURMD_NODENAME:-}" "$rc" "$((t1 - tstart))" | tee "$A/train_resources.json"
echo "[job] $(date -Is) exit=$rc"; exit $rc

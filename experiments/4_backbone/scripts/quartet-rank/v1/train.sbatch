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
# quartet-rank-v1 Part B (2026-11-15): the exported asym50 dRSA recipe (control drsa-v1-50-within03-asym50-w064-s17, drsa-within-v1
# train.sbatch step drsa, flags identical) + --quartet-weight L: native-pairing quartet loss mean softplus(1 - (s_ij + s_kl - s_il -
# s_kj)) on the UNMASKED free-branch aux_redge logits, 128 quartets per complex visit (64 general + 64 charge-class matched; crystal
# <= 10 A contact table), full gradient into the encoder (production_residue.quartet_*). History: pinder_quartet_softplus /
# _accuracy / _covered per epoch, quartet_counts.
#   q03 -> quartet-v1-50-asym50-q03-w064-s$SEED (L 0.3)     q10 -> quartet-v1-50-asym50-q10-w064-s$SEED (L 1.0)
# Optional GRADNORMS=1 adds --quartet-grad-norms (debug_quartet_grad_norms on the last batch of every epoch; recompiles per bucket).
# Code = snapshot quartet-rank-v1 = drsa-within-v1/src + production_{residue,loading,train}.py quartet edits.
# Bus-error guard (2026-10-21 node faults): exit 135 within 120 s -> retry x3, then exclude the node and requeue (<= 4).
#   sbatch -J qr_q03_s17 --export=ALL,STEP=q03,SEED=17 train.sbatch
#   sbatch -J qr_q10_s17 --export=ALL,STEP=q10,SEED=17 train.sbatch
set -uo pipefail
source /scratch/u6xq/hussain.u6xq/_submission/jaxka_env.sh
STEP=${STEP:-${1:-}}
CODE=$S/_submission/quartet-rank-v1
export PKABENCH_RUNTIME=$JAXKA_RUNTIME/pkabench XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=$CODE/src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PKATRAIN_LOADER_WORKERS=15 PKATRAIN_PREFETCH_DEPTH=4 PYTHONFAULTHANDLER=1 GQT_PARAMETER_SCALING=1 PYTHONDONTWRITEBYTECODE=1
PY=$JAXKA_REPO/.venv/bin/python
ulimit -c 0; cd "$CODE"; echo "[job] $(date -Is) ${SLURMD_NODENAME:-} restart=${SLURM_RESTART_COUNT:-0} step=$STEP"
sha256sum "$CODE"/src/pkatrain/production_{train,loading,residue,position}.py "$0" 2>/dev/null || true
SEED=${SEED:-17}
STORE=${MASK_STORE:-$PKABENCH_RUNTIME/training/gqt-production-v2/pinder/store-hotspot-v1}
case "$STEP" in
  q03) QW=0.3 ;;
  q10) QW=1.0 ;;
  *) echo "unknown step '$STEP' (q03 | q10)" >&2; exit 2 ;;
esac
RUN=quartet-v1-50-asym50-$STEP-w064-s$SEED
BASE=(--fraction 0.5 --seed "$SEED" --shift-weight none --batch 16 --width 64 --dropout 0.1 --aux-weight 1 --aux-heads burial --orientation --dihedrals
      --variant relations --relation-maps 4,1 --all-residue-tokens --token-cutoff 15 --drop-bonded-cys --gate-weight 0.1 --gate-cys-weight 10
      --residue-edge-weight 0.1 --residue-burial-siamese --decoy-retype-weight 0 --within-weight 0.3
      --mask-oracle pesto --mask-store "$STORE" --mask-sides one --mask-prob 0.5 --mask-keep 0.5 --within-target drsa)
ARM=(--quartet-weight "$QW" --quartet-count 128)
[ "${GRADNORMS:-0}" = 1 ] && ARM+=(--quartet-grad-norms)
TRAIN=("$PY" -m pkatrain.production_train train)
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
if [ ! -f "$STORE/verification.json" ]; then echo "[job] oracle store $STORE not built" >&2; exit 3; fi
A=$PKABENCH_RUNTIME/audits/quartet-rank-v1/$STEP; [ "$SEED" = 17 ] || A=$A-s$SEED; mkdir -p "$A"
echo "[job] run=$RUN ${ARM[*]}"
tstart=$(date +%s)
gpu "${TRAIN[@]}" "$RUN" "${BASE[@]}" "${ARM[@]}"
rc=$?; t1=$(date +%s)
printf '{"run": "%s", "job": "%s", "node": "%s", "exit": %d, "wall_seconds": %d, "quartet_weight": "%s"}\n' \
  "$RUN" "${SLURM_JOB_ID:-}" "${SLURMD_NODENAME:-}" "$rc" "$((t1 - tstart))" "$QW" | tee "$A/train_resources.json"
echo "[job] $(date -Is) exit=$rc"; exit $rc

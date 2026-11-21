#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=72
#SBATCH --mem-per-cpu=2G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/u6xq/hussain.u6xq/_logs/%x_%j.out
#SBATCH --error=/scratch/u6xq/hussain.u6xq/_logs/%x_%j.err
# quartet-rank-v2 (2026-11-19): scoring on PINDER val (CPU jax), outputs audits/quartet-rank-v2/score/$ARM[-sSEED]/$STEP.
#   ARM qc | conly -> run pool5-v2-50-asym50-$ARM-w064-s$SEED;  control | q03 -> pool5-v1-50-asym50-$ARM-w064-s$SEED (read-only)
#   STEP masked2 | residue | quartet: the UNCHANGED quartet-rank-v1 scorers (as abdata-v5 score_p5.sbatch; code $CODE1). For control / q03 an
#        existing abdata-v5 report is reused (the job exits 0 without scoring).
#   STEP chain: score_chain.py (this snapshot's jobs/ + src) for any ARM.
#   STEP compile: compile.py -> audits/quartet-rank-v2/summary.md (summary-sSEED.md for other seeds); needs every report above.
#   sbatch -J q2_chain_control --export=ALL,STEP=chain,ARM=control score.sbatch
set -euo pipefail
source /scratch/u6xq/hussain.u6xq/_submission/quartet-rank-v2/jobs/common.sh
STEP=${STEP:?}; SEED=${SEED:-17}; ARM=${ARM:-}
export JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=8
ulimit -c 0; echo "[job] $(date -Is) ${SLURMD_NODENAME:-} step=$STEP arm=$ARM seed=$SEED"
Q=$R/audits/quartet-rank-v2; sfx=""; [ "$SEED" = 17 ] || sfx="-s$SEED"
if [ "$STEP" = compile ]; then
  cd "$CODE/jobs"; sha256sum compile.py "$0"
  "$PY" compile.py --seed "$SEED" > "$Q/summary$sfx.md"; cat "$Q/summary$sfx.md"; exit 0
fi
case "$ARM" in
  qc|conly) RUN=pool5-v2-50-asym50-$ARM-w064-s$SEED ;;
  control|q03) RUN=pool5-v1-50-asym50-$ARM-w064-s$SEED ;;
  *) echo "unknown arm '$ARM' (qc | conly | control | q03)" >&2; exit 2 ;;
esac
[ -f "$R/training/gqt-production-v2/runs/$RUN/selection.json" ] || { echo "[job] $RUN not finished" >&2; exit 3; }
OUT=$Q/score/$ARM$sfx/$STEP; echo "[job] run=$RUN out=$OUT"
case "$STEP" in
  masked2|residue|quartet)
    old=$R/audits/abdata-v5/score/$ARM$sfx/$STEP/report.json
    if [ "$ARM" = control ] || [ "$ARM" = q03 ]; then
      if [ -f "$old" ]; then echo "[job] reusing $old"; exit 0; fi; fi
    mkdir -p "$OUT"; cd "$CODE1"; export PYTHONPATH=$CODE1:$CODE1/src; sha256sum "$CODE1"/score_{masked2,residue,quartet}.py "$0"
    case "$STEP" in
      masked2) "$PY" score_masked2.py "$OUT" "$P/store-hotspot-v1" "$ARM=$RUN" --modes unmasked ;;
      residue) "$PY" score_residue.py "$OUT" "$ARM=$RUN" ;;
      quartet) "$PY" score_quartet.py "$OUT" "$ARM=$RUN" ;;
    esac ;;
  chain)
    mkdir -p "$OUT"; cd "$CODE/jobs"; export PYTHONPATH=$CODE/jobs:$CODE/src; sha256sum score_chain.py "$0"
    "$PY" score_chain.py "$OUT" "$ARM=$RUN" ;;
  *) echo "unknown step '$STEP'" >&2; exit 2 ;;
esac
echo "[job] $(date -Is) done"

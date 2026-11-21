#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=72
#SBATCH --mem-per-cpu=2G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/u6xq/hussain.u6xq/_logs/%x_%j.out
#SBATCH --error=/scratch/u6xq/hussain.u6xq/_logs/%x_%j.err
# quartet-rank-v1 scoring (2026-11-15), CPU jax (cpu_attention.py), selected checkpoint per selection.json, PINDER val (800).
# Scorers score_masked2.py / score_residue.py / score_decoy.py / cpu_attention.py = the drsa-within-v1 copies (the control's
# scorers, unchanged); score_graft.py / score_quartet.py / compile*.py = this snapshot. Outputs under $Q = audits/quartet-rank-v1.
# Part A (ARM ignored): the 6 pairing-rank-v1 free edge heads + the edge-contrib-v1 baseline-edge.npz parity graft, each grafted
# in memory onto the control checkpoint (score_graft.py; no run folder is written) -> $Q/graft
#   graft    score_masked2.py --modes unmasked on parity + graft-{w0p0,w0p3}-h{17,29,43}         -> $Q/graft/masked2 (+ graft.json)
#   graftq   score_quartet.py on the same 7 (one embedding pass, heads swapped)                  -> $Q/graft/quartet
#   graftc   compile_graft.py (parity check + table)                                             -> $Q/graft/summary.md
#   parta    graft, graftq, graftc in sequence (one job)
# Part B (ARM = q03 | q10, SEED default 17 -> run quartet-v1-50-asym50-$ARM-w064-s$SEED; ARM=control for STEP=quartet only):
#   masked2  score_masked2.py --modes unmasked                                                   -> $Q/$ARM/masked2
#   residue  score_residue.py (pKa clean)                                                        -> $Q/$ARM/residue
#   quartet  score_quartet.py (val quartet accuracy, free tokens)                                -> $Q/$ARM/quartet
#   compile  compile.py --arms q03,q10 (control = drsa-within-v1 reports, no re-scoring)         -> $Q/summary.md
# Non-17 seeds write to $Q/$ARM-s$SEED (control: $Q/control-s$SEED).
#   sbatch -J qr_parta --time=02:00:00 --export=ALL,STEP=parta score.sbatch
#   sbatch -J qr_masked2_q03 --export=ALL,STEP=masked2,ARM=q03 score.sbatch      (residue / quartet likewise; ARM=q10)
#   sbatch -J qr_quartet_control --time=01:00:00 --export=ALL,STEP=quartet,ARM=control score.sbatch
#   sbatch -J qr_compile --time=00:15:00 --export=ALL,STEP=compile score.sbatch
set -euo pipefail
source /scratch/u6xq/hussain.u6xq/_submission/jaxka_env.sh
STEP=${STEP:-${1:-}}; ARM=${ARM:-}; SEED=${SEED:-17}
CODE=$S/_submission/quartet-rank-v1
export PKABENCH_RUNTIME=$JAXKA_RUNTIME/pkabench PYTHONPATH=$CODE:$CODE/src OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1
export PKATRAIN_LOADER_WORKERS=15 PKATRAIN_PREFETCH_DEPTH=4 PYTHONFAULTHANDLER=1 GQT_PARAMETER_SCALING=1 PYTHONUNBUFFERED=1
PY=$JAXKA_REPO/.venv/bin/python
ulimit -c 0; cd "$CODE"; echo "[job] $(date -Is) ${SLURMD_NODENAME:-} step=$STEP arm=$ARM seed=$SEED"; sha256sum "$CODE"/*.py "$0"
Q=$PKABENCH_RUNTIME/audits/quartet-rank-v1; G=$Q/graft
STORE=${MASK_STORE:-$PKABENCH_RUNTIME/training/gqt-production-v2/pinder/store-hotspot-v1}
[ -f "$STORE/verification.json" ] || { echo "[job] store $STORE not built" >&2; exit 3; }
CTRL=drsa-v1-50-within03-asym50-w064-s17
HEADS=$PKABENCH_RUNTIME/audits/pairing-rank-v1/main/models; BASELINE=$PKABENCH_RUNTIME/audits/edge-contrib-v1/baseline-edge.npz
GRAFTS=("parity=$CTRL@$BASELINE")
for s in 17 29 43; do for w in w0p0 w0p3; do GRAFTS+=("graft-$w-h$s=$CTRL@$HEADS/seed-$s/$w/params.npz"); done; done
part_a() {
  case "$1" in
    graft) for g in "${GRAFTS[@]}"; do f=${g#*@}; [ -f "$f" ] || { echo "[job] missing head $f" >&2; exit 3; }; done; sha256sum "${GRAFTS[@]#*@}"
           mkdir -p "$G"; "$PY" score_graft.py "$G/masked2" "$STORE" "${GRAFTS[@]}" --modes unmasked ;;
    graftq) mkdir -p "$G"; "$PY" score_quartet.py "$G/quartet" "${GRAFTS[@]}" ;;
    graftc) tmp="$G/summary.md.tmp.${SLURM_JOB_ID:-$$}"
            "$PY" compile_graft.py --graft-dir "$G" > "$tmp" || { rc=$?; rm -f "$tmp"; echo "[job] compile_graft.py exit=$rc" >&2; exit $rc; }
            mv "$tmp" "$G/summary.md"; cat "$G/summary.md" ;;
  esac
}
case "$STEP" in
  graft|graftq|graftc) part_a "$STEP"; exit 0 ;;
  parta) part_a graft; part_a graftq; part_a graftc; exit 0 ;;
  compile) tmp="$Q/summary.md.tmp.${SLURM_JOB_ID:-$$}"; mkdir -p "$Q"
           "$PY" compile.py --arms q03,q10 --seed "$SEED" > "$tmp" || { rc=$?; rm -f "$tmp"; echo "[job] compile.py exit=$rc, no summary written" >&2; exit $rc; }
           out="$Q/summary.md"; [ "$SEED" = 17 ] || out="$Q/summary-s$SEED.md"; mv "$tmp" "$out"; cat "$out"; exit 0 ;;
esac
case "$ARM" in
  q03|q10) RUN=quartet-v1-50-asym50-$ARM-w064-s$SEED ;;
  control) RUN=drsa-v1-50-within03-asym50-w064-s$SEED
           [ "$STEP" = quartet ] || { echo "ARM=control only for STEP=quartet (control masked2 / residue are in audits/drsa-within-v1)" >&2; exit 2; } ;;
  *) echo "unknown arm '$ARM' (q03 | q10 | control)" >&2; exit 2 ;;
esac
A=$Q/$ARM; [ "$SEED" = 17 ] || A=$Q/$ARM-s$SEED
[ -f "$PKABENCH_RUNTIME/training/gqt-production-v2/runs/$RUN/selection.json" ] || { echo "[job] $RUN has no selection.json (training not finished?)" >&2; exit 3; }
mkdir -p "$A"; echo "[job] run=$RUN out=$A"
case "$STEP" in
  masked2) "$PY" score_masked2.py "$A/masked2" "$STORE" "$ARM=$RUN" --modes unmasked ;;
  residue) "$PY" score_residue.py "$A/residue" "$ARM=$RUN" ;;
  quartet) "$PY" score_quartet.py "$A/quartet" "$ARM=$RUN" ;;
  *) echo "unknown step '$STEP'" >&2; exit 2 ;;
esac

#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=72
#SBATCH --mem-per-cpu=2G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/u6xq/hussain.u6xq/_logs/%x_%j.out
#SBATCH --error=/scratch/u6xq/hussain.u6xq/_logs/%x_%j.err
# abdata-v5 (2026-11-18): score the pool-v5 runs with the UNCHANGED quartet-rank-v1 scorers (val set = pool-v4 val, identical).
#   ARM = control | q03 -> run pool5-v1-50-asym50-$ARM-w064-s$SEED; STEP = masked2 | residue | quartet -> audits/abdata-v5/score/$ARM[-sSEED]
set -euo pipefail
source /scratch/u6xq/hussain.u6xq/_submission/jaxka_env.sh
STEP=${STEP:?}; ARM=${ARM:?}; SEED=${SEED:-17}
CODE=$S/_submission/quartet-rank-v1
export PKABENCH_RUNTIME=$JAXKA_RUNTIME/pkabench PYTHONPATH=$CODE:$CODE/src OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1
export PKATRAIN_LOADER_WORKERS=15 PKATRAIN_PREFETCH_DEPTH=4 PYTHONFAULTHANDLER=1 GQT_PARAMETER_SCALING=1 PYTHONUNBUFFERED=1
PY=$JAXKA_REPO/.venv/bin/python
ulimit -c 0; cd "$CODE"; echo "[job] $(date -Is) ${SLURMD_NODENAME:-} step=$STEP arm=$ARM seed=$SEED"; sha256sum "$CODE"/*.py "$0"
STORE=$PKABENCH_RUNTIME/training/gqt-production-v2/pinder/store-hotspot-v1
case "$ARM" in control|q03) RUN=pool5-v1-50-asym50-$ARM-w064-s$SEED ;; *) echo "unknown arm $ARM" >&2; exit 2 ;; esac
A=$PKABENCH_RUNTIME/audits/abdata-v5/score/$ARM; [ "$SEED" = 17 ] || A=$A-s$SEED
[ -f "$PKABENCH_RUNTIME/training/gqt-production-v2/runs/$RUN/selection.json" ] || { echo "[job] $RUN not finished" >&2; exit 3; }
mkdir -p "$A"; echo "[job] run=$RUN out=$A"
case "$STEP" in
  masked2) "$PY" score_masked2.py "$A/masked2" "$STORE" "$ARM=$RUN" --modes unmasked ;;
  residue) "$PY" score_residue.py "$A/residue" "$ARM=$RUN" ;;
  quartet) "$PY" score_quartet.py "$A/quartet" "$ARM=$RUN" ;;
  *) echo "unknown step $STEP" >&2; exit 2 ;;
esac

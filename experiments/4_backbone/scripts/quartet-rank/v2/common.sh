# quartet-rank-v2 jobs (2026-11-19): the pool-v5 asym50 dRSA recipe (= quartet-rank-v1b jobs/common.sh, unchanged) on the v2 snapshot.
# Sourced by smoke.sbatch / train.sbatch / score.sbatch.
source /scratch/u6xq/hussain.u6xq/_submission/jaxka_env.sh
CODE=$S/_submission/quartet-rank-v2; CODE1=$S/_submission/quartet-rank-v1; R=$JAXKA_RUNTIME/pkabench; P=$R/training/gqt-production-v2/pinder
export PKABENCH_RUNTIME=$R OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PKATRAIN_LOADER_WORKERS=15 PKATRAIN_PREFETCH_DEPTH=4
export PYTHONFAULTHANDLER=1 GQT_PARAMETER_SCALING=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
PY=$JAXKA_REPO/.venv/bin/python
STORE_V2=$P/store-hotspot-v2          # superset of store-hotspot-v1 (bitwise equal on shared records, pesto-full-v1 verify): pool-v5's 50% subset
POOL5=(--pinder-manifest "$R/training/pool-v5/pinder-manifest.json" --pinder-extra "$P/abag-v5" --mask-store-extra "$P/store-hotspot-v5abag")
recipe() {  # recipe MASK_STORE
  echo --fraction 0.5 --shift-weight none --batch 16 --width 64 --dropout 0.1 --aux-weight 1 --aux-heads burial --orientation --dihedrals \
       --variant relations --relation-maps 4,1 --all-residue-tokens --token-cutoff 15 --drop-bonded-cys --gate-weight 0.1 --gate-cys-weight 10 \
       --residue-edge-weight 0.1 --residue-burial-siamese --decoy-retype-weight 0 --within-weight 0.3 \
       --mask-oracle pesto --mask-store "$1" --mask-sides one --mask-prob 0.5 --mask-keep 0.5 --within-target drsa
}
# chain quartet flags: CHAINW (required by train.sbatch), ABAGMIN (default 2 = the brief), count 16, tau 1, k 32 (brief)
chain_flags() {  # chain_flags WEIGHT
  echo --chain-quartet-weight "$1" --chain-quartet-count 16 --chain-quartet-tau 1 --chain-quartet-k 32 --chain-quartet-abag-min "${ABAGMIN:-2}"
}
gpu() {  # bus-error guard (2026-10-21 node faults): exit 135 within 120 s -> retry x3, then (train only, REQUEUE=1) exclude the node and requeue (<= 4)
  for attempt in 1 2 3; do
    t0=$SECONDS; rc=0; "$@" || rc=$?
    [ $rc -eq 135 ] && [ $((SECONDS - t0)) -lt 120 ] || return $rc
    echo "[job] bus error at startup, attempt $attempt" >&2; sleep 20
  done
  if [ "${REQUEUE:-0}" = 1 ] && [ "${SLURM_RESTART_COUNT:-0}" -lt 4 ]; then
    ex=$(scontrol show job "$SLURM_JOB_ID" | sed -n 's/.*ExcNodeList=\([^ ]*\).*/\1/p'); [ "$ex" = "(null)" ] && ex=""
    scontrol update job="$SLURM_JOB_ID" ExcNodeList="${ex:+$ex,}$SLURMD_NODENAME"; echo "[job] requeue, excluding $SLURMD_NODENAME" >&2
    scontrol requeue "$SLURM_JOB_ID"; sleep 60
  fi
  return 135
}

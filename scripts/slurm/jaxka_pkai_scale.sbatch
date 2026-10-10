#!/bin/bash
#SBATCH --job-name=jaxka_pkai_scale
#SBATCH --output=/scratch/u6xq/hussain.u6xq/_logs/%x_%A_%a.out
#SBATCH --error=/scratch/u6xq/hussain.u6xq/_logs/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=72
#SBATCH --mem-per-cpu=2G
#SBATCH --time=06:00:00
# Experiment 48 (pkatrain.pkai_joint_scale) on Isambard-AI: one GH200 (72 cores) per job, pKAI uv venv, dataset images
# mounted by scripts/sqfs_run.sh. Actions:
#   sbatch jaxka_pkai_scale.sbatch feature-smoke | register | store-ids | pytest
#   sbatch jaxka_pkai_scale.sbatch import RUN_DIR {pkpdb,pinder}    (an earlier run's per-record files -> one store shard)
#   sbatch [--array=0-(J-1)] jaxka_pkai_scale.sbatch prepare {pkpdb,pinder}
#       runs J x 72 store shard tasks in total, 72 single-threaded processes per job (task = job index * 72 + slot)
#   sbatch jaxka_pkai_scale.sbatch pack-store {pkpdb,pinder} | compare-packed
#   sbatch jaxka_pkai_scale.sbatch train {backbone,full} {pkpdb,joint} [batch epochs]
#   PKAI_ENCODING=atom16 PKAI_FRACTION=100 sbatch jaxka_pkai_scale.sbatch export RUN_DIR [DESTINATION.pt]
# Features go to the per-encoding store training/pkai-features-v2/<PKAI_ENCODING>/ (zstd records, a few files);
# PKAI_FRACTION / PKAI_ENCODING select the run.
set -euo pipefail
export JAXKA_CHANNEL=pkai
source /scratch/u6xq/hussain.u6xq/_submission/jaxka_env.sh
export JAXKA_REPO=${PKAI_REPO:-$JAXKA_REPO}
export PYTHONPATH="$JAXKA_REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PKABENCH_RUNTIME=$JAXKA_RUNTIME/pkabench PKAI_THREADS=1
export TMPDIR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p "$TMPDIR"; trap 'rm -rf "$TMPDIR"' EXIT
PY=$UV_PROJECT_ENVIRONMENT/bin/python
cd "$JAXKA_REPO"; action=$1; shift
echo "commit $(git rev-parse --short HEAD) action $action $* job ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0} cpus $SLURM_CPUS_PER_TASK"
case $action in
  prepare)
    dataset=$1; per=$SLURM_CPUS_PER_TASK; jobs=${SLURM_ARRAY_TASK_COUNT:-1}; index=${SLURM_ARRAY_TASK_ID:-0}
    total=$(( per * jobs )); first=$(( index * per ))
    s=$(date +%s)
    scripts/sqfs_run.sh bash -c "seq $first $(( first + per - 1 )) | xargs -P $per -I{} $PY -m pkatrain.pkai_joint_scale prepare $dataset {} $total > $TMPDIR/prepare.log 2>&1; rc=\$?; tail -40 $TMPDIR/prepare.log; exit \$rc"
    echo "prepare $dataset tasks $first..$(( first + per - 1 )) of $total seconds=$(( $(date +%s) - s ))" ;;
  train)
    $PY -m pkatrain.pkai_joint_scale train "$@" ;;
  feature-smoke|register|store-ids|build-validation|build-interface)
    scripts/sqfs_run.sh $PY -m pkatrain.pkai_joint_scale "$action" ;;
  import|pack-store|compare-packed|rescore|export)
    $PY -m pkatrain.pkai_joint_scale "$action" "$@" ;;
  pytest)
    $PY -m pytest -q tests/test_pkai_feature_store.py "$@" ;;
  *) echo "unknown action $action" >&2; exit 2 ;;
esac

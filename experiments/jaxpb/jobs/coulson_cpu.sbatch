#!/bin/bash
#SBATCH --partition=amd96
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=2G
#SBATCH --output=/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/tierB/jaxpb-pipeline/logs/%x-%A_%a.out
# PB-track CPU stand-in runs (JAX 0.11.1 CPU from the existing ogqt-main env; no installs).
# usage: sbatch -J jpp-<name> -t HH:MM:SS [-a ...] coulson_cpu.sbatch <code_dir> <script.py> [args...]
#        the token @TASK@ in args is replaced by $SLURM_ARRAY_TASK_ID
set -uo pipefail
B=/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench
PY=$B/envs/ogqt-main-7c3e3e9/bin/python
CODE=$1; shift
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=$CODE/src JAX_PLATFORMS=cpu
export XLA_FLAGS="--xla_force_host_platform_device_count=1" OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
args=(); for a in "$@"; do args+=("${a//@TASK@/${SLURM_ARRAY_TASK_ID:-0}}"); done
echo "[job] $(date -Is) $(hostname) code $CODE args ${args[*]}"
cd $CODE
if [[ "${args[0]}" == *.sh ]]; then bash "${args[@]}"; else $PY "${args[@]}"; fi
rc=$?
rm -rf /tmp/jaxpb-cc-${SLURM_JOB_ID}* 2>/dev/null
echo "[job] exit $rc $(date -Is)"

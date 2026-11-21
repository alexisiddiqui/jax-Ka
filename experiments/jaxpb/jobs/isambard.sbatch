#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=72
#SBATCH --mem-per-cpu=2G
#SBATCH --output=/scratch/u6xq/hussain.u6xq/_runtime/jax-Ka/pkabench/audits/jaxpb-pipeline/logs/%x_%j.out
# jaxpb-pipeline PB track. usage: sbatch -J jpp-<name> -t HH:MM:SS isambard.sbatch <code_dir> <script.py> [args...]
# <code_dir> = $S/_submission/jaxpb-pipeline-dev (dev) or a frozen $S/_submission/jaxpb-pipeline-<milestone>
set -uo pipefail
source /scratch/u6xq/hussain.u6xq/_submission/jaxka_env.sh
PY=$JAXKA_REPO/.venv/bin/python
CODE=$1; shift
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=72 OPENBLAS_NUM_THREADS=8
export PYTHONPATH=$CODE/src
echo "[job] $(date -Is) ${SLURMD_NODENAME:-} code $CODE args $*"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd $CODE
"$PY" "$@"
rc=$?
rm -rf /tmp/jaxpb-cc-${SLURM_JOB_ID}* 2>/dev/null
echo "[job] exit $rc $(date -Is)"

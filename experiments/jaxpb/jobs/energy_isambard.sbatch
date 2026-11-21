#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=72
#SBATCH --mem-per-cpu=2G
#SBATCH --output=/scratch/u6xq/hussain.u6xq/_runtime/jax-Ka/pkabench/audits/jaxpb-pipeline/logs/%x_%j.out
# jaxpb-pipeline energy track (assemble/titrate) on one GH200. usage:
#   sbatch -J jpe-<name> -t HH:MM:SS energy_isambard.sbatch <code_dir> <script.py> <args...> [-- <script.py> <args...>]...
# Each '--'-separated command runs as its own python process (from <code_dir>/tests), all sharing one node-local
# persistent compile cache, so a later command measures compiles in a fresh process. The cache is removed at exit.
# M4: ... <code> test_titrate.py m4 $R/audits/jaxpb-pipeline/M4 -- test_titrate.py cachecheck $R/audits/jaxpb-pipeline/M4
# M3: JAXPB_MCE=$R/audits/jaxpb-pipeline/M3/inputs/pilot5-mce.npz sbatch ... <code> test_assemble.py m3 \
#       $R/audits/jaxpb-pipeline/M3 $R/audits/jaxpb-pilot-v1/ref $R/audits/mf-surrogate-v2/states-v2.npz
set -uo pipefail
source /scratch/u6xq/hussain.u6xq/_submission/jaxka_env.sh
PY=$JAXKA_REPO/.venv/bin/python
export R=$S/_runtime/jax-Ka/pkabench
CODE=$1; shift
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=/tmp/jaxpb-energy-$SLURM_JOB_ID
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
trap 'rm -rf "$JAX_COMPILATION_CACHE_DIR"' EXIT
echo "[job] $(date -Is) ${SLURMD_NODENAME:-} code $CODE args $*"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd "$CODE/tests"
rc=0; cmd=()
run() { echo "[job] run ${cmd[*]} $(date -Is)"; "$PY" "${cmd[@]}" || rc=$?; cmd=(); }
for a in "$@"; do
  if [ "$a" = "--" ]; then run; else cmd+=("$a"); fi
done
[ ${#cmd[@]} -gt 0 ] && run
echo "[job] cache files $(find "$JAX_COMPILATION_CACHE_DIR" -type f | wc -l)"
echo "[job] exit $rc $(date -Is)"
exit $rc

#!/bin/bash
#SBATCH --partition=oc
#SBATCH --nodelist=comp1400
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=2G
#SBATCH --output=/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/tierB/jaxpb-pipeline/a40/logs/%x-%j.out
# GPU closing tests on ONE A40 (Coulson comp1400) while Isambard is out of credits.
# usage: sbatch -J jpa-<name> -t HH:MM:SS [--dependency=afterany:<id>] a40.sbatch <frozen code dir> <jobs/a40_<stage>.sh>
set -uo pipefail
export B=/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench
export T=$B/tierB/jaxpb-pipeline A40=$B/tierB/jaxpb-pipeline/a40
export PY=$B/envs/ogqt-main-7c3e3e9/bin/python
export CODE=$1; shift
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=$CODE/jobs/a40site:$CODE/src XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export CC=/tmp/jpa-cc-$SLURM_JOB_ID
unset LD_LIBRARY_PATH
echo "[job] $(date -Is) $(hostname) gpus=${CUDA_VISIBLE_DEVICES:-?} code $CODE args $*"
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader 2>/dev/null || true
cd $CODE
bash "$@"
rc=$?
rm -rf /tmp/jaxpb-cc-${SLURM_JOB_ID}* $CC 2>/dev/null
echo "[job] exit $rc $(date -Is)"
exit $rc

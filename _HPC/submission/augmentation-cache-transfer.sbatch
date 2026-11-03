#!/bin/bash
#SBATCH --job-name=anm-feature-transfer
#SBATCH --partition=oc
#SBATCH --nodelist=comp1400
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=24:00:00
set -euo pipefail
pilot=/home/coulson/oc/lina4225/_runtime/jax-Ka/anm-pilot-v1
work="${1:?frozen transfer directory}"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"$pilot/env/bin/python" -m pytest "$work/test_augmentation_cache_transfer.py" -q
exec "$pilot/env/bin/python" "$work/augmentation_cache_transfer.py" transfer \
  --source "$pilot/feature-cache-release-v1" --work "$work" \
  --host u6xq.aip2.isambard \
  --destination /scratch/u6xq/hussain.u6xq/_runtime/jax-Ka/anm-pilot-v1/feature-cache-release-v1

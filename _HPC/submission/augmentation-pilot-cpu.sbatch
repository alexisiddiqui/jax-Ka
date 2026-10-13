#!/bin/bash
#SBATCH --job-name=anm-infra-pilot
#SBATCH --partition=amd96
#SBATCH --exclude=comp1400
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --time=02:00:00
#SBATCH --output=/home/coulson/oc/lina4225/_runtime/jax-Ka/anm-pilot-v1/infrastructure-%j.log
set -euo pipefail
repo=/home/coulson/oc/lina4225/jax-Ka
pilot=/home/coulson/oc/lina4225/_runtime/jax-Ka/anm-pilot-v1/native-large
release=/home/coulson/oc/lina4225/_runtime/jax-Ka/anm-pilot-v1/infrastructure-v1
python=/home/coulson/oc/lina4225/_runtime/jax-Ka/anm-pilot-v1/env/bin/python
export PYTHONPATH="$repo/src" JAX_PLATFORMS=cpu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENMM_CPU_THREADS=1
"$python" -m pytest "$repo/tests/test_augmentation_release.py" -q
"$python" -m pkatrain.augmentation_pilot pack "$pilot" "$release/coordinates"
"$python" -m pkatrain.augmentation_evidence "$release/coordinates" "$release/evidence" "$pilot"
"$python" -m pkatrain.augmentation_pilot graphs "$release/coordinates" "$release/graphs" \
  --runtime /home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench --atlas-tar "$pilot/atlas-records.tar"
"$python" -m pkatrain.augmentation_preflight reads "$release/graphs" "$release/host-read-preflight.json" \
  --workers 8 --prefetch 2 4 --batch 4
"$python" -m pkatrain.augmentation_report "$release" "$pilot"

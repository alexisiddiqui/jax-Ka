#!/bin/bash
#SBATCH --job-name=install-ogqt-triton
#SBATCH --partition=oc
#SBATCH --nodelist=comp1400
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:20:00
#SBATCH --output=/home/coulson/oc/lina4225/_runtime/jax-Ka/logs/install-ogqt-triton-%j.out

set -euo pipefail
destination=/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/overlays/ogqt-triton-v1
mkdir -p "$destination"
python=/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/envs/train-shared-gpu-v1/bin/python
$python -m ensurepip
$python -m pip install --target "$destination" --no-deps \
  jax-triton==0.3.0 triton==3.3.0

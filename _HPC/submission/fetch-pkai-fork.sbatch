#!/bin/bash
#SBATCH --job-name=fetch-pkai-fork
#SBATCH --partition=oc
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:20:00
#SBATCH --output=/home/coulson/oc/lina4225/_runtime/jax-Ka/logs/fetch-pkai-fork-%j.out

set -euo pipefail
/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/envs/runner/bin/python \
  /home/coulson/oc/lina4225/jax-Ka/_HPC/submission/fetch-pkai-fork.py

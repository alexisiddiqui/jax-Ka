#!/bin/bash
#SBATCH --job-name=pkai-feature-reg
#SBATCH --partition=oc
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:20:00
#SBATCH --output=/home/coulson/oc/lina4225/_runtime/jax-Ka/logs/pkai-feature-register-%j.out

set -euo pipefail
export PYTHONPATH=/home/coulson/oc/lina4225/jax-Ka/src
/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/envs/runner/bin/python -m pkabench.pkai_feature_pinder_eval register \
  --runtime /home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench

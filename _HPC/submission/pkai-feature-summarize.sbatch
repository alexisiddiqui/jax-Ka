#!/bin/bash
#SBATCH --job-name=pkai-feature-sum
#SBATCH --partition=oc
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:20:00
#SBATCH --output=/home/coulson/oc/lina4225/_runtime/jax-Ka/logs/pkai-feature-summarize-%j.out

set -euo pipefail
export PYTHONPATH=/home/coulson/oc/lina4225/jax-Ka/src
action=${PKAI_SUMMARY_ACTION:-summarize}
/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/envs/runner/bin/python -m pkabench.pkai_feature_pinder_eval "$action" \
  --runtime /home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench

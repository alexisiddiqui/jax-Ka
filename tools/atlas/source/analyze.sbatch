#!/bin/bash
#SBATCH --job-name=atlas_analyze
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/u6xq/hussain.u6xq/_logs/atlas_analyze_%j.out
#SBATCH --error=/scratch/u6xq/hussain.u6xq/_logs/atlas_analyze_%j.err
set -euo pipefail
source /scratch/u6xq/hussain.u6xq/_submission/jaxka_env.sh
exec "$JAXKA_REPO/.venv/bin/python" "$JAXKA_RUNTIME/pkabench/audits/atlas-v1/analyze_overlap.py" "$@"

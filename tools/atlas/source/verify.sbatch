#!/bin/bash
#SBATCH --job-name=atlas_verify
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/u6xq/hussain.u6xq/_logs/atlas_verify_%j.out
#SBATCH --error=/scratch/u6xq/hussain.u6xq/_logs/atlas_verify_%j.err
set -euo pipefail
source /scratch/u6xq/hussain.u6xq/_submission/jaxka_env.sh
A=$JAXKA_RUNTIME/pkabench/audits/atlas-v1
"$JAXKA_REPO/.venv/bin/python" "$A/verify_download.py" --full
"$JAXKA_REPO/.venv/bin/python" "$A/finalize_dataset.py"

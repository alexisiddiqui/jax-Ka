#!/bin/bash
#SBATCH --job-name=atlas_split_inventory
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=72
#SBATCH --mem-per-cpu=2G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/u6xq/hussain.u6xq/_logs/atlas_split_inventory_%j.out
#SBATCH --error=/scratch/u6xq/hussain.u6xq/_logs/atlas_split_inventory_%j.err
set -euo pipefail
source /scratch/u6xq/hussain.u6xq/_submission/jaxka_env.sh
export PKABENCH_RUNTIME=$JAXKA_RUNTIME/pkabench
cd "$JAXKA_REPO"
A=$PKABENCH_RUNTIME/audits/atlas-split-v1
mkdir -p "$A"
cp "$S/_transfer/atlas-split-inventory.py" "$A/inventory.py"
scripts/sqfs_run.sh .venv/bin/python "$A/inventory.py"
uv venv --python "$JAXKA_REPO/.venv/bin/python" "$A/venv"
uv pip install --python "$A/venv/bin/python" phate biopython biotite pyarrow matplotlib plotly
"$A/venv/bin/python" -m pip freeze > "$A/packages.txt" || uv pip freeze --python "$A/venv/bin/python" > "$A/packages.txt"

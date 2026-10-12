#!/bin/bash
#SBATCH --job-name=atlas_alignment_phate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=72
#SBATCH --mem-per-cpu=2G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/u6xq/hussain.u6xq/_logs/atlas_alignment_%j.out
#SBATCH --error=/scratch/u6xq/hussain.u6xq/_logs/atlas_alignment_%j.err
set -euo pipefail
source /scratch/u6xq/hussain.u6xq/_submission/jaxka_env.sh
export PKABENCH_RUNTIME=$JAXKA_RUNTIME/pkabench
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MAKEFLAGS=-j16
export TMPDIR=/tmp/$USER-atlas-alignment-$SLURM_JOB_ID
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
cd "$JAXKA_REPO"
A=$PKABENCH_RUNTIME/audits/atlas-alignment-v1
PY=$PKABENCH_RUNTIME/audits/atlas-split-v1/venv/bin/python
uv pip install --python "$PY" parasail > "$A/install.log" 2>&1
uv pip freeze --python "$PY" > "$A/packages.txt"
if [ ! -s "$A/sequences.json" ]; then
 scripts/sqfs_run.sh "$PY" "$A/extract.py"
fi
"$PY" "$A/align.py"
"$PY" "$A/embed.py"

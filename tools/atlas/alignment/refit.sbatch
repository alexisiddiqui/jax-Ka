#!/bin/bash
#SBATCH --job-name=atlas_alignment_refit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=72
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:30:00
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
"$PY" "$A/embed.py"

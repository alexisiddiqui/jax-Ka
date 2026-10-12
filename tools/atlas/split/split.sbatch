#!/bin/bash
#SBATCH --job-name=atlas_split_phate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=72
#SBATCH --mem-per-cpu=2G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/u6xq/hussain.u6xq/_logs/atlas_split_phate_%j.out
#SBATCH --error=/scratch/u6xq/hussain.u6xq/_logs/atlas_split_phate_%j.err
set -euo pipefail
source /scratch/u6xq/hussain.u6xq/_submission/jaxka_env.sh
export PKABENCH_RUNTIME=$JAXKA_RUNTIME/pkabench JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TMPDIR=/tmp/$USER-atlas-split-$SLURM_JOB_ID
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
cd "$JAXKA_REPO"
A=$PKABENCH_RUNTIME/audits/atlas-split-v1
.venv/bin/python "$A/split.py"
scripts/sqfs_run.sh "$A/venv/bin/python" "$A/features.py"
"$A/venv/bin/python" "$A/embed.py"

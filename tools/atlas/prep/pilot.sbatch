#!/bin/bash
#SBATCH --job-name=atlas_prep_pilot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=72
#SBATCH --mem-per-cpu=2G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/u6xq/hussain.u6xq/_logs/atlas_prep_pilot_%j.out
#SBATCH --error=/scratch/u6xq/hussain.u6xq/_logs/atlas_prep_pilot_%j.err
set -euo pipefail
source /scratch/u6xq/hussain.u6xq/_submission/jaxka_env.sh
export PKABENCH_RUNTIME=$JAXKA_RUNTIME/pkabench JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TMPDIR=/tmp/$USER-atlas-$SLURM_JOB_ID
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
cd "$JAXKA_REPO"
PY=$JAXKA_REPO/.venv/bin/python
A=$PKABENCH_RUNTIME/audits/atlas-prep-v1
$PY "$A/pilot.py" prep 4
$S/_install/jax-Ka-venvs/pkai/bin/python "$A/pilot.py" label 4
$PY -c 'import json,os; from pathlib import Path; p=Path(os.environ["PKABENCH_RUNTIME"])/"pretraining/atlas-prepared-v1"; assert all(r["status"]=="labelled" for r in json.loads((p/"label-4.json").read_text())["results"]), "smoke failure"'
$PY "$A/pilot.py" prep 288
$S/_install/jax-Ka-venvs/pkai/bin/python "$A/pilot.py" label 288

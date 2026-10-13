#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=2G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/u6xq/hussain.u6xq/_logs/%x_%j.out
#SBATCH --error=/scratch/u6xq/hussain.u6xq/_logs/%x_%j.err
set -euo pipefail
source /scratch/u6xq/hussain.u6xq/_submission/jaxka_env.sh
export CODE=$S/_submission/invsqrt-v2
export PKABENCH_RUNTIME=$JAXKA_RUNTIME/pkabench XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH=$CODE/src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PKATRAIN_LOADER_WORKERS=15 PKATRAIN_PREFETCH_DEPTH=4
cd "$CODE"
variant=$5
"$JAXKA_REPO/.venv/bin/python" "$S/_submission/check_largest.py" "$variant"
"$JAXKA_REPO/.venv/bin/python" -m pkatrain.production_train train "smoke-$1" --fraction "$2" --seed "$3" --shift-weight "$4" --variant "$variant" --batch 16 --width 64 --dropout .1 --aux-weight 1 --aux-heads burial --orientation --dihedrals --smoke
"$JAXKA_REPO/.venv/bin/python" -m pkatrain.production_train train "$1" --fraction "$2" --seed "$3" --shift-weight "$4" --batch 16 --width 64 --dropout 0.1 --aux-weight 1 --aux-heads burial --orientation --dihedrals --variant "$variant"

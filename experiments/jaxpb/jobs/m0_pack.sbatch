#!/bin/bash
#SBATCH --partition=amd96
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --output=/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/tierB/jaxpb-pipeline/logs/%x-%j.out
set -euo pipefail
T=/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/tierB/jaxpb-pipeline
/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/envs/runner/bin/python $T/code/jobs/m0_pack.py $T/m0/states.json $T/m0/rec $T/m0/shards 50
cd $T/m0/shards && sha256sum m0_goldens_*.npz > SHA256

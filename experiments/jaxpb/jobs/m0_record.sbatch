#!/bin/bash
#SBATCH --partition=amd96
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --output=/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/tierB/jaxpb-pipeline/logs/%x-%A_%a.out
# M0 goldens: one array task per state (index into m0/states.json). usage: sbatch -J jpp-m0rec -a 0-98%40 -t 03:00:00 m0_record.sbatch
set -euo pipefail
B=/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench
T=$B/tierB/jaxpb-pipeline
CODE=$T/code
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
read SID ROLE SRC < <($B/envs/runner/bin/python -c "import json,sys; s=json.load(open('$T/m0/states.json'))[int(sys.argv[1])]; print(s['state_id'], s['role'], s['source'])" $SLURM_ARRAY_TASK_ID)
W=$T/m0/work/$SID
echo "job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID node $(hostname) state $SID $(date -Is)"
if [ -s $T/m0/rec/$SID.npz ]; then echo "exists, skip"; exit 0; fi
mkdir -p $W $T/m0/rec
$B/envs/runner/bin/python $CODE/jobs/m0_inputpdb.py $SRC $W
export LD_LIBRARY_PATH=$B/fortran/lib
$B/envs/pypka/bin/python $CODE/src/jaxpb/prep_pypka.py record $SRC $W $T/m0/rec/$SID.npz --state-id $SID --role $ROLE --split val > $W/record.log 2>&1 || { tail -30 $W/record.log; exit 1; }
tail -40 $W/record.log | grep -E "bitwise_equal|pka_maxabs|pos_maxdiff|site_inv|pb_wall_sum|out_sha256"
# keep the work dir small: drop PyPKA's scratch outputs, keep logs/gate
find $W -maxdepth 1 -type f ! -name 'gate.json' ! -name 'record.log' ! -name 'input.pdb' -delete
echo "done $(date -Is)"

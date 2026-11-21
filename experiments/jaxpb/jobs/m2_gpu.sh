#!/bin/bash
# M2 closing test on Isambard (run via: sbatch -J jpp-m2 -t 04:00:00 jobs/isambard.sbatch $S/_submission/jaxpb-pipeline-m2 bash jobs/m2_gpu.sh)
# DelPhi-mode PB with the relfac estimator (no logged rho) on the ~100 M0 golden states vs DelPhi's recorded outputs.
set -u
source /scratch/u6xq/hussain.u6xq/_submission/jaxka_env.sh
PY=$JAXKA_REPO/.venv/bin/python
R=$S/_runtime/jax-Ka/pkabench; O=$R/audits/jaxpb-pipeline/m2; mkdir -p $O
SH=$R/audits/jaxpb-pipeline/m0/shards
$PY tests/test_pb_m2.py $O $SH/m0_goldens_*.npz --cache $R/audits/jaxpb-pipeline/m1/cc_m1.tar --specrad estimate --tag m2

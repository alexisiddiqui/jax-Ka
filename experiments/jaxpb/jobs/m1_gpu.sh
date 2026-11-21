#!/bin/bash
# M1 closing test on Isambard (run via: sbatch -J jpp-m1 -t 02:00:00 jobs/isambard.sbatch $S/_submission/jaxpb-pipeline-m1 bash jobs/m1_gpu.sh)
# main: 5 pilot states pass 1 + pass 2 (+ forced-overflow XL retry) with a fresh compile cache tarball;
# cache: a NEW process re-runs the 5 states from the tarball (expect 0 compiles).
set -u
source /scratch/u6xq/hussain.u6xq/_submission/jaxka_env.sh
PY=$JAXKA_REPO/.venv/bin/python
R=$S/_runtime/jax-Ka/pkabench; O=$R/audits/jaxpb-pipeline/m1; mkdir -p $O
G=$R/audits/jaxpb-pipeline/m0/goldens_pilot5.npz
rm -f $O/cc_m1.tar
$PY tests/test_pb_m1.py $G $O main --cache $O/cc_m1.tar
$PY tests/test_pb_m1.py $G $O cache --cache $O/cc_m1.tar

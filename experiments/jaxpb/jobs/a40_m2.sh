#!/bin/bash
# A40 stage 2: M2 closing test (relfac estimator, no logged rho) on all 99 M0 states in ONE process; then M3 on the 99 states.
set -u
O=$A40/m2; mkdir -p $O; SH=$T/m0/shards
cp $A40/m1/cc_m1.tar $O/cc_m2.tar
$PY tests/test_pb_m2.py $O $SH/m0_goldens_00.npz $SH/m0_goldens_01.npz $SH/m0_goldens_02.npz --cache $O/cc_m2.tar --specrad estimate --tag m2
$PY tests/test_assemble.py m0 $A40/m3 $SH/m0_goldens_00.npz $SH/m0_goldens_01.npz $SH/m0_goldens_02.npz

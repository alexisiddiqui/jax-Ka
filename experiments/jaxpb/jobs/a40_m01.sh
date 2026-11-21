#!/bin/bash
# A40 stage 1: M0 v2 reproduction (5 pilot states; keeps the A40 v2 outputs) + v2 from the cache tarball in a new process;
# M1 closing test vs the GH200 v2 goldens (pass 1/2 + overflow, then a new process from the tarball); then M1 like-with-like
# vs the A40 v2 outputs (goldens rebuilt with v2_* = A40 v2 run).
set -u
I=$A40/inputs; O0=$A40/m0; O1=$A40/m1; mkdir -p $O0/v2a40 $O0/v2repro $O1/vs_v2a40
G=$T/m0/pilot/goldens_pilot5.npz
rm -f $O0/cc_v2.tar $O1/cc_m1.tar
$PY jobs/m0_v2repro.py $CODE/v2pilot $I/v1ref $T/m0/pilot/v2out $O0/v2a40 final --cache $O0/cc_v2.tar --keep
$PY jobs/m0_v2repro.py $CODE/v2pilot $I/v1ref $T/m0/pilot/v2out $O0/v2repro run2cache --cache $O0/cc_v2.tar --states d0d5c619bd23b13d_B,d0d5c619bd23b13d_A
$PY tests/test_pb_m1.py $G $O1 main --cache $O1/cc_m1.tar
$PY tests/test_pb_m1.py $G $O1 cache --cache $O1/cc_m1.tar
JPP_REC_NAME="{sid}/rec.npz" $PY jobs/m0_goldens.py $B/tierB/jaxpb-pilot-v1/ref $O0/v2a40 $T/m0/pilot/pilotjson $O1/vs_v2a40/goldens_pilot5_v2a40.npz
cp $O1/cc_m1.tar $O1/vs_v2a40/cc.tar
$PY tests/test_pb_m1.py $O1/vs_v2a40/goldens_pilot5_v2a40.npz $O1/vs_v2a40 cache --cache $O1/vs_v2a40/cc.tar
rm -f $O1/vs_v2a40/cc.tar $O0/cc_v2.tar
